"""
PlantVerse — Plant Disease Detection Engine (`plant_disease_detection.py`)
==========================================================================
Integrates:
1. Primary Crop Disease Classifier: Vision Transformer (`models/model.onnx` + `models/labels.json`),
   fine-tuned on 38 PlantVillage + PlantDoc field classes (CC BY-SA 3.0).
2. Comparison / Fallback Classifier: MobileNetV2 (`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`)
   when `PLANTVERSE_DISEASE_BACKEND=mobilenet_v2`.
3. Plant -> Disease Consistency Gating (Task 4):
   - Gates disease classes strictly by the identified crop (`CROP_TO_CLASS_INDICES`).
   - If the plant is not one of the 14 supported crops, returns:
     "Disease classification is not available for this plant."
   - Impossible cross-crop combinations (e.g. Tomato + Apple cedar rust) can never occur.
4. Classical CV Affected-Area, Project-Defined Severity, and Derived Health Score (Tasks 5-6)
   via `vision_analysis.py`.
5. Provisional Confidence Policy (Task 3) & Robust Error Handling (Task 8) — no fake fallbacks.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

for _venv_sp in sorted((Path(__file__).resolve().parent / "venv" / "lib").glob("python*/site-packages")):
    if str(_venv_sp) not in sys.path:
        sys.path.append(str(_venv_sp))

import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]
from PIL import Image, ImageOps
import streamlit as st
import torch

from disease_recommendations import (
    EXPERT_REVIEW_NOTICE,
    NO_RECOMMENDATION_MESSAGE,
    RAW_LABEL_MAP,
    get_disease_info,
    normalize_disease_label,
)
import importlib as _importlib
import ml_config as _ml_config
if not hasattr(_ml_config, "load_conditions_metadata"):
    _importlib.reload(_ml_config)

from ml_config import (
    CROP_TO_CLASS_INDICES,
    DISEASE_BACKEND,
    DISEASE_HF_MODEL_NAME,
    DISEASE_HIGH_CONF_THRESHOLD,
    DISEASE_INPUT_SIZE,
    DISEASE_MEDIUM_CONF_THRESHOLD,
    DISEASE_MIN_TOP1_TOP2_MARGIN,
    DISEASE_PROVISIONAL_N,
    EXTERNAL_API_URL,
    LOCAL_KERAS_PATH,
    LOW_CONFIDENCE_DISEASE_MESSAGE,
    MAX_IMAGE_SIZE_BYTES,
    MIN_IMAGE_DIMENSION_PX,
    ONNX_DISEASE_LABELS_PATH,
    ONNX_DISEASE_MODEL_PATH,
    PLANTVILLAGE_38_RAW_CLASSES,
    SUPPORTED_DISEASE_CROPS,
    UNSUPPORTED_CROP_DISEASE_MESSAGE,
    load_conditions_metadata,
    load_plants_metadata,
    resolve_supported_disease_crop,
)
import plant_identification as _plant_identification
if not hasattr(_plant_identification, "predict_species_detailed"):
    _importlib.reload(_plant_identification)
from plant_identification import predict_species_detailed
from vision_analysis import analyze_foliar_lesions, crop_to_bbox, segment_leaf

warnings.filterwarnings("ignore")

DEFAULT_MODEL_NAME = DISEASE_HF_MODEL_NAME
CONFIDENCE_THRESHOLD = DISEASE_MEDIUM_CONF_THRESHOLD
_PIPELINE_VERSION = "v4_healthy_foliage_fix"


@st.cache_resource
def load_onnx_disease_classifier(
    model_path_str: Optional[str] = None,
    labels_path_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load the primary ViT int8 ONNX 38-class disease classifier (`models/model.onnx`) once.
    Raises RuntimeError if the model or labels file is missing or corrupted.
    """
    m_path = Path(model_path_str or os.getenv("PLANTVERSE_ONNX_DISEASE_MODEL", str(ONNX_DISEASE_MODEL_PATH)))
    l_path = Path(labels_path_str or os.getenv("PLANTVERSE_ONNX_DISEASE_LABELS", str(ONNX_DISEASE_LABELS_PATH)))

    if not m_path.exists():
        raise RuntimeError(
            f"Missing disease detection model file at '{m_path}'. "
            "Please verify that 'models/model.onnx' is present."
        )
    if m_path.stat().st_size < 1024:
        raise RuntimeError(
            f"Disease detection model file at '{m_path}' is empty or truncated ({m_path.stat().st_size} bytes)."
        )
    if not l_path.exists():
        raise RuntimeError(
            f"Missing disease labels file at '{l_path}'. "
            "Please verify that 'models/labels.json' is present."
        )

    try:
        labels = json.loads(l_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Corrupted disease labels JSON at '{l_path}': {exc}") from exc

    if labels != PLANTVILLAGE_38_RAW_CLASSES:
        raise RuntimeError(
            f"Disease labels in '{l_path}' do not match the canonical 38-class PlantVillage order."
        )

    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(str(m_path), opts, providers=["CPUExecutionProvider"])
        return {
            "type": "onnx_vit",
            "session": session,
            "labels": labels,
            "model_name": "ViT-Base ONNX (models/model.onnx, CC BY-SA 3.0)",
        }
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize ONNX disease model from '{m_path}' (weights may be corrupted): {exc}"
        ) from exc


@st.cache_resource
def load_mobilenet_disease_classifier(model_name_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the comparison/fallback PyTorch MobileNetV2 disease classifier.
    Raises RuntimeError if loading fails.
    """
    model_name = (model_name_override or os.getenv("PLANT_DISEASE_MODEL_NAME", DEFAULT_MODEL_NAME)).strip()
    try:
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            MobileNetV2ImageProcessor,
        )

        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
        except Exception:
            processor = MobileNetV2ImageProcessor.from_pretrained(model_name)

        num_labels = getattr(model.config, "num_labels", 0)
        if num_labels != len(PLANTVILLAGE_38_RAW_CLASSES):
            raise RuntimeError(
                f"Disease model '{model_name}' has {num_labels} output classes, "
                f"expected {len(PLANTVILLAGE_38_RAW_CLASSES)}."
            )

        return {
            "type": "pytorch",
            "model": model,
            "processor": processor,
            "labels": PLANTVILLAGE_38_RAW_CLASSES,
            "model_name": model_name,
        }
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load plant disease detection model '{model_name}': {exc}"
        ) from exc


def load_disease_classifier() -> Dict[str, Any]:
    """
    Load the configured disease classification backend (`onnx_vit` primary, or `mobilenet_v2`).
    Never returns a fake fallback: raises RuntimeError if the model cannot be loaded.
    """
    keras_path = os.getenv("PLANT_DISEASE_KERAS_MODEL_PATH", LOCAL_KERAS_PATH).strip()
    if keras_path and os.path.exists(keras_path):
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(keras_path)
            return {
                "type": "keras",
                "model": model,
                "labels": PLANTVILLAGE_38_RAW_CLASSES,
                "model_name": keras_path,
            }
        except Exception as err:
            raise RuntimeError(
                f"Failed to load configured local Keras disease model at '{keras_path}': {err}"
            ) from err

    backend = os.getenv("PLANTVERSE_DISEASE_BACKEND", DISEASE_BACKEND).strip().lower()
    if backend == "mobilenet_v2":
        return load_mobilenet_disease_classifier()
    return load_onnx_disease_classifier()


def preprocess_image(image_input: Any) -> Image.Image:
    """
    Validate, sanitize, apply EXIF orientation transpose, and convert an uploaded
    file, path, bytes, or PIL Image into a validated RGB PIL Image.
    Raises ValueError with descriptive human-readable errors.
    """
    if image_input is None:
        raise ValueError("No image provided. Please upload a specimen photograph.")

    try:
        if isinstance(image_input, Image.Image):
            img = ImageOps.exif_transpose(image_input.copy())
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.width < MIN_IMAGE_DIMENSION_PX or img.height < MIN_IMAGE_DIMENSION_PX:
                raise ValueError(
                    f"Image is too small ({img.width}x{img.height} px). "
                    f"Minimum required size is {MIN_IMAGE_DIMENSION_PX}x{MIN_IMAGE_DIMENSION_PX} px."
                )
            return img

        if isinstance(image_input, (str, os.PathLike)):
            if not os.path.exists(image_input):
                raise ValueError(f"Image file does not exist: {image_input}")
            size = os.path.getsize(image_input)
            if size > MAX_IMAGE_SIZE_BYTES:
                raise ValueError(
                    f"Uploaded file exceeds the maximum allowed size of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB."
                )
            if size == 0:
                raise ValueError("Uploaded file is empty (0 bytes).")
            with open(image_input, "rb") as f:
                data = f.read()
        elif hasattr(image_input, "read"):
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            data = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            if len(data) > MAX_IMAGE_SIZE_BYTES:
                raise ValueError(
                    f"Uploaded file exceeds the maximum allowed size of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB."
                )
            if len(data) == 0:
                raise ValueError("Uploaded file is empty (0 bytes).")
        elif isinstance(image_input, (bytes, bytearray)):
            if len(image_input) > MAX_IMAGE_SIZE_BYTES:
                raise ValueError(
                    f"Uploaded image exceeds the maximum allowed size of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB."
                )
            if len(image_input) == 0:
                raise ValueError("Image data is empty (0 bytes).")
            data = bytes(image_input)
        else:
            raise ValueError("Unsupported input format for image processing.")

        img_verify = Image.open(io.BytesIO(data))
        img_verify.verify()

        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width < MIN_IMAGE_DIMENSION_PX or img.height < MIN_IMAGE_DIMENSION_PX:
            raise ValueError(
                f"Image is too small ({img.width}x{img.height} px). "
                f"Minimum required size is {MIN_IMAGE_DIMENSION_PX}x{MIN_IMAGE_DIMENSION_PX} px."
            )

        return img

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Unable to process image file. The file may be corrupted or an unsupported format: {exc}"
        ) from exc


def preprocess_for_onnx_vit(image: Image.Image) -> np.ndarray:
    """
    Exact training-matched preprocessing for the ViT disease classifier (`models/model.onnx`):
    1. Resize to (224, 224) with BILINEAR interpolation.
    2. Rescale and normalize to [-1.0, 1.0] via x / 127.5 - 1.0 (mean=0.5, std=0.5).
    3. Transpose HWC -> CHW and add batch dimension [1, 3, 224, 224].
    """
    resized = image.resize(DISEASE_INPUT_SIZE, Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_v = np.exp(shifted)
    return exp_v / np.sum(exp_v)


def split_raw_disease_label(raw_label: str) -> Tuple[str, str, bool]:
    """
    Split a raw 'Crop___Condition' PlantVillage label into (crop, condition, is_healthy).
    """
    info = get_disease_info(raw_label)
    if info.get("known_disease"):
        crop = info.get("crop", "Unknown")
        condition = info.get("condition", raw_label)
        is_healthy = info.get("health_status") == "Healthy" or "___healthy" in raw_label.lower()
        return crop, condition, is_healthy

    if "___" in raw_label:
        raw_crop, raw_cond = raw_label.split("___", 1)
        crop = resolve_supported_disease_crop(raw_crop) or raw_crop.replace("_", " ").strip()
        is_healthy = raw_cond.strip().lower() == "healthy"
        condition = "Healthy Foliage" if is_healthy else raw_cond.replace("_", " ").strip()
        return crop, condition, is_healthy

    return "Unknown", raw_label, "healthy" in raw_label.lower()


def evaluate_disease_confidence_policy(
    top1_score: float,
    top2_score: float,
    high_threshold: float = DISEASE_HIGH_CONF_THRESHOLD,
    medium_threshold: float = DISEASE_MEDIUM_CONF_THRESHOLD,
    min_margin: float = DISEASE_MIN_TOP1_TOP2_MARGIN,
) -> Tuple[str, bool, str]:
    """
    Evaluate confidence tier ('high', 'medium', 'low') from measured top-1 probability
    and top-1 vs. top-2 margin (provisional thresholds derived on n=118 PlantDoc tuning images).
    Returns (tier, is_uncertain, message).
    """
    margin = top1_score - top2_score
    if top1_score >= high_threshold and margin >= min_margin:
        return "high", False, ""
    if top1_score >= medium_threshold:
        return (
            "medium",
            True,
            f"Moderate confidence diagnosis ({top1_score * 100:.1f}%; provisional threshold derived on n={DISEASE_PROVISIONAL_N} tuning images). "
            "Verify leaf symptoms against the diagnostic profile below.",
        )
    return "low", True, LOW_CONFIDENCE_DISEASE_MESSAGE


def get_supported_disease_crops_list() -> List[str]:
    """Return list of the 14 agricultural crops supported by the 38-class disease model."""
    return list(SUPPORTED_DISEASE_CROPS.keys())


def _resolve_plant_metadata(plant_name_or_id: Optional[str]) -> Dict[str, Any]:
    """Find the matching plant entry from data/plants_metadata.json by ID, common name, scientific name, or pv_crop."""
    if not plant_name_or_id:
        return {}
    plants_meta = load_plants_metadata()
    clean = plant_name_or_id.strip().lower()
    clean_id = clean.replace(" ", "_")
    if clean_id in plants_meta:
        return plants_meta[clean_id]
    for p in plants_meta.values():
        p_id = p.get("id", "").lower()
        p_name = p.get("name", "").lower()
        p_sci = p.get("scientific", "").lower()
        p_pv = (p.get("pv_crop") or "").lower()
        if clean in (p_id, p_name, p_sci, p_pv):
            return p
        if p_name and (clean.startswith(p_name.split("(")[0].strip()) or p_name.startswith(clean)):
            return p
    return {}


def _build_general_condition_diagnosis(
    image: Image.Image,
    detected_plant_name: str,
    plant_pred: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the 7-class general foliar condition & plant care diagnosis from
    `models/plant_model.onnx` (`condition_logits` head) + `data/conditions_metadata.json`
    + `data/plants_metadata.json`.
    When a leaf bounding box is found via classical CV (`segment_leaf`), evaluates the
    condition head on the cropped leaf (`crop_to_bbox`) just like FloraCare's `plant_id.py`.
    """
    try:
        img_rgb = np.asarray(image)
        seg = segment_leaf(img_rgb)
        leaf_crop_pil: Optional[Image.Image] = None
        if seg.get("found") and seg.get("bbox"):
            cropped_rgb = crop_to_bbox(img_rgb, seg["bbox"])
            if cropped_rgb.shape[:2] != img_rgb.shape[:2]:
                leaf_crop_pil = Image.fromarray(cropped_rgb)

        if leaf_crop_pil is not None:
            cond_pred_source = predict_species_detailed(image, leaf_crop=leaf_crop_pil)
        elif plant_pred is not None:
            cond_pred_source = plant_pred
        else:
            cond_pred_source = predict_species_detailed(image)
    except Exception:
        cond_pred_source = plant_pred or {}

    is_not_plant = bool(cond_pred_source.get("is_not_plant", False)) and (
        not detected_plant_name or "Unsupported / Non-Crop" in detected_plant_name
    )
    cond_list_raw = cond_pred_source.get("condition_predictions", [])
    if is_not_plant or not cond_list_raw:
        return {
            "available": False,
            "is_not_plant": is_not_plant,
        }

    conditions_meta = load_conditions_metadata()
    cond_list: List[Dict[str, Any]] = []
    for c_item in cond_list_raw:
        c_id = c_item.get("id", "healthy")
        c_meta = conditions_meta.get(c_id, {})
        c_score = float(c_item.get("score", 0.0))
        c_is_healthy = bool(c_meta.get("healthy", c_id == "healthy"))
        raw_c_name = c_meta.get("disease") or c_item.get("name") or c_id.replace("_", " ").title()
        if raw_c_name in ("Unknown Condition", "Healthy") or c_is_healthy:
            c_name = "Healthy Foliage" if c_is_healthy else c_id.replace("_", " ").title()
        else:
            c_name = raw_c_name
        raw_c_type = c_meta.get("type") or c_item.get("type") or ""
        c_type = (
            "Healthy Physiology"
            if (c_is_healthy or raw_c_type in ("", "None", "General foliar condition"))
            else raw_c_type
        )
        cond_list.append(
            {
                "id": c_id,
                "name": c_name,
                "type": c_type,
                "pathogen": c_meta.get("pathogen")
                or c_item.get("pathogen")
                or ("None — Healthy physiological state" if c_is_healthy else "Foliar pathogen / environmental stress"),
                "urgency": c_meta.get("urgency") or c_item.get("urgency") or ("none" if c_is_healthy else "medium"),
                "healthy": c_is_healthy,
                "symptoms": c_meta.get("symptoms") or c_item.get("symptoms") or "",
                "chemical": c_meta.get("chemical") if "chemical" in c_meta else c_item.get("chemical", []),
                "organic": c_meta.get("organic") if "organic" in c_meta else c_item.get("organic", []),
                "prevention": c_meta.get("prevention") if "prevention" in c_meta else c_item.get("prevention", []),
                "score": c_score,
                "confidence_pct": float(c_item.get("confidence_pct", round(c_score * 100, 1))),
            }
        )

    top_cond = cond_list[0]
    second_cond = cond_list[1] if len(cond_list) > 1 else {"score": 0.0}
    cond_score = float(top_cond.get("score", 0.0))
    cond_conf_pct = float(top_cond.get("confidence_pct", round(cond_score * 100, 1)))
    is_healthy = bool(top_cond.get("healthy", top_cond.get("id") == "healthy"))
    cond_name_clean = top_cond["name"]

    if cond_score >= 0.65:
        cond_tier = "high"
    elif cond_score >= 0.35:
        cond_tier = "medium"
    else:
        cond_tier = "low"

    species_source = (
        plant_pred
        if (plant_pred and plant_pred.get("is_reliable") and not plant_pred.get("is_not_plant"))
        else cond_pred_source
    )
    explicit_meta = (
        _resolve_plant_metadata(detected_plant_name)
        if (detected_plant_name and "Unsupported" not in detected_plant_name)
        else {}
    )
    species_reliable = bool(explicit_meta) or bool(species_source.get("is_reliable", False))

    if explicit_meta:
        p_meta = explicit_meta
    elif species_reliable:
        p_meta = (
            _resolve_plant_metadata(species_source.get("raw_label"))
            or species_source.get("plant_metadata")
            or {}
        )
    else:
        p_meta = {}

    if p_meta:
        plant_display = p_meta.get("name", detected_plant_name)
        scientific_name = p_meta.get("scientific", plant_display)
        hindi_name = p_meta.get("hindi", "")
        category = p_meta.get("category", "Botanical Specimen")
        common_problems = p_meta.get("problems", [])
        care_dict = p_meta.get("care", {})
        uses_list = p_meta.get("uses", [])
        toxic_note = p_meta.get("toxic", "")
    elif detected_plant_name and "Unsupported" not in detected_plant_name:
        plant_display = detected_plant_name
        scientific_name = detected_plant_name
        hindi_name = ""
        category = "Botanical Specimen"
        common_problems = []
        care_dict = {}
        uses_list = []
        toxic_note = ""
    else:
        plant_display = "Unidentified Plant (Low Species Confidence)"
        scientific_name = f"Tentative top match: {species_source.get('species', 'Unknown')} ({round(float(species_source.get('score', 0.0)) * 100, 1)}%)"
        hindi_name = ""
        category = "Unconfirmed Host Species"
        common_problems = []
        care_dict = {}
        uses_list = []
        toxic_note = ""

    organic_list = [x for x in top_cond.get("organic", []) if x and x != "No treatment needed"]
    chemical_list = [x for x in top_cond.get("chemical", []) if x and x != "No treatment needed"]
    prevention_list = list(top_cond.get("prevention", []))
    combined_recs: List[str] = []
    for item in organic_list + prevention_list:
        if item and item != "No treatment needed" and item not in combined_recs:
            combined_recs.append(item)

    pathogen_display = top_cond["pathogen"]

    return {
        "available": True,
        "is_not_plant": False,
        "plant_id": p_meta.get("id", species_source.get("raw_label", "")),
        "plant_name": plant_display,
        "scientific_name": scientific_name,
        "hindi_name": hindi_name,
        "category": category,
        "plant_confidence": float(species_source.get("score", 0.0)),
        "plant_confidence_pct": round(float(species_source.get("score", 0.0)) * 100, 1),
        "plant_confidence_tier": species_source.get("confidence_tier", "medium"),
        "condition_id": top_cond.get("id", "healthy"),
        "condition_name": cond_name_clean,
        "full_diagnosis_title": (
            f"{plant_display} — Healthy Foliage" if is_healthy else f"{plant_display} — {cond_name_clean}"
        ),
        "health_status": "Healthy" if is_healthy else "Diseased",
        "is_healthy": is_healthy,
        "confidence": cond_score,
        "confidence_pct": cond_conf_pct,
        "top2_confidence": float(second_cond.get("score", 0.0)),
        "confidence_tier": cond_tier,
        "type": top_cond["type"],
        "pathogen": pathogen_display,
        "urgency": top_cond.get("urgency", "none" if is_healthy else "medium"),
        "symptoms": top_cond.get("symptoms", ""),
        "recommendations": combined_recs,
        "organic_treatments": organic_list,
        "chemical_treatments": chemical_list,
        "preventive_measures": prevention_list,
        "care": care_dict,
        "common_species_problems": common_problems,
        "uses": uses_list,
        "toxic": toxic_note,
        "variegated": bool(p_meta.get("variegated", False)),
        "top_conditions": cond_list,
        "top_species_candidates": cond_pred_source.get("top_candidates", []),
        "model_name": "ConvNeXt-Tiny Dual-Head ONNX (models/plant_model.onnx — 7-class foliar condition head)",
        "expert_review_notice": EXPERT_REVIEW_NOTICE,
    }


def _build_unsupported_crop_response(
    image: Image.Image,
    image_bytes: bytes,
    detected_plant_name: str,
    plant_pred: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Return the standardized response when the identified plant is not one of the
    14 crops supported by the 38-class disease classifier (Task 4).
    - Never runs or displays a 38-class crop disease on non-crop plants (`health_status="Unsupported"`,
      `is_supported_crop=False`, `disease=UNSUPPORTED_CROP_DISEASE_MESSAGE`).
    - Simultaneously attaches `general_condition` from the 7-class foliar condition head of
      `models/plant_model.onnx` + `data/conditions_metadata.json` + `data/plants_metadata.json`
      so non-crop houseplants, trees, and medicinal plants receive accurate general foliar condition
      diagnosis, lesion segmentation, and species-specific cultural care.
    """
    gen_cond = _build_general_condition_diagnosis(
        image=image,
        detected_plant_name=detected_plant_name,
        plant_pred=plant_pred,
    )
    cond_is_healthy = bool(gen_cond.get("is_healthy", True)) if gen_cond.get("available") else True
    cond_urgency = str(gen_cond.get("urgency", "none")) if gen_cond.get("available") else "none"
    target_plant_for_cv = gen_cond.get("plant_name") or detected_plant_name

    vision_res = analyze_foliar_lesions(
        image,
        is_healthy=cond_is_healthy,
        plant_name_or_id=target_plant_for_cv,
        urgency_or_severity=cond_urgency,
    )
    return {
        "success": True,
        "plant": detected_plant_name or "Unsupported Plant",
        "disease": UNSUPPORTED_CROP_DISEASE_MESSAGE,
        "disease_name": UNSUPPORTED_CROP_DISEASE_MESSAGE,
        "health_status": "Unsupported",
        "is_supported_crop": False,
        "confidence": 0.0,
        "confidence_pct": 0.0,
        "top2_confidence": 0.0,
        "confidence_tier": "unsupported",
        "raw_label": "UNSUPPORTED_PLANT",
        "pathogen": "N/A (Plant is not one of the 14 supported agricultural crops)",
        "severity": vision_res["project_severity_category"],
        "severity_scheme_label": vision_res["severity_scheme_label"],
        "affected_area_pct": vision_res["affected_area_pct"],
        "affected_area_status": vision_res["estimation_status"],
        "derived_health_score": vision_res["derived_health_score"],
        "health_score_label": vision_res["health_score_label"],
        "symptoms": UNSUPPORTED_CROP_DISEASE_MESSAGE,
        "recommendations": [],
        "preventive_measures": [],
        "chemical_treatments": [],
        "organic_treatments": [],
        "care": gen_cond.get("care", {}),
        "general_condition": gen_cond,
        "needs_expert_review": False,
        "expert_review_notice": EXPERT_REVIEW_NOTICE,
        "no_recommendation_message": NO_RECOMMENDATION_MESSAGE,
        "is_uncertain": True,
        "uncertainty_message": UNSUPPORTED_CROP_DISEASE_MESSAGE,
        "top_candidates": [],
        "vision_analysis": vision_res,
        "quality_warnings": vision_res.get("quality_warnings", []),
        "image_bytes": image_bytes,
        "image_dimensions": f"{image.width} × {image.height} px",
        "_pipeline_version": _PIPELINE_VERSION,
    }


def _compute_raw_38_probabilities(image: Image.Image) -> Tuple[np.ndarray, str]:
    """Compute the raw 38-class softmax probability vector using the active disease backend."""
    classifier = load_disease_classifier()
    c_type = classifier.get("type")
    model_name = classifier.get("model_name", "unknown")

    if c_type == "onnx_vit":
        session = classifier["session"]
        inp = preprocess_for_onnx_vit(image)
        logits = session.run(None, {"pixel_values": inp})[0][0]
        probs = _softmax_1d(logits)
        return probs, model_name

    if c_type == "pytorch":
        model = classifier["model"]
        processor = classifier["processor"]
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs, model_name

    if c_type == "keras":
        model = classifier["model"]
        resized = image.resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
        preds = np.asarray(model.predict(arr)[0], dtype=np.float32)
        if preds.min() < 0.0 or not np.isclose(preds.sum(), 1.0, atol=1e-2):
            probs = _softmax_1d(preds)
        else:
            probs = preds
        return probs, model_name

    raise RuntimeError(f"Unsupported disease classifier backend type: {c_type}")


def predict_disease(
    image_input: Any,
    threshold: float = CONFIDENCE_THRESHOLD,
    host_plant: Optional[str] = None,
    enforce_crop_gate: bool = True,
) -> Dict[str, Any]:
    """
    Execute plant disease detection on an input image with strict plant->disease
    consistency gating (Task 4), classical CV affected-area & severity analysis (Tasks 5-6),
    and verified knowledge base lookup (Task 7).
    Never returns fake fallback predictions.
    """
    image = preprocess_image(image_input)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    image_bytes = buf.getvalue()

    # 1. External Microservice Mode (only when explicitly configured via env var)
    api_url = os.getenv("PLANT_DISEASE_API_URL", EXTERNAL_API_URL).strip()
    if api_url:
        try:
            import requests

            files = {"image": ("specimen.jpg", image_bytes, "image/jpeg")}
            resp = requests.post(api_url, files=files, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            data["image_bytes"] = image_bytes
            return data
        except Exception as exc:
            raise RuntimeError(
                f"Configured external disease detection service at '{api_url}' is offline or failed: {exc}"
            ) from exc

    # 2. Plant -> Disease Consistency Gate (Task 4)
    allowed_crop_weights: Optional[Dict[str, float]] = None
    plant_pred: Optional[Dict[str, Any]] = None
    if host_plant is not None:
        canonical_crop = resolve_supported_disease_crop(host_plant)
        if canonical_crop is None:
            return _build_unsupported_crop_response(image, image_bytes, host_plant)
        allowed_crop_weights = {canonical_crop: 1.0}
    elif enforce_crop_gate:
        plant_pred = predict_species_detailed(image)
        # If full-frame prediction was low-confidence or distracted by desk/indoor clutter,
        # check whether classical CV leaf segmentation isolates a clear leaf crop.
        if not plant_pred["is_reliable"] or plant_pred["is_not_plant"]:
            try:
                img_rgb = np.asarray(image)
                seg_check = segment_leaf(img_rgb)
                if seg_check.get("found") and seg_check.get("bbox"):
                    cropped_rgb = crop_to_bbox(img_rgb, seg_check["bbox"])
                    if cropped_rgb.shape[:2] != img_rgb.shape[:2]:
                        leaf_pred = predict_species_detailed(image, leaf_crop=Image.fromarray(cropped_rgb))
                        if leaf_pred["is_reliable"] and not leaf_pred["is_not_plant"]:
                            plant_pred = leaf_pred
            except Exception:
                pass

        top_crop = (
            resolve_supported_disease_crop(plant_pred["species"])
            or resolve_supported_disease_crop(plant_pred["raw_label"])
        )
        if not plant_pred["is_reliable"] or plant_pred["is_not_plant"] or top_crop is None:
            detected_label = (
                plant_pred["species"]
                if plant_pred["is_reliable"] and not plant_pred["is_not_plant"]
                else "Unsupported / Non-Crop Specimen"
            )
            return _build_unsupported_crop_response(image, image_bytes, detected_label, plant_pred=plant_pred)

        # Collect candidate supported crops with >= 5% species probability
        allowed_crop_weights = {}
        for cand in plant_pred.get("top_candidates", []):
            c_crop = (
                resolve_supported_disease_crop(cand.get("species"))
                or resolve_supported_disease_crop(cand.get("raw_label"))
                or resolve_supported_disease_crop(cand.get("pv_crop"))
            )
            if c_crop and float(cand.get("score", 0.0)) >= 0.05:
                allowed_crop_weights[c_crop] = allowed_crop_weights.get(c_crop, 0.0) + float(cand["score"])
        if not allowed_crop_weights:
            allowed_crop_weights = {top_crop: 1.0}

    # 3. Compute 38-class Disease Probabilities & Apply Consistency Mask
    raw_probs, model_name = _compute_raw_38_probabilities(image)

    if allowed_crop_weights is not None:
        gated_probs = np.zeros_like(raw_probs)
        for crop_name, crop_weight in allowed_crop_weights.items():
            for idx in CROP_TO_CLASS_INDICES.get(crop_name, []):
                gated_probs[idx] = raw_probs[idx] * crop_weight
        total_gated = float(np.sum(gated_probs))
        if total_gated > 0.0:
            ranking_probs = gated_probs / total_gated
        else:
            ranking_probs = gated_probs
        # When a single host_plant is explicitly specified, normalize within that crop's classes
        # so impossible crops have 0.0 probability.
        if len(allowed_crop_weights) == 1:
            only_crop = next(iter(allowed_crop_weights))
            crop_indices = CROP_TO_CLASS_INDICES[only_crop]
            crop_mass = float(np.sum(raw_probs[crop_indices]))
            display_probs = np.zeros_like(raw_probs)
            if crop_mass > 0:
                display_probs[crop_indices] = raw_probs[crop_indices] / crop_mass
        else:
            display_probs = raw_probs
    else:
        ranking_probs = raw_probs
        display_probs = raw_probs

    sorted_indices = np.argsort(ranking_probs)[::-1]
    top_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top_idx

    raw_label = PLANTVILLAGE_38_RAW_CLASSES[top_idx]
    top_score = float(display_probs[top_idx])
    second_score = float(display_probs[second_idx]) if len(sorted_indices) > 1 else 0.0

    top_candidates: List[Dict[str, Any]] = []
    for i in sorted_indices[:5]:
        idx_int = int(i)
        if ranking_probs[idx_int] <= 0.0:
            continue
        sc = float(display_probs[idx_int])
        top_candidates.append(
            {
                "raw_label": PLANTVILLAGE_38_RAW_CLASSES[idx_int],
                "label": normalize_disease_label(PLANTVILLAGE_38_RAW_CLASSES[idx_int]),
                "score": sc,
                "confidence_pct": round(sc * 100, 1),
            }
        )

    info = get_disease_info(raw_label)
    crop, condition, is_healthy = split_raw_disease_label(raw_label)
    health_status = "Healthy" if is_healthy else "Diseased"

    conf_tier, policy_uncertain, policy_msg = evaluate_disease_confidence_policy(
        top_score,
        second_score,
        medium_threshold=threshold,
    )
    is_uncertain = policy_uncertain or (top_score < threshold)
    uncertainty_msg = policy_msg if is_uncertain else ""

    if health_status == "Healthy":
        full_disease_name = f"Healthy {crop}"
    else:
        full_disease_name = f"{crop} {condition}" if crop not in condition else condition

    # 4. Classical CV Lesion Segmentation, Project-Defined Severity & Derived Health Score (Tasks 5-6)
    vision_res = analyze_foliar_lesions(
        image,
        is_healthy=is_healthy,
        plant_name_or_id=crop,
        urgency_or_severity=info.get("severity", "medium"),
    )

    p_meta = _resolve_plant_metadata(crop) or (plant_pred.get("plant_metadata", {}) if plant_pred else {})
    gen_cond = _build_general_condition_diagnosis(image, crop, plant_pred=plant_pred)

    return {
        "success": True,
        "plant": crop,
        "scientific_name": p_meta.get("scientific", crop),
        "hindi_name": p_meta.get("hindi", ""),
        "common_species_problems": p_meta.get("problems", []),
        "uses": p_meta.get("uses", []),
        "disease": condition,
        "disease_name": full_disease_name,
        "health_status": health_status,
        "is_supported_crop": True,
        "confidence": top_score,
        "confidence_pct": round(top_score * 100, 1),
        "top2_confidence": second_score,
        "confidence_tier": conf_tier,
        "raw_label": raw_label,
        "model_name": model_name,
        "pathogen": info.get("pathogen", "N/A"),
        "severity": vision_res["project_severity_category"],
        "agronomic_severity_note": info.get("severity", "N/A"),
        "severity_scheme_label": vision_res["severity_scheme_label"],
        "affected_area_pct": vision_res["affected_area_pct"],
        "affected_area_status": vision_res["estimation_status"],
        "derived_health_score": vision_res["derived_health_score"],
        "health_score_label": vision_res["health_score_label"],
        "vision_analysis": vision_res,
        "quality_warnings": vision_res.get("quality_warnings", []),
        "symptoms": info.get("symptoms", NO_RECOMMENDATION_MESSAGE),
        "recommendations": info.get("recommendations", []),
        "preventive_measures": info.get("preventive_measures", []),
        "chemical_treatments": info.get("chemical_treatments", []),
        "organic_treatments": info.get("organic_treatments", []),
        "care": info.get("care") or p_meta.get("care", {}),
        "general_condition": gen_cond,
        "known_disease": info.get("known_disease", True),
        "needs_expert_review": info.get("needs_expert_review", True),
        "expert_review_notice": info.get("expert_review_notice", EXPERT_REVIEW_NOTICE),
        "no_recommendation_message": info.get("no_recommendation_message", ""),
        "is_uncertain": is_uncertain,
        "uncertainty_message": uncertainty_msg,
        "top_candidates": top_candidates,
        "image_bytes": image_bytes,
        "image_dimensions": f"{image.width} × {image.height} px",
        "_pipeline_version": _PIPELINE_VERSION,
    }

