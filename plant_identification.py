# ============================================================
# PlantVerse - Plant Identification
# ============================================================

from functools import lru_cache
import io
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

for _venv_sp in sorted((Path(__file__).resolve().parent / "venv" / "lib").glob("python*/site-packages")):
    if str(_venv_sp) not in sys.path:
        sys.path.append(str(_venv_sp))

import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]
from PIL import Image, ImageOps
import requests
import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

import importlib as _importlib
import ml_config as _ml_config
if not hasattr(_ml_config, "load_conditions_metadata"):
    _importlib.reload(_ml_config)

from ml_config import (
    ENABLE_FLORASENSE_SECONDARY,
    FLORASENSE_HIGH_CONF_THRESHOLD,
    FLORASENSE_INPUT_SIZE,
    FLORASENSE_MEAN,
    FLORASENSE_MEDIUM_CONF_THRESHOLD,
    FLORASENSE_MIN_TOP1_TOP2_RATIO,
    FLORASENSE_MODEL_NAME,
    FLORASENSE_STD,
    LOW_CONFIDENCE_PLANT_MESSAGE,
    MAX_IMAGE_SIZE_BYTES,
    MIN_IMAGE_DIMENSION_PX,
    ONNX_NOT_PLANT_THRESHOLD,
    ONNX_PLANT_HIGH_CONF_THRESHOLD,
    ONNX_PLANT_INPUT_SIZE,
    ONNX_PLANT_LABELS_PATH,
    ONNX_PLANT_MEAN,
    ONNX_PLANT_MEDIUM_CONF_THRESHOLD,
    ONNX_PLANT_MIN_TOP1_TOP2_MARGIN,
    ONNX_PLANT_MODEL_PATH,
    ONNX_PLANT_PROVISIONAL_N,
    ONNX_PLANT_STD,
    PLANT_ID_BACKEND,
    SUPPORTED_PLANT_SUMMARY,
    load_conditions_metadata,
    load_onnx_plant_labels,
    load_plants_metadata,
)


# ============================================================
# 1. Language Setup
# ============================================================

LANGUAGES = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "తెలుగు (Telugu)": "te",
    "தமிழ் (Tamil)": "ta",
}


# ============================================================
# 2. Translation Function
# ============================================================

@st.cache_data
def translate_text(text, target_lang):
    """
    Translate English text into the selected language.

    If Google Translate is unavailable, the original English
    text is returned instead of crashing the application.
    """
    if not text or target_lang == "en":
        return text

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }
        response = requests.get(
            url,
            params=params,
            timeout=10,
            headers={"User-Agent": "PlantVerse/1.0"},
        )
        response.raise_for_status()
        data = response.json()
        if data and data[0]:
            return "".join(
                part[0]
                for part in data[0]
                if part and len(part) > 0
            )
        return text
    except Exception:
        return text


# ============================================================
# 3. Primary ONNX ConvNeXt-Tiny Plant Classifier & Secondary FloraSense
# ============================================================

@st.cache_resource
def load_onnx_plant_classifier(
    model_path_str: Optional[str] = None,
    labels_path_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load the primary ConvNeXt-Tiny ONNX plant + foliar condition classifier once.
    Raises RuntimeError with a clear message if the model or labels file is missing or corrupted.
    """
    m_path = Path(model_path_str or os.getenv("PLANTVERSE_ONNX_PLANT_MODEL", str(ONNX_PLANT_MODEL_PATH)))
    l_path = Path(labels_path_str or os.getenv("PLANTVERSE_ONNX_PLANT_LABELS", str(ONNX_PLANT_LABELS_PATH)))

    if not m_path.exists():
        raise RuntimeError(
            f"Missing plant identification model file at '{m_path}'. "
            "Please verify that 'models/plant_model.onnx' is present."
        )
    if m_path.stat().st_size < 1024:
        raise RuntimeError(
            f"Plant identification model file at '{m_path}' is empty or truncated ({m_path.stat().st_size} bytes)."
        )
    if not l_path.exists():
        raise RuntimeError(
            f"Missing plant labels file at '{l_path}'. "
            "Please verify that 'models/plant_labels.json' is present."
        )

    try:
        meta = json.loads(l_path.read_text(encoding="utf-8")) if "json" in globals() else load_onnx_plant_labels()
    except Exception as exc:
        raise RuntimeError(f"Corrupted plant labels JSON at '{l_path}': {exc}") from exc

    try:
        import json as _json
        meta = _json.loads(l_path.read_text(encoding="utf-8"))
        species = meta["species"]
        conditions = meta["conditions"]
        input_size = int(meta.get("input_size", 224))
        mean = np.array(meta.get("mean", ONNX_PLANT_MEAN), dtype=np.float32)
        std = np.array(meta.get("std", ONNX_PLANT_STD), dtype=np.float32)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(str(m_path), opts, providers=["CPUExecutionProvider"])
        plants_meta = load_plants_metadata()
        conditions_meta = load_conditions_metadata()

        return {
            "session": session,
            "species": species,
            "conditions": conditions,
            "input_size": input_size,
            "mean": mean,
            "std": std,
            "plants_meta": plants_meta,
            "conditions_meta": conditions_meta,
            "model_path": str(m_path),
        }
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize ONNX plant identification model from '{m_path}' (weights may be corrupted): {exc}"
        ) from exc


@st.cache_resource
def load_florasense_classifier() -> Dict[str, Any]:
    """
    Load the secondary FloraSense Vision Transformer model in eval mode.
    Used only when secondary 9,985-taxon suggestions are requested or when
    PLANTVERSE_PLANT_ID_BACKEND=florasense is set for benchmark comparison.
    """
    try:
        model = AutoModelForImageClassification.from_pretrained(FLORASENSE_MODEL_NAME)
        model.eval()
        processor = AutoImageProcessor.from_pretrained(FLORASENSE_MODEL_NAME)
        return {
            "model": model,
            "processor": processor,
            "id2label": model.config.id2label,
        }
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load secondary FloraSense model '{FLORASENSE_MODEL_NAME}': {exc}"
        ) from exc


def load_classifier() -> Dict[str, Any]:
    """Load the configured active plant classifier."""
    backend = os.getenv("PLANTVERSE_PLANT_ID_BACKEND", PLANT_ID_BACKEND).strip().lower()
    if backend == "florasense":
        return load_florasense_classifier()
    return load_onnx_plant_classifier()


def _validate_and_orient_image(image: Any) -> Image.Image:
    """
    Validate image input, apply EXIF orientation transpose, convert to RGB,
    and enforce minimum dimensions (>= 32x32 px).
    """
    if image is None:
        raise ValueError("No image provided for plant identification.")
    if isinstance(image, (str, os.PathLike)):
        p = Path(image)
        if not p.exists():
            raise ValueError(f"Image file does not exist: {image}")
        if p.stat().st_size == 0:
            raise ValueError("Image file is empty (0 bytes).")
        if p.stat().st_size > MAX_IMAGE_SIZE_BYTES:
            raise ValueError("Image file exceeds maximum allowed size (10 MB).")
        pil_img = Image.open(p)
    elif isinstance(image, Image.Image):
        pil_img = image.copy()
    elif hasattr(image, "read"):
        if hasattr(image, "seek"):
            image.seek(0)
        data = image.read()
        if hasattr(image, "seek"):
            image.seek(0)
        if not data:
            raise ValueError("Image stream is empty (0 bytes).")
        pil_img = Image.open(io.BytesIO(data))
    elif isinstance(image, (bytes, bytearray)):
        if not image:
            raise ValueError("Image data is empty (0 bytes).")
        pil_img = Image.open(io.BytesIO(bytes(image)))
    else:
        raise ValueError("Unsupported image input type for plant identification.")

    img = ImageOps.exif_transpose(pil_img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width < MIN_IMAGE_DIMENSION_PX or img.height < MIN_IMAGE_DIMENSION_PX:
        raise ValueError(
            f"Image is too small ({img.width}x{img.height} px). "
            f"Minimum dimension is {MIN_IMAGE_DIMENSION_PX}x{MIN_IMAGE_DIMENSION_PX} px."
        )
    return img


def preprocess_for_onnx_plant(
    image: Image.Image,
    input_size: int = 224,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Exact training-matched preprocessing for FloraCare's `plant_model.onnx` (ConvNeXt-Tiny):
    1. EXIF orientation transpose + RGB conversion.
    2. Squash-resize to (224, 224) with BILINEAR interpolation.
    3. Rescale pixel intensities to [0.0, 1.0] (/ 255.0).
    4. Normalize with ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    5. Transpose HWC -> CHW and add batch dimension [1, 3, 224, 224].
    """
    img = _validate_and_orient_image(image)
    m = mean if mean is not None else np.array(ONNX_PLANT_MEAN, dtype=np.float32)
    s = std if std is not None else np.array(ONNX_PLANT_STD, dtype=np.float32)
    resized = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    chw = ((x - m) / s).transpose(2, 0, 1)
    return np.expand_dims(chw, axis=0).astype(np.float32)


def preprocess_for_florasense(image: Image.Image) -> torch.Tensor:
    """
    Exact training-matched preprocessing for Sisigoks/FloraSense (google/vit-base-patch16-224).
    """
    img = _validate_and_orient_image(image)
    resized = img.resize(FLORASENSE_INPUT_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array(FLORASENSE_MEAN, dtype=np.float32)
    std = np.array(FLORASENSE_STD, dtype=np.float32)
    norm = (arr - mean) / std
    chw = np.transpose(norm, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / np.sum(exp_v, axis=-1, keepdims=True)


def evaluate_onnx_plant_confidence_policy(
    top1_id: str,
    top1_score: float,
    top2_score: float,
    not_plant_score: float = 0.0,
    high_threshold: float = ONNX_PLANT_HIGH_CONF_THRESHOLD,
    medium_threshold: float = ONNX_PLANT_MEDIUM_CONF_THRESHOLD,
    min_margin: float = ONNX_PLANT_MIN_TOP1_TOP2_MARGIN,
    not_plant_threshold: float = ONNX_NOT_PLANT_THRESHOLD,
) -> Tuple[str, str]:
    """
    Evaluate confidence tier ('high', 'medium', 'low') for the ONNX ConvNeXt-Tiny plant model.
    Thresholds are provisional (derived on n=163 tuning images: 45 iNaturalist + 118 PlantDoc tuning half).
    """
    if top1_id == "not_plant" or not_plant_score >= not_plant_threshold:
        return "low", LOW_CONFIDENCE_PLANT_MESSAGE

    margin = top1_score - top2_score
    if top1_score >= high_threshold and margin >= min_margin:
        return "high", ""
    if top1_score >= medium_threshold and margin >= (min_margin * 0.5):
        return (
            "medium",
            f"Moderate confidence identification ({top1_score * 100:.1f}%; provisional threshold derived on n={ONNX_PLANT_PROVISIONAL_N} tuning images). "
            "Please verify morphological traits against the species profile below.",
        )
    return "low", LOW_CONFIDENCE_PLANT_MESSAGE


def evaluate_plant_confidence_policy(
    top1_score: float,
    top2_score: float,
    high_threshold: float = FLORASENSE_HIGH_CONF_THRESHOLD,
    medium_threshold: float = FLORASENSE_MEDIUM_CONF_THRESHOLD,
    min_ratio: float = FLORASENSE_MIN_TOP1_TOP2_RATIO,
) -> Tuple[str, str]:
    """Evaluate confidence tier for the secondary FloraSense model."""
    ratio = top1_score / max(top2_score, 1e-7)
    if top1_score >= high_threshold and ratio >= min_ratio:
        return "high", ""
    if top1_score >= medium_threshold and ratio >= min_ratio:
        return (
            "medium",
            "Moderate confidence prediction: fine-grained visual similarity detected across related taxa.",
        )
    return "low", LOW_CONFIDENCE_PLANT_MESSAGE


def _clean_florasense_label(raw_label: str) -> str:
    if not raw_label:
        return "Unknown"
    if "_" not in raw_label:
        return raw_label.strip()
    parts = [p for p in raw_label.replace("__", "_").split("_") if p]
    if len(parts) >= 2 and parts[0][0].isupper() and parts[1][0].islower():
        return f"{parts[0]} {parts[1]}"
    return " ".join(parts[:2]) if len(parts) >= 2 else parts[0]


def get_florasense_secondary_suggestions(image: Image.Image, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Optional secondary top-5 taxonomic suggestions from Sisigoks/FloraSense (9,985 classes).
    Presented strictly as 'other possible matches (low confidence)', never as the headline.
    """
    try:
        pixel_values = preprocess_for_florasense(image)
        bundle = load_florasense_classifier()
        model = bundle["model"]
        id2label = bundle["id2label"]
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:50]
        species_scores: Dict[str, float] = {}
        species_raw_map: Dict[str, str] = {}
        for idx in top_indices:
            raw_lbl = id2label[int(idx)]
            clean_lbl = _clean_florasense_label(raw_lbl)
            species_scores[clean_lbl] = species_scores.get(clean_lbl, 0.0) + float(probs[int(idx)])
            if clean_lbl not in species_raw_map:
                species_raw_map[clean_lbl] = raw_lbl

        ranked = sorted(species_scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            {
                "species": sp,
                "raw_label": species_raw_map.get(sp, sp),
                "score": float(sc),
                "confidence_pct": round(float(sc) * 100, 2),
                "source": "Sisigoks/FloraSense (secondary 9,985-taxon index)",
            }
            for sp, sc in ranked
        ]
    except Exception:
        return []


def _predict_species_florasense_detailed(image: Image.Image) -> Dict[str, Any]:
    """Run FloraSense directly (used only when PLANTVERSE_PLANT_ID_BACKEND=florasense in benchmarks)."""
    pixel_values = preprocess_for_florasense(image)
    bundle = load_florasense_classifier()
    model = bundle["model"]
    id2label = bundle["id2label"]

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:50]
    species_scores: Dict[str, float] = {}
    species_raw_map: Dict[str, str] = {}
    for idx in top_indices:
        raw_lbl = id2label[int(idx)]
        clean_lbl = _clean_florasense_label(raw_lbl)
        species_scores[clean_lbl] = species_scores.get(clean_lbl, 0.0) + float(probs[int(idx)])
        if clean_lbl not in species_raw_map:
            species_raw_map[clean_lbl] = raw_lbl

    ranked = sorted(species_scores.items(), key=lambda kv: -kv[1])
    top1_label, top1_score = ranked[0] if ranked else ("Unknown", 0.0)
    top2_score = ranked[1][1] if len(ranked) > 1 else 0.0

    tier, message = evaluate_plant_confidence_policy(top1_score, top2_score)
    top_candidates = [
        {
            "species": sp,
            "scientific_name": sp,
            "raw_label": species_raw_map.get(sp, sp),
            "score": float(sc),
            "confidence_pct": round(float(sc) * 100, 2),
        }
        for sp, sc in ranked[:5]
    ]

    return {
        "species": top1_label,
        "scientific_name": top1_label,
        "raw_label": species_raw_map.get(top1_label, top1_label),
        "score": float(top1_score),
        "top2_score": float(top2_score),
        "margin": round(float(top1_score - top2_score), 4),
        "confidence_tier": tier,
        "is_reliable": tier in ("high", "medium"),
        "is_not_plant": False,
        "warning_message": message,
        "top_candidates": top_candidates,
        "secondary_suggestions": [],
        "condition_predictions": [],
        "plant_metadata": {},
        "model_name": FLORASENSE_MODEL_NAME,
    }


def predict_species_detailed(
    image: Any,
    include_secondary_florasense: bool = False,
    leaf_crop: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """
    Run primary plant identification using ConvNeXt-Tiny (`models/plant_model.onnx`,
    68 species + `not_plant`, and 7 foliar condition classes).
    When `leaf_crop` is provided (from classical CV leaf segmentation), species probabilities
    are averaged over the full frame and the cropped leaf, and condition probabilities are
    taken from the leaf crop (matching FloraCare's dual-view inference).
    Optionally attaches secondary low-confidence suggestions from `Sisigoks/FloraSense`.
    """
    backend = os.getenv("PLANTVERSE_PLANT_ID_BACKEND", PLANT_ID_BACKEND).strip().lower()
    img = _validate_and_orient_image(image)
    if backend == "florasense":
        return _predict_species_florasense_detailed(img)

    bundle = load_onnx_plant_classifier()
    session = bundle["session"]
    species_list: List[str] = bundle["species"]
    conditions_list: List[str] = bundle["conditions"]
    plants_meta: Dict[str, Dict[str, Any]] = bundle["plants_meta"]
    conditions_meta: Dict[str, Dict[str, Any]] = bundle.get("conditions_meta", {})

    inp_full = preprocess_for_onnx_plant(
        img,
        input_size=bundle["input_size"],
        mean=bundle["mean"],
        std=bundle["std"],
    )
    if leaf_crop is not None:
        inp_leaf = preprocess_for_onnx_plant(
            leaf_crop,
            input_size=bundle["input_size"],
            mean=bundle["mean"],
            std=bundle["std"],
        )
        inp_batch = np.concatenate([inp_full, inp_leaf], axis=0)
        species_logits, condition_logits = session.run(None, {"pixel_values": inp_batch})
        p_species = _softmax_np(species_logits).mean(axis=0)
        p_condition = _softmax_np(condition_logits)[-1]
    else:
        species_logits, condition_logits = session.run(None, {"pixel_values": inp_full})
        p_species = _softmax_np(species_logits)[0]
        p_condition = _softmax_np(condition_logits)[0]

    not_plant_idx = species_list.index("not_plant") if "not_plant" in species_list else -1
    not_plant_prob = float(p_species[not_plant_idx]) if not_plant_idx >= 0 else 0.0

    sorted_sp_idx = np.argsort(p_species)[::-1]
    top1_idx = int(sorted_sp_idx[0])
    top2_idx = int(sorted_sp_idx[1]) if len(sorted_sp_idx) > 1 else top1_idx

    top1_id = species_list[top1_idx]
    top1_score = float(p_species[top1_idx])
    top2_score = float(p_species[top2_idx]) if len(sorted_sp_idx) > 1 else 0.0

    tier, warning_msg = evaluate_onnx_plant_confidence_policy(
        top1_id=top1_id,
        top1_score=top1_score,
        top2_score=top2_score,
        not_plant_score=not_plant_prob,
    )
    is_not_plant = top1_id == "not_plant" or not_plant_prob >= ONNX_NOT_PLANT_THRESHOLD

    if is_not_plant:
        display_name = "Not a plant"
        scientific_name = "Non-botanical image"
        wiki_query = ""
        p_meta = {}
    else:
        p_meta = plants_meta.get(top1_id, {})
        display_name = p_meta.get("name", top1_id.replace("_", " ").capitalize())
        scientific_name = p_meta.get("scientific", display_name)
        wiki_query = p_meta.get("wiki", scientific_name)

    top_candidates: List[Dict[str, Any]] = []
    for idx in sorted_sp_idx:
        sp_id = species_list[int(idx)]
        if sp_id == "not_plant":
            continue
        meta_i = plants_meta.get(sp_id, {})
        name_i = meta_i.get("name", sp_id.replace("_", " ").capitalize())
        sci_i = meta_i.get("scientific", name_i)
        sc_i = float(p_species[int(idx)])
        top_candidates.append(
            {
                "id": sp_id,
                "species": name_i,
                "scientific_name": sci_i,
                "raw_label": sp_id,
                "pv_crop": meta_i.get("pv_crop"),
                "score": sc_i,
                "confidence_pct": round(sc_i * 100, 2),
            }
        )
        if len(top_candidates) >= 5:
            break

    sorted_cond_idx = np.argsort(p_condition)[::-1]
    condition_predictions = []
    for i in sorted_cond_idx:
        c_id = conditions_list[int(i)]
        c_meta = conditions_meta.get(c_id, {})
        c_score = float(p_condition[int(i)])
        condition_predictions.append(
            {
                "id": c_id,
                "name": c_meta.get("disease", c_id.replace("_", " ").capitalize()),
                "type": c_meta.get("type", "General foliar condition"),
                "pathogen": c_meta.get("pathogen", "N/A"),
                "urgency": c_meta.get("urgency", "medium"),
                "healthy": bool(c_meta.get("healthy", c_id == "healthy")),
                "symptoms": c_meta.get("symptoms", ""),
                "chemical": c_meta.get("chemical", []),
                "organic": c_meta.get("organic", []),
                "prevention": c_meta.get("prevention", []),
                "score": c_score,
                "confidence_pct": round(c_score * 100, 1),
            }
        )

    secondary_suggestions: List[Dict[str, Any]] = []
    if include_secondary_florasense or ENABLE_FLORASENSE_SECONDARY:
        secondary_suggestions = get_florasense_secondary_suggestions(img, top_k=5)

    return {
        "species": display_name,
        "scientific_name": scientific_name,
        "wiki_query": wiki_query,
        "raw_label": top1_id,
        "score": float(top1_score),
        "top2_score": float(top2_score),
        "not_plant_score": round(not_plant_prob, 4),
        "margin": round(float(top1_score - top2_score), 4),
        "confidence_tier": tier,
        "is_reliable": (tier in ("high", "medium")) and not is_not_plant,
        "is_not_plant": is_not_plant,
        "warning_message": warning_msg,
        "top_candidates": top_candidates,
        "secondary_suggestions": secondary_suggestions,
        "condition_predictions": condition_predictions,
        "plant_metadata": p_meta,
        "model_name": "ConvNeXt-Tiny ONNX (plant_model.onnx)",
    }


def predict_species(image: Any) -> Tuple[str, float]:
    """
    Predict plant species from uploaded image behind the existing PlantVerse interface.
    Returns (species_name, confidence_score).
    """
    res = predict_species_detailed(image)
    return res["species"], res["score"]


def get_supported_plants_info() -> Dict[str, Any]:
    """Return metadata and full 68-species list for config/UI display."""
    info = dict(SUPPORTED_PLANT_SUMMARY["onnx_convnext"])
    plants_meta = load_plants_metadata()
    species_names = [
        f"{p['name']} ({p['scientific']})" if p.get("scientific") else p["name"]
        for p in plants_meta.values()
    ]
    info["supported_species_list"] = sorted(species_names)
    return info


# ============================================================
# 4. Wikipedia Search
# ============================================================

WIKIMEDIA_HEADERS = {
    "User-Agent": "PlantVerse/1.0 (https://github.com/plantverse/app; contact@plantverse.org) python-requests/2.31"
}


@st.cache_data
def get_wikipedia_title(name: str):
    """
    Search Wikipedia for the predicted plant.

    If Wikipedia is unavailable, return the original
    model prediction instead of crashing.
    """
    clean_query = _clean_florasense_label(name) if "_" in name else name
    # Also map known plant metadata names to their clean Wikipedia title if available
    plants_meta = load_plants_metadata()
    for p in plants_meta.values():
        if clean_query.lower() in (p["id"].lower(), p["name"].lower(), p.get("scientific", "").lower()):
            if p.get("wiki"):
                clean_query = p["wiki"]
                break

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": clean_query,
        "format": "json",
        "utf8": 1,
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10,
            headers=WIKIMEDIA_HEADERS,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "json" not in content_type:
            return clean_query
        data = response.json()
        results = data.get("query", {}).get("search", [])
        if results:
            return results[0].get("title", clean_query)
        return clean_query
    except Exception:
        return clean_query


# ============================================================
# 5. Wikidata Taxonomy
# ============================================================

_KNOWN_RANKS = {
    "Q7432": "Species",
    "Q34740": "Genus",
    "Q227936": "Tribe",
    "Q164280": "Subfamily",
    "Q35409": "Family",
    "Q5867959": "Suborder",
    "Q36602": "Order",
    "Q5867051": "Subclass",
    "Q37517": "Class",
    "Q235536": "Subdivision",
    "Q334460": "Division",
    "Q38348": "Phylum",
    "Q2752679": "Subkingdom",
    "Q36732": "Kingdom",
    "Q146481": "Domain",
    "Q19858692": "Superdomain",
    "Q713623": "Clade",
}


@st.cache_data
def get_taxonomy_from_wikidata(label: str):
    """
    Get plant taxonomy from Wikidata.

    If Wikidata fails, return an error dictionary instead
    of crashing the application.
    """
    clean_label = _clean_florasense_label(label) if "_" in label else label
    search_url = "https://www.wikidata.org/w/api.php"

    try:
        eid = None

        # 1. Try resolving exact Wikidata QID from the Wikipedia page first
        try:
            wp_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "prop": "pageprops",
                    "titles": clean_label,
                    "redirects": 1,
                    "format": "json",
                },
                timeout=8,
                headers=WIKIMEDIA_HEADERS,
            )
            if wp_resp.status_code == 200:
                pages = wp_resp.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    qid = p.get("pageprops", {}).get("wikibase_item")
                    if qid:
                        eid = qid
                        break
        except Exception:
            pass

        # 2. Fall back to Wikidata wbsearchentities if not found via Wikipedia pageprops
        if not eid:
            response = requests.get(
                search_url,
                params={
                    "action": "wbsearchentities",
                    "search": clean_label,
                    "language": "en",
                    "format": "json",
                },
                timeout=10,
                headers=WIKIMEDIA_HEADERS,
            )
            response.raise_for_status()
            results = response.json().get("search", [])
            if results:
                eid = results[0].get("id")

        if not eid:
            return {"error": f"No Wikidata entity found for '{clean_label}'."}

        # 3. Single-request Wikidata SPARQL query for the complete taxonomy chain
        sparql_query = f"""
        SELECT ?rankLabel ?taxonName ?taxonLabel WHERE {{
          wd:{eid} (wdt:P1582|wdt:P31|wdt:P279)? / wdt:P171* ?taxon .
          ?taxon wdt:P105 ?rank .
          OPTIONAL {{ ?taxon wdt:P225 ?taxonName . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        try:
            sparql_resp = requests.get(
                "https://query.wikidata.org/sparql",
                params={"query": sparql_query, "format": "json"},
                timeout=10,
                headers=WIKIMEDIA_HEADERS,
            )
            if sparql_resp.status_code == 200:
                bindings = (
                    sparql_resp.json()
                    .get("results", {})
                    .get("bindings", [])
                )
                taxonomy = {}
                for b in bindings:
                    rank_val = b.get("rankLabel", {}).get("value", "").strip()
                    sci_val = (
                        b.get("taxonName", {}).get("value")
                        or b.get("taxonLabel", {}).get("value")
                        or ""
                    ).strip()
                    if rank_val and sci_val and rank_val.lower() != "clade":
                        rank_key = rank_val.capitalize()
                        if rank_key not in taxonomy:
                            taxonomy[rank_key] = sci_val
                if taxonomy:
                    return taxonomy
        except Exception:
            pass

        # 4. Fallback: lightweight wbgetentities traversal if SPARQL is unavailable
        @lru_cache(None)
        def fetch(e):
            try:
                r = requests.get(
                    search_url,
                    params={
                        "action": "wbgetentities",
                        "ids": e,
                        "props": "claims|labels",
                        "languages": "en",
                        "format": "json",
                    },
                    timeout=8,
                    headers=WIKIMEDIA_HEADERS,
                )
                r.raise_for_status()
                return r.json().get("entities", {}).get(e, {})
            except Exception:
                return {}

        def find_taxon(e, depth=0):
            if depth > 4:
                return None
            ent = fetch(e)
            if not ent:
                return None
            claims = ent.get("claims", {})
            if "P225" in claims or "P171" in claims:
                return e
            for prop in ("P1582", "P31", "P279"):
                for claim in claims.get(prop, []):
                    try:
                        val = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                        if isinstance(val, dict) and val.get("id"):
                            res = find_taxon(val["id"], depth + 1)
                            if res:
                                return res
                    except Exception:
                        continue
            return None

        taxon_e = find_taxon(eid)
        if not taxon_e:
            return {"error": f"No taxonomic information found for '{clean_label}'."}

        taxonomy = {}
        visited = set()

        def collect(e, lvl=0):
            if lvl > 14 or e in visited:
                return
            visited.add(e)
            ent = fetch(e)
            if not ent:
                return
            claims = ent.get("claims", {})
            sci = ""
            try:
                p225 = claims.get("P225", [])
                if p225:
                    sci = p225[0].get("mainsnak", {}).get("datavalue", {}).get("value", "")
                if not sci:
                    sci = ent.get("labels", {}).get("en", {}).get("value", "")
            except Exception:
                sci = ""

            rank_id = ""
            try:
                p105 = claims.get("P105", [])
                if p105:
                    rank_id = p105[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id", "")
            except Exception:
                rank_id = ""

            rank = _KNOWN_RANKS.get(rank_id, "")
            if sci and rank and rank.lower() != "clade":
                taxonomy[rank.capitalize()] = sci

            try:
                p171 = claims.get("P171", [])
                if p171:
                    parent = p171[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                    if parent:
                        collect(parent, lvl + 1)
            except Exception:
                pass

        collect(taxon_e)
        return taxonomy

    except requests.exceptions.RequestException:
        return {"error": "Unable to connect to Wikidata."}
    except Exception:
        return {"error": "Unable to retrieve taxonomy information."}


# ============================================================
# 6. Medicinal Uses from Wikipedia
# ============================================================

def get_medicinal_uses_from_wikipedia(
    title: str,
    target_lang="en",
):
    """
    Extract medicinal/traditional medicine/uses information
    from the Wikipedia page using a single lightweight API call.
    """
    clean_title = _clean_florasense_label(title) if "_" in title else title

    try:
        url = "https://en.wikipedia.org/w/api.php"
        response = requests.get(
            url,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": clean_title,
                "redirects": 1,
                "format": "json",
            },
            timeout=10,
            headers=WIKIMEDIA_HEADERS,
        )
        response.raise_for_status()

        pages = response.json().get("query", {}).get("pages", {})
        extract = ""
        for p in pages.values():
            if p.get("extract"):
                extract = p["extract"]
                break

        if not extract:
            return None

        priority_keywords = (
            "medicinal",
            "traditional medicine",
            "pharmacology",
            "ethnobotan",
            "folk medicine",
            "herbal",
            "therapeutic",
            "uses",
            "culinary",
            "phytochem",
            "chemical",
            "nutrition",
            "toxicity",
            "cultivation",
            "description",
        )

        sections = []
        current_heading = "lead"
        current_lines = []
        for line in extract.splitlines():
            m = re.match(r"^=+\s*(.*?)\s*=+$", line.strip())
            if m:
                sections.append((current_heading.lower(), "\n".join(current_lines).strip()))
                current_heading = m.group(1)
                current_lines = []
            else:
                current_lines.append(line)
        sections.append((current_heading.lower(), "\n".join(current_lines).strip()))

        target_text = ""
        for kw in priority_keywords:
            for heading, body in sections:
                if heading != "lead" and kw in heading and len(body.split()) >= 8:
                    target_text = body
                    break
            if target_text:
                break

        if not target_text:
            for heading, body in sections:
                if heading == "lead" and body:
                    target_text = body
                    break

        if not target_text:
            return None

        bullet_points = []
        for paragraph in target_text.split("\n"):
            text = re.sub(r"\[\d+\]", "", paragraph.strip())
            if len(text.split()) < 5:
                continue
            sentences = re.split(r"(?<!\b[A-Z])\.\s+", text)
            for sentence in sentences:
                sentence = sentence.strip().strip(".")
                if len(sentence.split()) < 5:
                    continue
                if sentence.lower().startswith("there is insufficient"):
                    continue
                if target_lang != "en":
                    sentence = translate_text(sentence, target_lang)
                bullet_points.append(f"{sentence}.")
                if len(bullet_points) >= 5:
                    return bullet_points

        return bullet_points[:5] if bullet_points else None
    except Exception:
        return None
