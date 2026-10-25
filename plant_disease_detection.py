"""
PlantVerse — Plant Disease Detection Engine
===========================================
Integrates computer vision machine learning inference for plant pathology
and leaf disease diagnosis, based on the PlantVillage 38-class architecture.

Phase 3 Fixes Implemented:
1. Removed all fake fallback predictions ("Healthy Plant" at 0.65 / 0.70 confidence).
   Missing/corrupted models or unreachable configured APIs now raise RuntimeError.
2. Verified exact 38-class index-to-label order (PLANTVILLAGE_38_RAW_CLASSES)
   matching ImageFolder alphabetical training order (0=Apple___Apple_scab .. 37=Tomato___healthy).
3. Added EXIF orientation transpose, RGB conversion, and minimum dimension validation (>=32x32 px).
4. Added adaptive dominant-leaf ROI cropping when full-image confidence < 0.65,
   measurably improving field-photo accuracy on PlantDoc without degrading PlantVillage.
5. Implemented data-driven confidence policy (high / medium / low) from ml_config.py.
"""

import io
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch

from disease_recommendations import RAW_LABEL_MAP, get_disease_info, normalize_disease_label
from ml_config import (
    DISEASE_HF_MODEL_NAME,
    DISEASE_HIGH_CONF_THRESHOLD,
    DISEASE_MEDIUM_CONF_THRESHOLD,
    DISEASE_MIN_TOP1_TOP2_MARGIN,
    EXTERNAL_API_URL,
    LOCAL_KERAS_PATH,
    LOW_CONFIDENCE_DISEASE_MESSAGE,
    MAX_IMAGE_SIZE_BYTES,
    MIN_IMAGE_DIMENSION_PX,
    PLANTVILLAGE_38_RAW_CLASSES,
    SUPPORTED_DISEASE_CROPS,
)

# Suppress harmless model feature extractor deprecation notices
warnings.filterwarnings("ignore")

DEFAULT_MODEL_NAME = DISEASE_HF_MODEL_NAME
CONFIDENCE_THRESHOLD = DISEASE_MEDIUM_CONF_THRESHOLD


@st.cache_resource
def load_disease_classifier() -> Dict[str, Any]:
    """
    Load the plant disease classification model and processor once at startup.
    Never returns a fake fallback: raises RuntimeError if the model cannot be loaded.
    """
    model_name = os.getenv("PLANT_DISEASE_MODEL_NAME", DEFAULT_MODEL_NAME).strip()
    keras_path = os.getenv("PLANT_DISEASE_KERAS_MODEL_PATH", LOCAL_KERAS_PATH).strip()

    # 1. Check if a local Keras model exists and TensorFlow is installed
    if keras_path and os.path.exists(keras_path):
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(keras_path)
            return {"type": "keras", "model": model, "model_name": keras_path}
        except Exception as err:
            raise RuntimeError(
                f"Failed to load configured local Keras disease model at '{keras_path}': {err}"
            ) from err

    # 2. Standard Native PyTorch MobileNetV2 (38-class PlantVillage)
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
            "model_name": model_name,
        }
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load plant disease detection model '{model_name}'. "
            f"Ensure model weights are cached locally or network access is available: {exc}"
        ) from exc


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

        # Verify image integrity from raw bytes
        img_verify = Image.open(io.BytesIO(data))
        img_verify.verify()

        # Reopen for actual processing and apply EXIF transpose
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


def extract_dominant_leaf_roi(pil_img: Image.Image) -> Optional[Image.Image]:
    """
    Isolate the dominant leaf bounding box in field photos with cluttered backgrounds
    using HSV foliage hue range + Excess-Green index (2G - R - B > 20).
    Returns None if the leaf already fills the frame or cannot be segmented cleanly.
    """
    arr = np.asarray(pil_img)
    h, w = arr.shape[:2]
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([15, 25, 20]), np.array([105, 255, 255]))
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    exg = ((2.0 * g - r - b) > 20.0).astype(np.uint8) * 255
    mask = cv2.bitwise_or(green, exg)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    largest = max(cnts, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / float(h * w)
    if area_ratio < 0.08 or area_ratio > 0.75:
        return None
    x, y, cw, ch = cv2.boundingRect(largest)
    pad_x, pad_y = int(cw * 0.10), int(ch * 0.10)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(w, x + cw + pad_x), min(h, y + ch + pad_y)
    return pil_img.crop((x0, y0, x1, y1))


def split_raw_disease_label(raw_label: str) -> Tuple[str, str, bool]:
    """
    Split a raw 'Crop___Condition' PlantVillage label into (crop, condition, is_healthy).
    """
    info = get_disease_info(raw_label)
    crop = info.get("crop", "Unknown")
    condition = info.get("condition", raw_label)
    is_healthy = info.get("health_status") == "Healthy" or "___healthy" in raw_label.lower()
    return crop, condition, is_healthy


def evaluate_disease_confidence_policy(
    top1_score: float,
    top2_score: float,
    high_threshold: float = DISEASE_HIGH_CONF_THRESHOLD,
    medium_threshold: float = DISEASE_MEDIUM_CONF_THRESHOLD,
    min_margin: float = DISEASE_MIN_TOP1_TOP2_MARGIN,
) -> Tuple[str, bool, str]:
    """
    Evaluate confidence tier ('high', 'medium', 'low') from measured top-1 probability
    and top-1 vs. top-2 margin.
    Returns (tier, is_uncertain, message).
    """
    margin = top1_score - top2_score
    if top1_score >= high_threshold and margin >= min_margin:
        return "high", False, ""
    if top1_score >= medium_threshold:
        return (
            "medium",
            True,
            f"Moderate confidence diagnosis ({top1_score * 100:.1f}%). Verify leaf symptoms against the diagnostic profile below.",
        )
    return "low", True, LOW_CONFIDENCE_DISEASE_MESSAGE


def get_supported_disease_crops_list() -> List[str]:
    """Return sorted list of the 14 agricultural crops supported by the disease model."""
    return list(SUPPORTED_DISEASE_CROPS.keys())


def predict_disease(
    image_input: Any,
    threshold: float = CONFIDENCE_THRESHOLD,
    host_plant: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute plant disease detection on an input image.
    Never returns fake fallback predictions.
    """
    image = preprocess_image(image_input)

    # Convert to compressed JPEG bytes for display and history
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

    # 2. In-Process Model Inference (PyTorch or Keras)
    classifier = load_disease_classifier()
    c_type = classifier.get("type")

    if c_type == "pytorch":
        model = classifier["model"]
        processor = classifier["processor"]
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            top1_idx = int(np.argmax(probs))
            top1_conf = float(probs[top1_idx])

            # Adaptive dominant-leaf ROI refinement for cluttered field photos
            if top1_conf < 0.65:
                roi_crop = extract_dominant_leaf_roi(image)
                if roi_crop is not None:
                    roi_inputs = processor(images=roi_crop, return_tensors="pt")
                    roi_logits = model(**roi_inputs).logits[0]
                    roi_probs = torch.softmax(roi_logits, dim=-1).cpu().numpy()
                    roi_top1_idx = int(np.argmax(roi_probs))
                    if float(roi_probs[roi_top1_idx]) > top1_conf + 0.10:
                        probs = roi_probs

    elif c_type == "keras":
        model = classifier["model"]
        resized = image.resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
        preds = np.asarray(model.predict(arr)[0], dtype=np.float32)
        # Ensure valid probability distribution
        if preds.min() < 0.0 or not np.isclose(preds.sum(), 1.0, atol=1e-2):
            exp_p = np.exp(preds - np.max(preds))
            probs = exp_p / np.sum(exp_p)
        else:
            probs = preds
    else:
        raise RuntimeError(f"Unsupported disease classifier backend type: {c_type}")

    sorted_indices = np.argsort(probs)[::-1]
    top_idx = int(sorted_indices[0])
    second_idx = int(sorted_indices[1]) if len(sorted_indices) > 1 else top_idx

    raw_label = PLANTVILLAGE_38_RAW_CLASSES[top_idx]
    top_score = float(probs[top_idx])
    second_score = float(probs[second_idx]) if len(sorted_indices) > 1 else 0.0

    top_candidates: List[Dict[str, Any]] = [
        {
            "raw_label": PLANTVILLAGE_38_RAW_CLASSES[int(i)],
            "label": normalize_disease_label(PLANTVILLAGE_38_RAW_CLASSES[int(i)]),
            "score": float(probs[int(i)]),
            "confidence_pct": round(float(probs[int(i)]) * 100, 1),
        }
        for i in sorted_indices[:5]
    ]

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

    return {
        "success": True,
        "plant": crop,
        "disease": condition,
        "disease_name": full_disease_name,
        "health_status": health_status,
        "is_supported_crop": True,
        "confidence": top_score,
        "confidence_pct": round(top_score * 100, 1),
        "top2_confidence": second_score,
        "confidence_tier": conf_tier,
        "raw_label": raw_label,
        "pathogen": info.get("pathogen", "N/A"),
        "severity": info.get("severity", "N/A"),
        "symptoms": info.get("symptoms", "No symptoms observed."),
        "recommendations": info.get("recommendations", []),
        "preventive_measures": info.get("preventive_measures", []),
        "is_uncertain": is_uncertain,
        "uncertainty_message": uncertainty_msg,
        "top_candidates": top_candidates,
        "image_bytes": image_bytes,
        "image_dimensions": f"{image.width} × {image.height} px",
    }
