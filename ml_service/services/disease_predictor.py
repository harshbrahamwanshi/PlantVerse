"""
PlantVerse ML Service — Disease Predictor
=========================================
Encapsulates model loading (loaded once at startup) and prediction inference.
Removed all fake fallback predictions ("Healthy Plant" at 0.70 confidence).
"""

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageOps
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from disease_recommendations import get_disease_info, normalize_disease_label
from ml_config import (
    DISEASE_HF_MODEL_NAME,
    DISEASE_MEDIUM_CONF_THRESHOLD,
    LOCAL_KERAS_PATH,
    PLANTVILLAGE_38_RAW_CLASSES,
)


class DiseasePredictor:
    def __init__(self, model_name_or_path: Optional[str] = None):
        self.model_name = model_name_or_path or os.getenv(
            "PLANT_DISEASE_MODEL_NAME",
            DISEASE_HF_MODEL_NAME,
        )
        self.keras_model_path = os.getenv("PLANT_DISEASE_KERAS_MODEL_PATH", LOCAL_KERAS_PATH)
        self.threshold = float(os.getenv("DISEASE_CONFIDENCE_THRESHOLD", str(DISEASE_MEDIUM_CONF_THRESHOLD)))
        self.model = None
        self.processor = None
        self.keras_model = None
        self.is_loaded = False
        self.backend = "uninitialized"

    def load_model(self) -> None:
        """Load model into memory once. Raises RuntimeError if loading fails."""
        if self.is_loaded:
            return

        if self.keras_model_path and os.path.exists(self.keras_model_path):
            try:
                import tensorflow as tf

                self.keras_model = tf.keras.models.load_model(self.keras_model_path)
                self.backend = "keras"
                self.is_loaded = True
                return
            except Exception as exc:
                raise RuntimeError(f"[ML Service] Failed to load Keras model '{self.keras_model_path}': {exc}") from exc

        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForImageClassification,
                MobileNetV2ImageProcessor,
            )

            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.eval()
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            except Exception:
                self.processor = MobileNetV2ImageProcessor.from_pretrained(self.model_name)
            self.backend = "pytorch"
            self.is_loaded = True
        except Exception as exc:
            self.backend = "error"
            self.is_loaded = False
            raise RuntimeError(
                f"[ML Service] Unable to load disease classification model '{self.model_name}': {exc}"
            ) from exc

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Execute inference and return structured diagnosis without fake fallbacks."""
        if not self.is_loaded:
            self.load_model()

        img = ImageOps.exif_transpose(image)
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.backend == "pytorch" and self.model is not None and self.processor is not None:
            with torch.no_grad():
                inputs = self.processor(images=img, return_tensors="pt")
                logits = self.model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
        elif self.backend == "keras" and self.keras_model is not None:
            resized = img.resize((128, 128), Image.Resampling.BILINEAR)
            arr = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
            probs = np.asarray(self.keras_model.predict(arr)[0], dtype=np.float32)
        else:
            raise RuntimeError("DiseasePredictor model is not initialized.")

        top_indices = np.argsort(probs)[::-1][:5]
        top_idx = int(top_indices[0])
        top_score = float(probs[top_idx])
        top_label = PLANTVILLAGE_38_RAW_CLASSES[top_idx]

        top_candidates: List[Dict[str, Any]] = [
            {
                "rawLabel": PLANTVILLAGE_38_RAW_CLASSES[int(i)],
                "label": normalize_disease_label(PLANTVILLAGE_38_RAW_CLASSES[int(i)]),
                "confidence": float(probs[int(i)]),
                "confidencePct": round(float(probs[int(i)]) * 100, 1),
            }
            for i in top_indices
        ]

        info = get_disease_info(top_label)
        crop = info.get("crop", "Plant")
        condition = info.get("condition", "Healthy")
        health_status = info.get("health_status", "Healthy" if "healthy" in top_label.lower() else "Diseased")
        is_uncertain = top_score < self.threshold

        return {
            "success": True,
            "plant": crop,
            "disease": condition,
            "diseaseName": (
                f"{crop} {condition}"
                if (health_status != "Healthy" and crop not in condition)
                else (condition if health_status != "Healthy" else f"Healthy {crop}")
            ),
            "healthStatus": health_status,
            "confidence": round(top_score * 100, 1),
            "rawLabel": top_label,
            "pathogen": info.get("pathogen", "N/A"),
            "severity": info.get("severity", "N/A"),
            "symptoms": info.get("symptoms", ""),
            "recommendations": info.get("recommendations", []),
            "preventiveMeasures": info.get("preventive_measures", []),
            "isUncertain": is_uncertain,
            "uncertaintyMessage": (
                f"Prediction confidence ({top_score * 100:.1f}%) is below diagnostic threshold ({self.threshold * 100:.0f}%). "
                "Please capture a clearer image of the affected plant leaf."
            )
            if is_uncertain
            else None,
            "topCandidates": top_candidates,
            "backend": self.backend,
        }


predictor = DiseasePredictor()
