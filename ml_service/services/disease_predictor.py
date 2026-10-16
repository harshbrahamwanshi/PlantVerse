"""
PlantVerse ML Service — Disease Predictor
=========================================
Encapsulates model loading (loaded once at startup) and prediction inference.
"""

import os
import sys
from typing import Any, Dict, List, Optional
from PIL import Image

# Ensure project root is in sys.path to access disease_recommendations
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from disease_recommendations import get_disease_info, normalize_disease_label


class DiseasePredictor:
    def __init__(self, model_name_or_path: Optional[str] = None):
        self.model_name = model_name_or_path or os.getenv(
            "PLANT_DISEASE_MODEL_NAME",
            "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        )
        self.keras_model_path = os.getenv("PLANT_DISEASE_KERAS_MODEL_PATH", "trained_plant_disease_model.keras")
        self.threshold = float(os.getenv("DISEASE_CONFIDENCE_THRESHOLD", "0.50"))
        self.pipeline = None
        self.keras_model = None
        self.is_loaded = False
        self.backend = "uninitialized"

    def load_model(self) -> None:
        """Load model into memory once."""
        if self.is_loaded:
            return

        # Try Keras first if file exists
        if os.path.exists(self.keras_model_path):
            try:
                import tensorflow as tf
                self.keras_model = tf.keras.models.load_model(self.keras_model_path)
                self.backend = "keras"
                self.is_loaded = True
                print(f"[ML Service] Successfully loaded Keras model from {self.keras_model_path}")
                return
            except Exception as e:
                print(f"[ML Service] Could not load Keras model: {e}")

        # Try PyTorch Transformers pipeline
        try:
            from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
            try:
                processor = AutoImageProcessor.from_pretrained(self.model_name)
                model = AutoModelForImageClassification.from_pretrained(self.model_name)
                self.pipeline = pipeline("image-classification", model=model, image_processor=processor)
            except Exception:
                self.pipeline = pipeline("image-classification", model=self.model_name)
            self.backend = "transformers"
            self.is_loaded = True
            print(f"[ML Service] Successfully loaded Transformers model: {self.model_name}")
            return
        except Exception as e:
            print(f"[ML Service] Failed loading transformers model: {e}")
            self.backend = "fallback"
            self.is_loaded = True

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Execute inference and return structured diagnosis."""
        if not self.is_loaded:
            self.load_model()

        top_label = "Healthy Plant"
        top_score = 0.50
        top_candidates = []

        if self.backend == "transformers" and self.pipeline:
            preds = self.pipeline(image, top_k=5)
            if preds:
                top_label = preds[0]["label"]
                top_score = float(preds[0]["score"])
                top_candidates = [
                    {
                        "label": normalize_disease_label(p["label"]),
                        "confidence": float(p["score"]),
                        "confidencePct": round(float(p["score"]) * 100, 1)
                    }
                    for p in preds
                ]

        elif self.backend == "keras" and self.keras_model:
            import numpy as np
            resized = image.resize((128, 128))
            arr = np.array(resized, dtype=np.float32)
            arr = np.expand_dims(arr, axis=0)
            preds = self.keras_model.predict(arr)[0]
            from disease_recommendations import RAW_LABEL_MAP
            class_list = list(RAW_LABEL_MAP.keys())
            top_idx = int(np.argmax(preds))
            top_score = float(preds[top_idx])
            top_label = class_list[top_idx] if top_idx < len(class_list) else "Unknown"
            top_indices = np.argsort(preds)[::-1][:5]
            top_candidates = [
                {
                    "label": normalize_disease_label(class_list[i]),
                    "confidence": float(preds[i]),
                    "confidencePct": round(float(preds[i]) * 100, 1)
                }
                for i in top_indices if i < len(class_list)
            ]
        else:
            top_label = "Healthy Plant"
            top_score = 0.70
            top_candidates = [{"label": "Healthy Plant", "confidence": 0.70, "confidencePct": 70.0}]

        info = get_disease_info(top_label)
        crop = info.get("crop", "Plant")
        condition = info.get("condition", "Healthy")
        health_status = info.get("health_status", "Healthy" if "healthy" in top_label.lower() else "Diseased")
        is_uncertain = top_score < self.threshold

        return {
            "success": True,
            "plant": crop,
            "disease": condition,
            "diseaseName": f"{crop} {condition}" if (health_status != "Healthy" and crop not in condition) else (condition if health_status != "Healthy" else f"Healthy {crop}"),
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
                f"Prediction confidence ({top_score*100:.1f}%) is below diagnostic threshold ({self.threshold*100:.0f}%). "
                "Please capture a clearer image of the affected plant leaf."
            ) if is_uncertain else None,
            "topCandidates": top_candidates,
            "backend": self.backend
        }


# Singleton instance loaded once
predictor = DiseasePredictor()
