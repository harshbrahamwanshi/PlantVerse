"""
PlantVerse ML Service — Disease Predictor (`ml_service/services/disease_predictor.py`)
=====================================================================================
Synchronized with the main PlantVerse application (`plant_disease_detection.py` and
`ml_config.py`) so that the FastAPI microservice and Streamlit app use the exact
same models, preprocessing, label mappings, consistency gating, and confidence thresholds.
"""

import os
import sys
from typing import Any, Dict, Optional
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_config import (
    DISEASE_BACKEND,
    DISEASE_HF_MODEL_NAME,
    DISEASE_MEDIUM_CONF_THRESHOLD,
    LOCAL_KERAS_PATH,
    ONNX_DISEASE_MODEL_PATH,
)
from plant_disease_detection import load_disease_classifier, predict_disease


class DiseasePredictor:
    def __init__(self, model_name_or_path: Optional[str] = None):
        self.model_name = model_name_or_path or os.getenv(
            "PLANT_DISEASE_MODEL_NAME",
            str(ONNX_DISEASE_MODEL_PATH) if DISEASE_BACKEND == "onnx_vit" else DISEASE_HF_MODEL_NAME,
        )
        self.keras_model_path = os.getenv("PLANT_DISEASE_KERAS_MODEL_PATH", LOCAL_KERAS_PATH)
        self.threshold = float(os.getenv("DISEASE_CONFIDENCE_THRESHOLD", str(DISEASE_MEDIUM_CONF_THRESHOLD)))
        self.is_loaded = False
        self.backend = "uninitialized"

    def load_model(self) -> None:
        """Load the unified disease classifier once. Raises RuntimeError on failure."""
        if self.is_loaded:
            return
        bundle = load_disease_classifier()
        self.backend = bundle.get("type", "onnx_vit")
        self.is_loaded = True

    def predict(self, image: Image.Image, host_plant: Optional[str] = None) -> Dict[str, Any]:
        """Execute inference using the unified PlantVerse pipeline (no fake fallbacks)."""
        if not self.is_loaded:
            self.load_model()

        res = predict_disease(image, threshold=self.threshold, host_plant=host_plant)
        return {
            "success": res["success"],
            "plant": res["plant"],
            "disease": res["disease"],
            "diseaseName": res["disease_name"],
            "healthStatus": res["health_status"],
            "isSupportedCrop": res.get("is_supported_crop", True),
            "confidence": res["confidence_pct"],
            "confidenceTier": res.get("confidence_tier", "high"),
            "rawLabel": res["raw_label"],
            "pathogen": res["pathogen"],
            "severity": res["severity"],
            "affectedAreaPct": res.get("affected_area_pct"),
            "derivedHealthScore": res.get("derived_health_score"),
            "symptoms": res["symptoms"],
            "recommendations": res["recommendations"],
            "preventiveMeasures": res["preventive_measures"],
            "chemicalTreatments": res.get("chemical_treatments", []),
            "organicTreatments": res.get("organic_treatments", []),
            "care": res.get("care", {}),
            "isUncertain": res["is_uncertain"],
            "uncertaintyMessage": res["uncertainty_message"] if res["is_uncertain"] else None,
            "topCandidates": [
                {
                    "rawLabel": c.get("raw_label", ""),
                    "label": c["label"],
                    "confidence": c["score"],
                    "confidencePct": c["confidence_pct"],
                }
                for c in res.get("top_candidates", [])
            ],
            "backend": self.backend,
        }


predictor = DiseasePredictor()
