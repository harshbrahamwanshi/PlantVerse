"""
PlantVerse — Comprehensive Test Suite for Plant Disease Detection
================================================================
Implements automated verification for disease detection, sighting persistence,
error handling, and existing PlantVerse integrations.
"""

import io
import os
import sys
import unittest
from PIL import Image

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ar_location_plant_map import PlantLocationFinder
from disease_recommendations import get_disease_info, normalize_disease_label
from plant_disease_detection import predict_disease, preprocess_image
from plant_identification import predict_species, translate_text
from ui.presentation import (
    load_persistent_diagnoses,
    save_persistent_diagnoses,
)

DIAGNOSES_FILE = os.path.join(PROJECT_ROOT, "data/diagnoses.json")


class TestPlantDiseaseIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.healthy_leaf_path = os.path.join(PROJECT_ROOT, "static/sample_leaves/healthy_tomato.jpg")
        cls.diseased_leaf_path = os.path.join(PROJECT_ROOT, "static/sample_leaves/tomato_early_blight.jpg")
        cls.apple_scab_path = os.path.join(PROJECT_ROOT, "static/sample_leaves/apple_scab.jpg")
        cls._orig_diagnoses_bytes = None
        if os.path.exists(DIAGNOSES_FILE):
            with open(DIAGNOSES_FILE, "rb") as f:
                cls._orig_diagnoses_bytes = f.read()

    @classmethod
    def tearDownClass(cls):
        if cls._orig_diagnoses_bytes is not None:
            with open(DIAGNOSES_FILE, "wb") as f:
                f.write(cls._orig_diagnoses_bytes)

    def test_01_healthy_leaf_prediction(self):
        """Test 1: Upload healthy leaf -> Expected: Healthy prediction."""
        self.assertTrue(os.path.exists(self.healthy_leaf_path), "Healthy tomato image missing")
        result = predict_disease(self.healthy_leaf_path)

        self.assertTrue(result["success"])
        self.assertEqual(result["health_status"], "Healthy")
        self.assertIn("Tomato", result["plant"])
        self.assertGreater(result["confidence"], 0.50)
        self.assertEqual(result["severity"], "Healthy")
        print(f"\n[PASS] Test 1: Healthy Leaf -> {result['disease_name']} ({result['confidence_pct']}%)")

    def test_02_diseased_leaf_prediction(self):
        """Test 2: Upload diseased leaf -> Expected: Disease + confidence + classical CV lesion analysis."""
        self.assertTrue(os.path.exists(self.diseased_leaf_path), "Diseased tomato image missing")
        result = predict_disease(self.diseased_leaf_path)

        self.assertTrue(result["success"])
        self.assertEqual(result["health_status"], "Diseased")
        self.assertIn("Tomato", result["plant"])
        self.assertIn("Early Blight", result["disease"])
        self.assertGreater(result["confidence"], 0.60)
        self.assertGreater(len(result["recommendations"]), 0)
        self.assertIsNotNone(result.get("vision_analysis"))
        print(f"[PASS] Test 2: Diseased Leaf -> {result['disease_name']} ({result['confidence_pct']}%)")

    def test_03_invalid_file_handling(self):
        """Test 3: Upload invalid file -> Expected: Friendly error (never fake fallback)."""
        invalid_bytes = b"This is not a real image file, just corrupt text bytes."
        with self.assertRaises(ValueError) as ctx:
            preprocess_image(invalid_bytes)
        self.assertIn("Unable to process image file", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            preprocess_image(b"")
        self.assertIn("Image data is empty", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            preprocess_image(None)
        self.assertIn("No image provided", str(ctx.exception))

        print("[PASS] Test 3: Invalid File Handling -> Friendly ValueError caught")

    def test_04_save_diagnosis_as_sighting(self):
        """Test 4: Save diagnosis as sighting -> Expected: Diagnosis + location + image correctly associated."""
        test_diag = {
            "id": "test-diag-001",
            "plant_name": "Tomato",
            "disease_name": "Tomato Early Blight",
            "health_status": "Diseased",
            "confidence": 0.85,
            "confidence_pct": 85.0,
            "pathogen": "Alternaria solani",
            "severity": "Moderate",
            "recommendations": ["Prune lower leaves", "Apply copper spray"],
            "image_bytes": b"fake_jpeg_bytes_for_testing",
            "is_sighting": True,
            "latitude": 28.6139,
            "longitude": 77.2090,
            "location_name": "Field Trial Plot A, New Delhi",
            "notes": "Spotted after recent monsoon rainfall.",
        }

        save_persistent_diagnoses([test_diag])
        loaded = load_persistent_diagnoses()
        self.assertGreater(len(loaded), 0)
        match = next((d for d in loaded if d.get("id") == "test-diag-001"), None)
        self.assertIsNotNone(match)
        self.assertEqual(match["plant_name"], "Tomato")
        self.assertEqual(match["disease_name"], "Tomato Early Blight")
        self.assertEqual(match["latitude"], 28.6139)
        self.assertEqual(match["longitude"], 77.2090)
        self.assertTrue(match["is_sighting"])
        print("[PASS] Test 4: Save Diagnosis as Sighting -> Successfully saved with coordinates and notes")

    def test_05_refresh_application_persistence(self):
        """Test 5: Refresh application -> Expected: Saved diagnosis remains available."""
        loaded = load_persistent_diagnoses()
        self.assertGreater(len(loaded), 0)
        self.assertTrue(any(d.get("id") == "test-diag-001" for d in loaded))
        print(f"[PASS] Test 5: Refresh Application -> {len(loaded)} diagnoses recovered from persistent disk")

    def test_06_open_previous_diagnosis(self):
        """Test 6: Open previous diagnosis -> Expected: Correct historical data."""
        loaded = load_persistent_diagnoses()
        record = next(d for d in loaded if d.get("id") == "test-diag-001")
        self.assertEqual(record["plant_name"], "Tomato")
        self.assertEqual(record["disease_name"], "Tomato Early Blight")
        self.assertEqual(record["confidence_pct"], 85.0)
        self.assertEqual(len(record["recommendations"]), 2)
        print("[PASS] Test 6: Open Previous Diagnosis -> Historical payload validated")

    def test_07_ml_service_offline_explicit_error(self):
        """Test 7: ML service offline -> Expected: Descriptive RuntimeError (never fake prediction)."""
        orig_url = os.environ.get("PLANT_DISEASE_API_URL")
        try:
            os.environ["PLANT_DISEASE_API_URL"] = "http://127.0.0.1:9999/unreachable"
            with self.assertRaises(RuntimeError) as ctx:
                predict_disease(self.diseased_leaf_path)
            self.assertIn("offline or failed", str(ctx.exception))
        finally:
            if orig_url:
                os.environ["PLANT_DISEASE_API_URL"] = orig_url
            else:
                os.environ.pop("PLANT_DISEASE_API_URL", None)

        print("[PASS] Test 7: ML Service Offline -> Descriptive RuntimeError raised")

    def test_08_existing_plantverse_features(self):
        """Test 8: Existing PlantVerse features -> Expected: Everything continues working."""
        finder = PlantLocationFinder()
        dist = finder.calculate_distance(28.6139, 77.2090, 19.0760, 72.8777)
        self.assertGreater(dist, 1100.0)
        self.assertLess(dist, 1300.0)

        trans = translate_text("Hello", "en")
        self.assertEqual(trans, "Hello")

        info = get_disease_info("Tomato___Early_blight")
        self.assertEqual(info["crop"], "Tomato")
        self.assertEqual(info["health_status"], "Diseased")

        print("[PASS] Test 8: Existing PlantVerse Features -> Distance, translation, and taxonomy intact")

    def test_09_stream_preview_rewind(self):
        """Test 9: Streamlit file upload after preview -> Expected: Stream pointer rewound and diagnosed without 0 bytes error."""
        with open(self.diseased_leaf_path, "rb") as f:
            stream = io.BytesIO(f.read())

        preview = Image.open(stream)
        preview.load()
        stream.seek(0, 2)

        result = predict_disease(stream)
        self.assertTrue(result["success"])
        self.assertEqual(result["health_status"], "Diseased")
        self.assertIn("Tomato", result["plant"])

        pil_img = Image.open(self.healthy_leaf_path)
        pil_result = predict_disease(pil_img)
        self.assertTrue(pil_result["success"])
        self.assertEqual(pil_result["health_status"], "Healthy")

        print("[PASS] Test 9: Stream Preview Rewind & PIL Input -> Successfully handled without 0 bytes error")


if __name__ == "__main__":
    unittest.main()
