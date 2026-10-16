"""
PlantVerse — Comprehensive Test Suite for Plant Disease Detection
================================================================
Implements automated verification for all 8 test scenarios defined in Phase 18.
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

from plant_disease_detection import predict_disease, preprocess_image
from disease_recommendations import get_disease_info, normalize_disease_label
from ui.presentation import load_persistent_diagnoses, save_persistent_diagnoses, upsert_disease_diagnosis
from ar_location_plant_map import PlantLocationFinder
from plant_identification import predict_species, translate_text


class TestPlantDiseaseIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.healthy_leaf_path = os.path.join(PROJECT_ROOT, "static/sample_leaves/healthy_tomato.jpg")
        cls.diseased_leaf_path = os.path.join(PROJECT_ROOT, "static/sample_leaves/tomato_early_blight.jpg")
        cls.apple_scab_path = os.path.join(PROJECT_ROOT, "static/sample_leaves/apple_scab.jpg")

    def test_01_healthy_leaf_prediction(self):
        """Test 1: Upload healthy leaf -> Expected: Healthy prediction."""
        self.assertTrue(os.path.exists(self.healthy_leaf_path), "Healthy tomato image missing")
        result = predict_disease(self.healthy_leaf_path)

        self.assertTrue(result["success"])
        self.assertEqual(result["health_status"], "Healthy")
        self.assertIn("Tomato", result["plant"])
        self.assertGreater(result["confidence"], 0.50)
        self.assertFalse(result["is_uncertain"])
        print(f"\n[PASS] Test 1: Healthy Leaf -> {result['disease_name']} ({result['confidence_pct']}%)")

    def test_02_diseased_leaf_prediction(self):
        """Test 2: Upload diseased leaf -> Expected: Disease + confidence."""
        self.assertTrue(os.path.exists(self.diseased_leaf_path), "Diseased tomato image missing")
        result = predict_disease(self.diseased_leaf_path)

        self.assertTrue(result["success"])
        self.assertEqual(result["health_status"], "Diseased")
        self.assertIn("Tomato", result["plant"])
        self.assertIn("Early Blight", result["disease"])
        self.assertGreater(result["confidence"], 0.60)
        self.assertGreater(len(result["recommendations"]), 0)
        print(f"[PASS] Test 2: Diseased Leaf -> {result['disease_name']} ({result['confidence_pct']}%)")

    def test_03_invalid_file_handling(self):
        """Test 3: Upload invalid file -> Expected: Friendly error."""
        # 3a. Random text string as file content
        invalid_bytes = b"This is not a real image file, just corrupt text bytes."
        with self.assertRaises(ValueError) as ctx:
            preprocess_image(invalid_bytes)
        self.assertIn("Unable to process image file", str(ctx.exception))

        # 3b. Empty file
        with self.assertRaises(ValueError) as ctx:
            preprocess_image(b"")
        self.assertIn("Image data is empty", str(ctx.exception))

        # 3c. None input
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
            "severity": "High",
            "recommendations": ["Prune lower leaves", "Apply copper spray"],
            "image_bytes": b"fake_jpeg_bytes_for_testing",
            "is_sighting": True,
            "latitude": 28.6139,
            "longitude": 77.2090,
            "location_name": "Field Trial Plot A, New Delhi",
            "notes": "Spotted after recent monsoon rainfall."
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
        # Simulated reload: read directly from disk storage
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

    def test_07_ml_service_unavailable_fallback(self):
        """Test 7: ML service unavailable -> Expected: Graceful frontend/backend error or in-process fallback."""
        # Set an unreachable external URL to test fallback
        orig_url = os.environ.get("PLANT_DISEASE_API_URL")
        try:
            os.environ["PLANT_DISEASE_API_URL"] = "http://127.0.0.1:9999/unreachable"
            # Should not crash; falls back gracefully to in-process model
            res = predict_disease(self.diseased_leaf_path)
            self.assertTrue(res["success"])
            self.assertEqual(res["health_status"], "Diseased")
        finally:
            if orig_url:
                os.environ["PLANT_DISEASE_API_URL"] = orig_url
            else:
                os.environ.pop("PLANT_DISEASE_API_URL", None)

        print("[PASS] Test 7: ML Service Unavailable -> Gracefully fell back without application crash")

    def test_08_existing_plantverse_features(self):
        """Test 8: Existing Plantverse features -> Expected: Everything continues working."""
        # 8a. Haversine distance calculation in PlantLocationFinder
        finder = PlantLocationFinder()
        dist = finder.calculate_distance(28.6139, 77.2090, 19.0760, 72.8777)
        self.assertGreater(dist, 1100.0)  # Delhi to Mumbai ~ 1150 km
        self.assertLess(dist, 1300.0)

        # 8b. Translation helper function
        trans = translate_text("Hello", "en")
        self.assertEqual(trans, "Hello")

        # 8c. Knowledge base normalized lookups
        info = get_disease_info("Tomato___Early_blight")
        self.assertEqual(info["crop"], "Tomato")
        self.assertEqual(info["health_status"], "Diseased")

        print("[PASS] Test 8: Existing PlantVerse Features -> Distance, translation, and taxonomy intact")

    def test_09_stream_preview_rewind(self):
        """Test 9: Streamlit file upload after preview -> Expected: Stream pointer rewound and diagnosed without 0 bytes error."""
        with open(self.diseased_leaf_path, "rb") as f:
            stream = io.BytesIO(f.read())

        # Simulate Streamlit Image.open preview reading the stream
        preview = Image.open(stream)
        preview.load()
        stream.seek(0, 2)  # Place at EOF

        # Test predict_disease on stream at EOF
        result = predict_disease(stream)
        self.assertTrue(result["success"])
        self.assertEqual(result["health_status"], "Diseased")
        self.assertIn("Tomato", result["plant"])

        # Also test with instantiated PIL Image
        pil_img = Image.open(self.healthy_leaf_path)
        pil_result = predict_disease(pil_img)
        self.assertTrue(pil_result["success"])
        self.assertEqual(pil_result["health_status"], "Healthy")

        print("[PASS] Test 9: Stream Preview Rewind & PIL Input -> Successfully handled without 0 bytes error")


if __name__ == "__main__":
    unittest.main()

