import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import app as agro_app
from ai.predictor import ModelUnavailableError, predict_image, preprocess_image


def create_png_bytes():
    """Create a small in-memory image for API tests."""
    image = Image.new("RGB", (16, 16), color=(30, 120, 45))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class ApiRouteTests(unittest.TestCase):
    def setUp(self):
        agro_app.app.config.update(TESTING=True)
        self.client = agro_app.app.test_client()

    def test_homepage_renders_website(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AgroVision", response.data)
        self.assertIn(b"leaf-upload", response.data)
        self.assertEqual(response.mimetype, "text/html")

    def test_health_reports_missing_model_without_crashing(self):
        with patch.object(agro_app, "MODEL_DIR", Path("missing-model-folder")):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["model_available"])

    def test_disease_list_is_empty_when_model_is_missing(self):
        with patch.object(agro_app, "MODEL_DIR", Path("missing-model-folder")):
            response = self.client.get("/api/diseases")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["diseases"], [])

    def test_disease_list_only_includes_model_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "model.keras").touch()
            (model_dir / "class_names.json").write_text(
                json.dumps({"class_names": ["Tomato___Early_blight", "Other_class"]}),
                encoding="utf-8",
            )
            with patch.object(agro_app, "MODEL_DIR", model_dir):
                response = self.client.get("/api/diseases")
        self.assertEqual(response.status_code, 200)
        diseases = response.get_json()["diseases"]
        self.assertEqual(len(diseases), 1)
        self.assertEqual(diseases[0]["class_name"], "Tomato___Early_blight")

    def test_prediction_requires_image_field(self):
        response = self.client.post("/api/predict")
        self.assertEqual(response.status_code, 400)
        self.assertIn("image", response.get_json()["error"])

    def test_invalid_image_is_rejected(self):
        response = self.client.post(
            "/api/predict",
            data={"image": (io.BytesIO(b"not an image"), "leaf.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    def test_unsupported_image_format_is_rejected(self):
        response = self.client.post(
            "/api/predict",
            data={"image": (io.BytesIO(b"gif data"), "leaf.gif")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_model_returns_clear_service_unavailable(self):
        with patch.object(agro_app, "MODEL_DIR", Path("missing-model-folder")):
            response = self.client.post(
                "/api/predict",
                data={"image": (io.BytesIO(create_png_bytes()), "leaf.png")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("AI model is not installed", response.get_json()["error"])

    def test_valid_image_returns_prediction_shape(self):
        fake_result = {
            "class_name": "Tomato___Early_blight",
            "class_key": "tomato_early_blight",
            "classification": "potentially_diseased",
            "confidence": 0.82,
            "confidence_percent": 82.0,
            "uncertain": False,
            "uncertainty_threshold_percent": 60,
        }
        with patch.object(agro_app, "predict_image", return_value=fake_result):
            response = self.client.post(
                "/api/predict",
                data={"image": (io.BytesIO(create_png_bytes()), "leaf.png")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"]["confidence_percent"], 82.0)
        self.assertEqual(response.get_json()["guide"]["crop"], "Tomato")

    def test_uncertain_prediction_does_not_return_disease_guide(self):
        uncertain_result = {
            "class_name": "Tomato___Early_blight",
            "class_key": "tomato_early_blight",
            "classification": "uncertain",
            "confidence": 0.42,
            "confidence_percent": 42.0,
            "uncertain": True,
            "uncertainty_threshold_percent": 60,
        }
        with patch.object(agro_app, "predict_image", return_value=uncertain_result):
            response = self.client.post(
                "/api/predict",
                data={"image": (io.BytesIO(create_png_bytes()), "leaf.png")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"]["classification"], "uncertain")
        self.assertIsNone(response.get_json()["guide"])

    def test_large_image_file_is_rejected(self):
        with patch.object(agro_app, "MAX_IMAGE_BYTES", 10):
            response = self.client.post(
                "/api/predict",
                data={"image": (io.BytesIO(b"x" * 11), "leaf.png")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 413)


class PredictorTests(unittest.TestCase):
    def test_preprocess_resizes_and_normalizes_image(self):
        image = Image.new("RGB", (4, 4), color=(255, 128, 0))
        batch = preprocess_image(image, {"image_size": [2, 3], "rescaling": "0_1"})
        self.assertEqual(batch.shape, (1, 2, 3, 3))
        np.testing.assert_allclose(batch[0, 0, 0], [1.0, 128 / 255, 0.0], atol=1e-6)

    def test_prediction_uses_label_order_and_uncertainty_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "model.keras").touch()
            (model_dir / "class_names.json").write_text(
                json.dumps(
                    {
                        "class_names": ["Tomato___Early_blight", "Tomato_healthy"],
                        "image_size": [8, 8],
                        "confidence_threshold": 0.9,
                    }
                ),
                encoding="utf-8",
            )

            class FakeModel:
                def predict(self, _batch, verbose=0):
                    return np.array([[0.82, 0.18]])

            with patch("ai.predictor.load_model", return_value=FakeModel()):
                result = predict_image(Image.new("RGB", (8, 8)), model_dir)

        self.assertEqual(result["class_name"], "Tomato___Early_blight")
        self.assertEqual(result["confidence_percent"], 82.0)
        self.assertTrue(result["uncertain"])
        self.assertEqual(result["classification"], "uncertain")

    def test_label_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "model.keras").touch()
            (model_dir / "class_names.json").write_text(
                json.dumps({"class_names": ["Tomato_healthy", "Potato_healthy"]}),
                encoding="utf-8",
            )

            class FakeModel:
                def predict(self, _batch, verbose=0):
                    return np.array([[1.0]])

            with patch("ai.predictor.load_model", return_value=FakeModel()):
                with self.assertRaises(ModelUnavailableError):
                    predict_image(Image.new("RGB", (8, 8)), model_dir)


if __name__ == "__main__":
    unittest.main()
