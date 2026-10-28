"""Preprocess images and run a locally installed Keras classifier."""

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)
DEFAULT_CONFIDENCE_THRESHOLD = 0.60


class ModelUnavailableError(RuntimeError):
    """Raised when a usable local model or its labels are missing."""


class PredictionError(RuntimeError):
    """Raised when an image cannot be prepared or inference fails."""


def get_model_config(model_dir):
    """Read model labels and preprocessing details from class_names.json."""
    model_dir = Path(model_dir)
    model_path = model_dir / "model.keras"
    labels_path = model_dir / "class_names.json"

    if not model_path.is_file():
        raise ModelUnavailableError(
            "AI model is not installed. Add model/model.keras and model/class_names.json; see README.md."
        )
    if not labels_path.is_file():
        raise ModelUnavailableError(
            "Model labels are missing. Add model/class_names.json with the model's exact output order."
        )

    try:
        with labels_path.open("r", encoding="utf-8") as labels_file:
            config = json.load(labels_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ModelUnavailableError(f"Could not read model/class_names.json: {error}") from error

    if not isinstance(config, dict):
        raise ModelUnavailableError("class_names.json must be a JSON object with a class_names list.")

    class_names = config.get("class_names")
    if (
        not isinstance(class_names, list)
        or len(class_names) < 2
        or not all(isinstance(name, str) and name.strip() for name in class_names)
    ):
        raise ModelUnavailableError("class_names must contain at least two non-empty class names.")
    return config


@lru_cache(maxsize=2)
def load_model(model_path):
    """Load model weights once and keep them cached between requests."""
    try:
        from tensorflow import keras
    except ImportError as error:
        raise ModelUnavailableError(
            "TensorFlow is not installed. Install the project requirements and restart the API."
        ) from error

    try:
        return keras.models.load_model(model_path)
    except (OSError, ValueError, TypeError, ImportError) as error:
        logger.exception("Could not load the configured Keras model.")
        raise ModelUnavailableError(
            "Could not load model/model.keras. Check that it is a compatible Keras model file."
        ) from error


def preprocess_image(image, config):
    """Resize and scale a Pillow image according to its model metadata."""
    image_size = config.get("image_size", [224, 224])
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or not all(type(size) is int and size > 0 for size in image_size)
    ):
        raise PredictionError("image_size must be [height, width] in class_names.json.")

    resized = image.convert("RGB").resize(
        (image_size[1], image_size[0]),
        Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(resized, dtype=np.float32)

    rescaling = config.get("rescaling", "0_1")
    if rescaling == "0_1":
        pixels /= 255.0
    elif rescaling != "none":
        raise PredictionError("rescaling must be either '0_1' or 'none' in class_names.json.")
    return np.expand_dims(pixels, axis=0)


def normalize_class_name(label):
    """Convert common PlantVillage class labels into a stable key."""
    return label.strip().lower().replace("___", "_").replace(" ", "_").replace("-", "_")


def predict_image(image, model_dir):
    """Return the top class and its score, marking low scores as uncertain."""
    config = get_model_config(model_dir)
    model_path = str((Path(model_dir) / "model.keras").resolve())
    model = load_model(model_path)
    batch = preprocess_image(image, config)

    try:
        scores = np.asarray(model.predict(batch, verbose=0), dtype=np.float64)
    except (ValueError, RuntimeError, OSError) as error:
        logger.exception("Keras inference failed.")
        raise PredictionError("The model could not analyze this image. Check the model input settings.") from error

    class_names = config["class_names"]
    if scores.ndim != 2 or scores.shape != (1, len(class_names)):
        raise ModelUnavailableError("Model output count does not match class_names.json.")

    probabilities = scores[0]
    if not np.all(np.isfinite(probabilities)):
        raise PredictionError("The model returned invalid scores. Check the installed model.")
    if (
        np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(probabilities.sum(), 1.0, atol=0.02)
    ):
        raise ModelUnavailableError("The model must return softmax probabilities in the class_names.json order.")

    threshold = config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ModelUnavailableError("confidence_threshold must be a number from 0 to 1.")

    best_index = int(np.argmax(probabilities))
    label = class_names[best_index].strip()
    confidence = float(probabilities[best_index])
    class_key = normalize_class_name(label)
    uncertain = confidence < threshold
    return {
        "class_name": label,
        "class_key": class_key,
        "classification": (
            "uncertain"
            if uncertain
            else "healthy"
            if "healthy" in class_key
            else "potentially_diseased"
        ),
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 1),
        "uncertain": uncertain,
        "uncertainty_threshold_percent": round(threshold * 100),
    }
