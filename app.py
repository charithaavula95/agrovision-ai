"""Beginner-friendly Flask API for AgroVision AI."""

import io
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from ai.predictor import (
    ModelUnavailableError,
    PredictionError,
    get_model_config,
    normalize_class_name,
    predict_image,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Extra space allows multipart form headers around the 8 MB image.
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + 1024 * 1024


DISEASE_GUIDES = {
    "tomato_early_blight": {
        "name": "Tomato early blight",
        "crop": "Tomato",
        "symptoms": "Brown, target-like spots often start on older leaves; yellowing and leaf drop may follow.",
        "prevention": "Use healthy planting material, rotate crops, improve airflow, water near the soil, and remove infected debris.",
        "next_steps": "Inspect nearby plants and ask a local agricultural expert to confirm the cause.",
    },
    "tomato_late_blight": {
        "name": "Tomato late blight",
        "crop": "Tomato",
        "symptoms": "Dark, water-soaked areas can expand quickly; pale growth may appear beneath leaves in humid conditions.",
        "prevention": "Use healthy planting material, improve airflow, avoid prolonged leaf wetness, and remove infected debris.",
        "next_steps": "If symptoms are spreading quickly, contact a local agricultural expert promptly.",
    },
    "potato_early_blight": {
        "name": "Potato early blight",
        "crop": "Potato",
        "symptoms": "Dark spots, often with concentric rings, commonly begin on older leaves; nearby tissue may yellow.",
        "prevention": "Rotate crops, use healthy seed potatoes, encourage airflow, and clear crop residues.",
        "next_steps": "Check nearby plants and ask a qualified local expert to confirm the cause.",
    },
    "potato_late_blight": {
        "name": "Potato late blight",
        "crop": "Potato",
        "symptoms": "Water-soaked or dark lesions may expand quickly; pale growth can appear under leaves in humid weather.",
        "prevention": "Plant healthy seed potatoes, avoid extended leaf wetness, and manage volunteer plants.",
        "next_steps": "Seek local expert confirmation, especially when symptoms are spreading.",
    },
    "tomato_healthy": {
        "name": "Healthy tomato leaf",
        "crop": "Tomato",
        "symptoms": "The model class indicates a resemblance to its healthy-leaf examples; it does not rule out other problems.",
        "prevention": "Monitor plants regularly and provide appropriate water and airflow.",
        "next_steps": "Check the plant again if symptoms appear and seek expert advice if it declines.",
    },
    "potato_healthy": {
        "name": "Healthy potato leaf",
        "crop": "Potato",
        "symptoms": "The model class indicates a resemblance to its healthy-leaf examples; it does not rule out other problems.",
        "prevention": "Monitor plants regularly and use healthy seed potatoes.",
        "next_steps": "Check the plant again if symptoms appear and seek expert advice if it declines.",
    },
}


def json_error(message, status_code):
    """Return one consistent JSON shape for API errors."""
    return jsonify({"error": message}), status_code


def get_model_status():
    """Return model metadata if the model and its labels are installed."""
    try:
        return get_model_config(MODEL_DIR), None
    except ModelUnavailableError as error:
        return None, str(error)


@app.get("/")
def home():
    """Render the browser-based crop health checker."""
    return render_template("index.html")


@app.get("/api/health")
def health():
    """Report whether the API can find its local model files."""
    config, model_error = get_model_status()
    return jsonify(
        {
            "status": "ok",
            "model_available": config is not None,
            "model_error": model_error,
        }
    )


@app.get("/api/diseases")
def diseases():
    """List guides only for classes configured in the actual model labels."""
    config, model_error = get_model_status()
    if config is None:
        return jsonify(
            {
                "model_available": False,
                "diseases": [],
                "message": model_error,
            }
        )

    guides = []
    seen_classes = set()
    for label in config["class_names"]:
        class_key = normalize_class_name(label)
        if class_key in DISEASE_GUIDES and class_key not in seen_classes:
            guides.append({"class_name": label, **DISEASE_GUIDES[class_key]})
            seen_classes.add(class_key)

    return jsonify({"model_available": True, "diseases": guides})


@app.post("/api/predict")
def predict():
    """Validate an image and ask the local Keras model for a prediction."""
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return json_error("Send an image using the multipart form field named 'image'.", 400)

    image_bytes = uploaded_file.stream.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return json_error("Image is larger than 8 MB. Choose a smaller image.", 413)

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG"}:
                return json_error("Unsupported image format. Use a JPG, JPEG, or PNG image.", 400)
            if image.width * image.height > MAX_IMAGE_PIXELS:
                return json_error("Image has too many pixels. Choose an image under 20 megapixels.", 400)
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            leaf_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return json_error("The uploaded file is not a readable JPG or PNG image.", 400)

    try:
        result = predict_image(leaf_image, MODEL_DIR)
    except ModelUnavailableError as error:
        return json_error(str(error), 503)
    except PredictionError as error:
        return json_error(str(error), 422)

    guide = None if result["uncertain"] else DISEASE_GUIDES.get(result["class_key"])
    return jsonify(
        {
            "prediction": result,
            "guide": guide,
            "disclaimer": "AI scores are estimates, not guaranteed accuracy or a professional diagnosis.",
        }
    )


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    """Keep Flask's request-size error response in the API's JSON format."""
    return json_error("Request is too large. Image files must be 8 MB or smaller.", 413)
if __name__ == "__main__":
    # This local development server is for learning and testing only.
    app.run(host="127.0.0.1", port=5000, debug=False)
