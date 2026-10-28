# AgroVision AI — Crop Health Checker

AgroVision AI is a beginner-friendly website for uploading a crop-leaf photo and viewing a prediction from the local Flask API.

> **Model weights are not included.** The API does not invent predictions. It returns a clear `503` response from `/api/predict` until a real compatible Keras model and its exact class labels are installed.

## What the project does

- `/` serves the website.
- `GET /api/health` reports whether model files are present.
- `GET /api/diseases` lists information only for known classes in the model label file.
- `POST /api/predict` accepts an image in a multipart form field called `image`.
- The website previews an image before upload and displays prediction confidence, disease guidance, and errors from the API.
- Checks image format and file size, rejects very large images, and does not save uploads on disk.
- Loads a local Keras model once and reuses it across API requests.
- Reports low model scores as uncertain instead of presenting them as a confident diagnosis.

## Requirements

- Windows 10 or 11.
- Python 3.10 or 3.11, 64-bit, recommended for TensorFlow compatibility.
- VS Code (recommended).

TensorFlow support depends on Python version and platform. If TensorFlow does not install on Windows, use a supported version in WSL2/Linux and follow TensorFlow's platform installation requirements. The API tests do not need TensorFlow because they use a small mocked model.

## Set up in the VS Code PowerShell terminal

Open this project folder in VS Code. Then create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents environment activation, run this in the current terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Start the API

```powershell
python app.py
```

The local API listens at **http://127.0.0.1:5000**. Stop it with `Ctrl+C`.

This built-in server is for local learning only. Do not expose it to the public internet or use it as a production server.

## Try the API

Open the website in your browser:

```text
http://127.0.0.1:5000/
```

Check model availability from the API:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
```

After you install a real model, send an image:

```powershell
curl.exe -X POST -F "image=@C:\path\to\leaf.jpg" http://127.0.0.1:5000/api/predict
```

Accepted formats are JPG, JPEG, and PNG. The maximum image file size is 8 MB and the maximum image dimensions are 20 megapixels.

## Add a real model

Place these files in the `model` folder:

1. `model/model.keras` — a genuine compatible Keras classifier.
2. `model/class_names.json` — metadata for that exact model, including its label order and preprocessing.

No model weights are included or downloaded automatically. Install a real model and its matching labels from a source whose license and use conditions you have checked. Without model files the website still opens, but prediction requests show that a model is not installed.

### Expected JSON format

```json
{
  "class_names": [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato_healthy",
    "Potato_healthy"
  ],
  "image_size": [224, 224],
  "rescaling": "0_1",
  "confidence_threshold": 0.6,
  "model_architecture": "Document the actual architecture",
  "dataset": "Document the actual dataset and version"
}
```

Replace the example names with the **actual model output names and order**. The example is not a claim that those classes are installed or supported.

The backend expects RGB images shaped as `(batch, height, width, 3)`. By default it resizes images to 224 by 224 and divides pixels by 255 (`rescaling: "0_1"`). Use `rescaling: "none"` only when that matches the actual training preprocessing. Model output must be a softmax probability vector with one output per label.

Scores below `confidence_threshold` (default 0.60) set `uncertain` to true. This threshold is a demonstration setting, not validated real-world accuracy. Even a high score can be wrong.

## Run the tests

With the virtual environment activated:

```powershell
python -m unittest discover -s tests -v
```

The tests cover API responses, invalid images, missing model behavior, preprocessing, label mapping, and uncertainty. They mock Keras inference and do not test real trained weights.

## Project structure

```text
AgroVision-AI/
├── app.py
├── requirements.txt
├── README.md
├── static/
│   ├── app.js
│   └── styles.css
├── templates/
│   └── index.html
├── ai/
│   ├── __init__.py
│   └── predictor.py
├── model/
│   └── README.md
└── tests/
    └── test_api.py
```

## Important limitations

Image classification is only a prediction. Field lighting, backgrounds, crop varieties, pests, nutrient stress, and multiple plant conditions can make results wrong. This API is an educational project, not a substitute for an agricultural expert. Confirm serious or spreading symptoms locally before taking action. No pesticide dosage or treatment is prescribed.

## Deploy from GitHub

This repository includes `render.yaml` for a one-click deployment on Render. Open the [Render deployment page](https://render.com/deploy?repo=https://github.com/charithaavula95/agrovision-ai), sign in, connect GitHub, and approve the `agrovision-ai` blueprint. Render builds the app and gives you a public `onrender.com` URL when deployment finishes.

The deployed website works without a model, but cannot make predictions until compatible model files are installed. For local inference with a model, install `requirements-model.txt`; the standard `requirements.txt` stays lightweight for hosting.
