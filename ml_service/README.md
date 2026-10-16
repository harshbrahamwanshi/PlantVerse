# PlantVerse — Plant Disease Detection ML Service

Standalone, isolated Python machine learning microservice for plant leaf disease prediction.

## Overview
Based on the convolutional neural network architecture and 38-class PlantVillage dataset taxonomy from [animesh1012/machineLearning/Plant_Disease_Prediction](https://github.com/animesh1012/machineLearning/tree/main/Plant_Disease_Prediction).

Exposes standard REST endpoints for image classification, health status evaluation, and diagnosis logging.

## Directory Structure
```
ml_service/
├── app.py                     # ASGI Web Service (Health, Predict, Diagnosis)
├── model/                     # Directory for custom .keras or .h5 weights
├── services/
│   └── disease_predictor.py   # Model loading and inference engine
├── utils/
│   └── preprocessing.py       # Image validation and tensor resizing
├── requirements.txt           # Python dependencies
└── README.md                  # Service documentation
```

## Running the Service

### 1. Install Dependencies
```bash
pip install -r ml_service/requirements.txt
```

### 2. Start the Server
```bash
uvicorn ml_service.app:app --host 0.0.0.0 --port 8000
```
Or directly:
```bash
python ml_service/app.py
```

## API Endpoints

### `GET /health`
Returns service status, model loading status, and active backend.

**Response:**
```json
{
  "status": "healthy",
  "service": "PlantVerse Plant Disease Prediction ML API",
  "version": "1.0.0",
  "uptimeSeconds": 14.2,
  "modelBackend": "transformers",
  "modelLoaded": true,
  "threshold": 0.5
}
```

### `POST /api/disease/predict`
Accepts `multipart/form-data` with an `image` field.

**Example cURL Request:**
```bash
curl -X POST http://localhost:8000/api/disease/predict \
  -F "image=@static/sample_leaves/tomato_early_blight.jpg"
```

**Response:**
```json
{
  "success": true,
  "plant": "Tomato",
  "disease": "Early Blight",
  "diseaseName": "Tomato Early Blight",
  "healthStatus": "Diseased",
  "confidence": 78.4,
  "rawLabel": "Tomato with Early Blight",
  "pathogen": "Alternaria solani (Fungus)",
  "severity": "High",
  "symptoms": "Concentric dark brown 'target-like' rings on older lower leaves...",
  "recommendations": [
    "Prune off all infected lower leaves touching or close to the ground using sanitized shears.",
    "Apply organic copper fungicide, chlorothalonil, or biofungicide immediately.",
    "Mulch soil heavily to prevent rain-splash of soilborne spores.",
    "Avoid overhead watering; irrigate strictly at the base in early morning."
  ],
  "preventiveMeasures": [
    "Stake or cage tomato plants to maintain upright, airy growth.",
    "Rotate crops annually with non-solanaceous species.",
    "Select early blight-tolerant varieties."
  ],
  "isUncertain": false,
  "topCandidates": [
    {
      "label": "Tomato with Early Blight",
      "confidence": 0.784,
      "confidencePct": 78.4
    }
  ],
  "backend": "transformers"
}
```

### `POST /api/disease/diagnosis`
Validate or save a structured diagnosis record.

**Request Body (JSON):**
```json
{
  "plant": "Tomato",
  "disease": "Early Blight",
  "healthStatus": "Diseased",
  "confidence": 78.4,
  "location": {
    "latitude": 28.6139,
    "longitude": 77.2090
  }
}
```

## Model Configuration
- **Hugging Face Model (Default)**: Set via `PLANT_DISEASE_MODEL_NAME` (default: `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`).
- **Local Keras Model**: Place `trained_plant_disease_model.keras` into `ml_service/model/` and set `PLANT_DISEASE_KERAS_MODEL_PATH=ml_service/model/trained_plant_disease_model.keras`.
- **Diagnostic Confidence Threshold**: Set via `DISEASE_CONFIDENCE_THRESHOLD` (default: `0.50`).
