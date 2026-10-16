---
title: PlantVerse — Botanical Intelligence & Pathology Framework
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.46.1"
app_file: app.py
pinned: false
---

# 🌿 PlantVerse — Botanical Intelligence & Plant Pathology

**Discover, diagnose, and explore plants — visually, pathologically, geographically, and culturally.**  
PlantVerse is a comprehensive, multilingual platform combining neural plant identification, foliar disease diagnosis, dynamic biological taxonomy, ethnobotanical pharmacology, and geospatial field habitat mapping.

---

## ✨ Features

- 📷 **Plant Species Identifier:** Upload botanical photographs to predict plant species using deep neural computer vision (FloraSense).
- 🔬 **Plant Disease Detection:** Screen foliage across 38 distinct agricultural crop pathology classes (fungal, bacterial, viral, and healthy conditions) with calibrated confidence percentages.
- 💊 **Actionable Agronomic Recommendations:** Receive evidence-based cultural practices, aeration advice, and remediation guidelines for diagnosed foliar pathologies.
- ⚡ **Unified Identification & Screening Workflow:** Automatically screen for leaf diseases simultaneously during plant identification, correlating species and pathology results.
- 🌐 **Multilingual Localization:** Translate plant taxonomy, disease conditions, and recommendations into Indic languages (Hindi, Telugu, Tamil) and English.
- 📖 **Wikidata & Wikipedia Synthesis:** Traverse taxonomic phylogenetic lineages (Kingdom to Genus) and extract pharmacological notes.
- 🗺️ **Field Observations & Habitat GIS:** Query nearby research-grade wild sightings using the global iNaturalist repository.
- 📍 **Location-Based Disease Sightings:** Associate disease diagnoses with geographic coordinates, observation notes, and photographic vouchers.
- 📜 **Session Herbarium & Diagnosis History:** Automatically catalog identified species and pathology diagnoses, persisting across browser sessions.
- 🚀 **Standalone ML Microservice:** Run inference via native in-process pipelines or a standalone ASGI/REST API (`ml_service/app.py`).

---

## 🛠️ Architecture

```
                    ┌─────────────────────────┐
                    │   User Photograph /     │
                    │   Benchmark Specimen    │
                    └────────────┬────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
        ┌─────────────────────┐     ┌─────────────────────┐
        │  Species Inference  │     │ Pathology Inference │
        │    (FloraSense)     │     │ (PlantVillage CNN / │
        └──────────┬──────────┘     │    MobileNetV2)     │
                   │                └──────────┬──────────┘
                   ▼                           ▼
        ┌─────────────────────┐     ┌─────────────────────┐
        │ Wikidata & Wikipedia│     │ Disease & Pathogen  │
        │ Taxonomy + Medicine │     │   Recommendations   │
        └──────────┬──────────┘     └──────────┬──────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ Unified Specimen Result │
                    │ Plant • Health • Advice │
                    └────────────┬────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
        ┌─────────────────────┐     ┌─────────────────────┐
        │  Digital Herbarium  │     │  Geospatial GIS Map │
        │  Diagnosis History  │     │ Field Observation   │
        └─────────────────────┘     └─────────────────────┘
```

---

## 🦠 Supported Plant Pathology Classes (38 Classes)

Adapted from the PlantVillage dataset referenced by [animesh1012/machineLearning/Plant_Disease_Prediction](https://github.com/animesh1012/machineLearning/tree/main/Plant_Disease_Prediction):

1. **Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy
2. **Blueberry:** Healthy
3. **Cherry:** Powdery Mildew, Healthy
4. **Corn (Maize):** Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
5. **Grape:** Black Rot, Esca (Black Measles), Isariopsis Leaf Blight, Healthy
6. **Orange / Citrus:** Citrus Greening (Huanglongbing)
7. **Peach:** Bacterial Spot, Healthy
8. **Bell Pepper:** Bacterial Spot, Healthy
9. **Potato:** Early Blight, Late Blight, Healthy
10. **Raspberry:** Healthy
11. **Soybean:** Healthy
12. **Squash:** Powdery Mildew
13. **Strawberry:** Leaf Scorch, Healthy
14. **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus (TYLCV), Tomato Mosaic Virus (ToMV), Healthy

---

## 🚀 Running PlantVerse

### 1. Environment Setup
```bash
# Clone the repository
git clone https://huggingface.co/spaces/Aashritha05/PlantVerse
cd PlantVerse

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the PlantVerse Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 3. (Optional) Run the Standalone ML Microservice
PlantVerse can perform inference natively in-process, or you can run the standalone ASGI REST microservice:
```bash
python ml_service/app.py
# Or with uvicorn:
uvicorn ml_service.app:app --host 0.0.0.0 --port 8000
```

To route PlantVerse through the microservice, set:
```bash
export PLANT_DISEASE_API_URL="http://localhost:8000/api/disease/predict"
streamlit run app.py
```

---

## 🔧 Environment Variables & Configuration

| Variable | Default Value | Description |
|---|---|---|
| `PLANT_DISEASE_MODEL_NAME` | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` | Hugging Face model repository ID |
| `DISEASE_CONFIDENCE_THRESHOLD` | `0.50` | Minimum confidence threshold before triggering uncertainty warnings |
| `PLANT_DISEASE_API_URL` | *None* | Optional URL to route requests to an external ML service |
| `PLANT_DISEASE_KERAS_MODEL_PATH`| `trained_plant_disease_model.keras` | Path to local Keras/TF model weights file if using Keras |
| `MAX_IMAGE_SIZE_BYTES` | `10485760` (10 MB) | Maximum upload file size limit |

---

## 📡 REST API Documentation (ML Microservice)

### `GET /health`
Returns service status and active model backend.

### `POST /api/disease/predict`
Accepts `multipart/form-data` with an `image` file field.
Returns JSON with:
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
  "symptoms": "Concentric dark brown rings on older lower leaves...",
  "recommendations": [ ... ],
  "preventiveMeasures": [ ... ],
  "isUncertain": false,
  "topCandidates": [ ... ]
}
```

### `POST /api/disease/diagnosis`
Validates and logs a structured diagnosis record.

---

## 📂 Project Directory Structure

```
PlantVerse/
├── app.py                      # Multi-page Streamlit application entry point
├── plant_disease_detection.py  # Core plant disease detection engine
├── disease_recommendations.py  # 38-class plant pathology knowledge base
├── plant_identification.py     # Species identification & translation module
├── ar_location_plant_map.py    # iNaturalist GIS & Haversine distance module
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
├── app_pages/
│   ├── explore.py              # Overview, hero, four pillars, and gallery
│   ├── identify.py             # Dual Species Identification & Screening Studio
│   ├── disease_detection.py    # Dedicated Foliar Pathology Diagnostic Studio
│   ├── library.py              # Botanical herbarium & diagnosis history
│   ├── ar_explorer.py          # Field GIS map & local pathology sightings
│   └── about.py                # System specifications, architecture & attribution
├── ml_service/
│   ├── app.py                  # ASGI REST API service
│   ├── requirements.txt        # ML microservice dependencies
│   ├── README.md               # Microservice documentation
│   ├── services/
│   │   └── disease_predictor.py# Isolated inference service
│   └── utils/
│       └── preprocessing.py    # Image validation & tensor resizing
├── static/
│   ├── logo.svg
│   ├── hero.jpg
│   ├── ginkgo.jpg
│   ├── herbarium.jpg
│   ├── fern-plate.jpg
│   └── sample_leaves/          # Authentic PlantVillage benchmark test images
├── ui/
│   ├── chrome.py               # Chrome helpers & session state initialization
│   ├── presentation.py         # Taxonomy rendering & diagnosis persistence
│   └── styles.css              # Editorial botanical CSS design system
└── tests/
    └── test_disease_detection.py # Comprehensive test suite (8 test scenarios)
```

---

## 📜 Attribution & References

- **Plant Disease Prediction Base**: The classification taxonomy, model parameters, and training methodology are derived from [animesh1012/machineLearning/Plant_Disease_Prediction](https://github.com/animesh1012/machineLearning/tree/main/Plant_Disease_Prediction).
- **Dataset Foundation**: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (Hughes & Salathé), comprising ~87,000 foliar images across 38 crop condition classes.
- **FloraSense Specimen Classifier**: Sisigoks/FloraSense (Hugging Face).
- **Geospatial & Biodiversity Records**: [iNaturalist Open API](https://api.inaturalist.org/v1).
- **Taxonomic Ontology**: [Wikidata Entity Claims API](https://www.wikidata.org).
