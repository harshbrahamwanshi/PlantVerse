"""
PlantVerse — Centralized ML Pipeline Configuration (`ml_config.py`)
===================================================================
Single source of truth for all model identifiers, file paths, preprocessing
parameters, confidence thresholds, supported plant & crop class lists,
host-to-crop consistency mappings, and lesion severity thresholds.

No scattered constants in page or service modules (Ground Rule 8).
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

# ==============================================================================
# 1. PLANT IDENTIFICATION CONFIGURATION
# ==============================================================================
PLANT_ID_BACKEND = os.getenv("PLANTVERSE_PLANT_ID_BACKEND", "florasense").strip().lower()
FLORASENSE_MODEL_NAME = os.getenv("PLANTVERSE_FLORASENSE_MODEL", "Sisigoks/FloraSense").strip()
ONNX_PLANT_MODEL_PATH = Path(
    os.getenv("PLANTVERSE_ONNX_PLANT_MODEL", str(MODELS_DIR / "plant_model.onnx"))
)
ONNX_PLANT_LABELS_PATH = Path(
    os.getenv("PLANTVERSE_ONNX_PLANT_LABELS", str(MODELS_DIR / "plant_labels.json"))
)

# FloraSense preprocessing (verified against Sisigoks/FloraSense preprocessor_config.json):
# - RGB conversion + EXIF orientation transpose
# - Squash resize to (224, 224) with BILINEAR interpolation (default_to_square=True, do_center_crop=False)
# - Rescale by 1/255.0
# - Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
FLORASENSE_INPUT_SIZE: Tuple[int, int] = (224, 224)
FLORASENSE_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
FLORASENSE_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# Data-driven confidence thresholds for FloraSense (9,985 classes; uniform = 0.0001):
# Measured on iNaturalist (45), PlantDoc (236), and PlantVillage (76):
#   - 90th percentile of incorrect predictions across all three sets is <= 0.0039
#   - Median of incorrect predictions is 0.0009 - 0.0012
#   - Median of correct iNaturalist predictions is 0.0130 (mean 0.0211)
FLORASENSE_HIGH_CONF_THRESHOLD = float(os.getenv("FLORASENSE_HIGH_CONF", "0.0250"))
FLORASENSE_MEDIUM_CONF_THRESHOLD = float(os.getenv("FLORASENSE_MEDIUM_CONF", "0.0040"))
FLORASENSE_MIN_TOP1_TOP2_RATIO = float(os.getenv("FLORASENSE_MIN_MARGIN_RATIO", "1.60"))

# Data-driven confidence thresholds for ONNX ConvNeXt-Tiny plant model (68 species + not_plant):
ONNX_PLANT_HIGH_CONF_THRESHOLD = float(os.getenv("ONNX_PLANT_HIGH_CONF", "0.65"))
ONNX_PLANT_MEDIUM_CONF_THRESHOLD = float(os.getenv("ONNX_PLANT_MEDIUM_CONF", "0.35"))
ONNX_NOT_PLANT_THRESHOLD = float(os.getenv("ONNX_NOT_PLANT_THRESHOLD", "0.50"))

LOW_CONFIDENCE_PLANT_MESSAGE = (
    "Plant could not be identified reliably. Please upload a clearer image."
)

# Representative supported plant groups & species exposed in UI
SUPPORTED_PLANT_SUMMARY = {
    "florasense": {
        "model_name": "Sisigoks/FloraSense (ViT-Base, 9,985 fine-grained botanical taxa)",
        "total_raw_classes": 9985,
        "unique_binomials": 9481,
        "reported_eval_accuracy": 0.3546,
        "key_supported_genera": [
            "Adiantum (Maidenhair ferns)",
            "Aloe (Aloe vera & relatives)",
            "Azadirachta (Neem)",
            "Capsicum (Bell & Chilli peppers)",
            "Citrus (Orange, Lemon, Lime)",
            "Codiaeum (Croton)",
            "Cucurbita (Squash, Pumpkin)",
            "Epipremnum (Money plant / Pothos)",
            "Ficus (Peepal, Banyan, Rubber plant)",
            "Fragaria (Strawberry)",
            "Ginkgo (Ginkgo biloba)",
            "Glycine (Soybean)",
            "Hibiscus (Hibiscus rosa-sinensis & relatives)",
            "Malus (Apple)",
            "Mangifera (Mango)",
            "Murraya / Bergera (Curry leaf)",
            "Ocimum (Tulsi / Basil)",
            "Prunus (Cherry, Peach)",
            "Rosa (Rose)",
            "Rubus (Raspberry)",
            "Sansevieria / Dracaena (Snake plant)",
            "Solanum (Tomato, Potato, Eggplant)",
            "Vaccinium (Blueberry)",
            "Vitis (Grape)",
            "Zea (Corn / Maize)",
        ],
    },
    "onnx_convnext": {
        "model_name": "ConvNeXt-Tiny Field Plant Identifier (68 regional & crop species + non-plant rejection)",
        "total_raw_classes": 69,
        "unique_binomials": 68,
        "reported_eval_accuracy": 0.9053,
    },
}


# ==============================================================================
# 2. DISEASE DETECTION CONFIGURATION
# ==============================================================================
DISEASE_BACKEND = os.getenv("PLANTVERSE_DISEASE_BACKEND", "mobilenet_v2").strip().lower()
DISEASE_HF_MODEL_NAME = os.getenv(
    "PLANT_DISEASE_MODEL_NAME",
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
).strip()
ONNX_DISEASE_MODEL_PATH = Path(
    os.getenv("PLANTVERSE_ONNX_DISEASE_MODEL", str(MODELS_DIR / "model.onnx"))
)
ONNX_DISEASE_LABELS_PATH = Path(
    os.getenv("PLANTVERSE_ONNX_DISEASE_LABELS", str(MODELS_DIR / "labels.json"))
)
EXTERNAL_API_URL = os.getenv("PLANT_DISEASE_API_URL", "").strip()
LOCAL_KERAS_PATH = os.getenv("PLANT_DISEASE_KERAS_MODEL_PATH", "trained_plant_disease_model.keras").strip()

# Image validation limits
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_IMAGE_DIMENSION_PX = 32

# MobileNetV2 preprocessing (verified against linkanjarad preprocessor_config.json):
# - RGB conversion + EXIF orientation transpose
# - Resize shortest edge to 256 (BILINEAR) -> Center crop (224, 224)
# - Rescale 1/255.0 -> Normalize mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
DISEASE_INPUT_SIZE: Tuple[int, int] = (224, 224)
DISEASE_RESIZE_SHORTEST_EDGE: int = 256
DISEASE_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
DISEASE_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# Data-driven confidence thresholds for Disease Detection:
# Measured on PlantVillage (76) and PlantDoc (236):
#   - On PlantVillage: correct min=0.7025, p25=0.9923, median=0.9981
#   - On PlantDoc: correct median=0.7849 (mean=0.7226), whereas incorrect median=0.5500 (p25=0.3571)
#   - High confidence >= 0.75 (above incorrect median & near correct median on field photos)
#   - Medium confidence >= 0.55 (warn user that diagnosis is tentative)
#   - Low confidence < 0.55 (reject as unreliable)
DISEASE_HIGH_CONF_THRESHOLD = float(os.getenv("DISEASE_HIGH_CONF_THRESHOLD", "0.75"))
DISEASE_MEDIUM_CONF_THRESHOLD = float(os.getenv("DISEASE_CONFIDENCE_THRESHOLD", "0.55"))
DISEASE_MIN_TOP1_TOP2_MARGIN = float(os.getenv("DISEASE_MIN_MARGIN", "0.10"))

LOW_CONFIDENCE_DISEASE_MESSAGE = (
    "Disease could not be diagnosed reliably from this image (low model confidence). "
    "Please upload a clearer, close-up photograph of a single affected leaf."
)
UNSUPPORTED_CROP_DISEASE_MESSAGE = (
    "Disease classification is not available for this plant."
)

# Exact 38-class ordered list matching training alphabetical folder order (indices 0..37)
PLANTVILLAGE_38_RAW_CLASSES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Supported 14 crops in the 38-class PlantVillage disease model and their botanical/common aliases
SUPPORTED_DISEASE_CROPS: Dict[str, Set[str]] = {
    "Apple": {"apple", "malus", "malus domestica", "malus pumila", "malus sylvestris", "malus baccata", "malus prunifolia", "crabapple"},
    "Blueberry": {"blueberry", "vaccinium", "vaccinium corymbosum", "vaccinium angustifolium", "vaccinium myrtillus"},
    "Cherry": {"cherry", "cherry (including sour)", "prunus avium", "prunus cerasus", "prunus serrulata", "sweet cherry", "sour cherry"},
    "Corn (Maize)": {"corn", "corn (maize)", "maize", "zea", "zea mays"},
    "Grape": {"grape", "grapevine", "vitis", "vitis vinifera", "vitis labrusca", "vitis riparia"},
    "Orange / Citrus": {"orange", "orange / citrus", "orange (fruit)", "citrus", "citrus sinensis", "citrus reticulata", "citrus limon", "citrus aurantium", "lemon", "lime", "mandarin"},
    "Peach": {"peach", "prunus persica", "nectarine"},
    "Bell Pepper": {"bell pepper", "pepper, bell", "pepper", "capsicum", "capsicum annuum", "capsicum frutescens", "capsicum chinense", "capsicum / chilli", "chilli", "sweet pepper"},
    "Potato": {"potato", "solanum tuberosum"},
    "Raspberry": {"raspberry", "rubus idaeus", "rubus occidentalis", "rubus strigosus"},
    "Soybean": {"soybean", "soya bean", "soyabean", "glycine max", "glycine soja"},
    "Squash": {"squash", "squash / pumpkin", "pumpkin", "zucchini", "cucurbita", "cucurbita pepo", "cucurbita maxima", "cucurbita moschata"},
    "Strawberry": {"strawberry", "fragaria", "fragaria ananassa", "fragaria x ananassa", "fragaria vesca"},
    "Tomato": {"tomato", "solanum lycopersicum", "lycopersicon esculentum", "solanum pimpinellifolium"},
}


def resolve_supported_disease_crop(plant_name: str) -> str | None:
    """
    Given a plant name (common name or Latin binomial), return the canonical
    supported crop name from SUPPORTED_DISEASE_CROPS if and only if it belongs
    to one of the 14 supported crops. Otherwise return None.
    """
    if not plant_name:
        return None
    clean = plant_name.strip().lower().replace("_", " ")
    if clean in ("unknown", "unknown plant", "unidentified plant", "not_plant", "no plant"):
        return None

    for canonical_crop, aliases in SUPPORTED_DISEASE_CROPS.items():
        if clean == canonical_crop.lower() or clean in aliases:
            return canonical_crop
        for alias in aliases:
            # Match exact token prefix (e.g. "malus domestica borkh" -> "malus domestica")
            if len(alias) >= 4 and (clean.startswith(alias + " ") or alias.startswith(clean + " ")):
                return canonical_crop
    return None


# ==============================================================================
# 3. CLASSICAL CV LESION SEGMENTATION, SEVERITY & HEALTH SCORE CONFIG (PHASE 6)
# ==============================================================================
# Project-defined severity categories mapped from affected leaf area %
SEVERITY_THRESHOLDS_PCT = {
    "early_max": 5.0,       # > 0% and < 5% affected leaf area -> Early
    "moderate_max": 20.0,   # >= 5% and < 20% affected leaf area -> Moderate
                            # >= 20% affected leaf area -> Severe
}

# Plants with naturally variegated or non-green foliage where HSV color-threshold
# lesion segmentation is disabled to prevent false-positive lesion masks
VARIEGATED_OR_COLORED_FOLIAGE_PLANTS: Set[str] = {
    "croton",
    "codiaeum",
    "codiaeum variegatum",
    "coleus",
    "coleus scutellarioides",
    "bougainvillea",
    "money plant",
    "money_plant",
    "epipremnum",
    "epipremnum aureum",
    "snake plant",
    "snake_plant",
    "sansevieria",
    "dracaena trifasciata",
    "spider plant",
    "spider_plant",
    "chlorophytum comosum",
    "aloe",
    "aloe vera",
    "jade",
    "crassula ovata",
}
