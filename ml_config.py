"""
PlantVerse — Centralized ML Pipeline Configuration (`ml_config.py`)
===================================================================
Single source of truth for all model identifiers, file paths, preprocessing
parameters, confidence thresholds (with provisional sample-size annotations),
supported plant & crop class lists, host-to-crop consistency mappings,
classical CV lesion severity thresholds, and local feedback logging.

No scattered constants in page or service modules (Ground Rule 8).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ==============================================================================
# 1. PLANT IDENTIFICATION CONFIGURATION (Primary: ONNX ConvNeXt-Tiny)
# ==============================================================================
PLANT_ID_BACKEND = os.getenv("PLANTVERSE_PLANT_ID_BACKEND", "onnx_convnext").strip().lower()
ONNX_PLANT_MODEL_PATH = Path(
    os.getenv("PLANTVERSE_ONNX_PLANT_MODEL", str(MODELS_DIR / "plant_model.onnx"))
)
ONNX_PLANT_LABELS_PATH = Path(
    os.getenv("PLANTVERSE_ONNX_PLANT_LABELS", str(MODELS_DIR / "plant_labels.json"))
)
PLANTS_METADATA_PATH = Path(
    os.getenv("PLANTVERSE_PLANTS_METADATA", str(DATA_DIR / "plants_metadata.json"))
)
CONDITIONS_METADATA_PATH = Path(
    os.getenv("PLANTVERSE_CONDITIONS_METADATA", str(DATA_DIR / "conditions_metadata.json"))
)
CROP_DISEASES_KB_PATH = Path(
    os.getenv("PLANTVERSE_CROP_DISEASES_KB", str(DATA_DIR / "crop_diseases_kb.json"))
)
INAT_ATTRIBUTION_CSV_PATH = MODELS_DIR / "inat_attribution.csv"

# Secondary optional top-5 suggestions model (never used as headline)
FLORASENSE_MODEL_NAME = os.getenv("PLANTVERSE_FLORASENSE_MODEL", "Sisigoks/FloraSense").strip()
ENABLE_FLORASENSE_SECONDARY = os.getenv("PLANTVERSE_ENABLE_FLORASENSE_SECONDARY", "0").strip() == "1"

# ONNX ConvNeXt-Tiny preprocessing (verified against models/plant_labels.json & training notebook):
# - EXIF orientation transpose + RGB conversion
# - Squash resize to (224, 224) with BILINEAR interpolation
# - Scale to [0.0, 1.0] (/ 255.0)
# - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
ONNX_PLANT_INPUT_SIZE: Tuple[int, int] = (224, 224)
ONNX_PLANT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
ONNX_PLANT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Secondary FloraSense preprocessing:
FLORASENSE_INPUT_SIZE: Tuple[int, int] = (224, 224)
FLORASENSE_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
FLORASENSE_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# Provisional data-driven confidence thresholds for ONNX ConvNeXt-Tiny (68 species + not_plant):
# Derived from tuning set (n=163: 45 iNaturalist + 118 PlantDoc tuning half; 144 correct, 19 incorrect):
#   - Correct predictions (n=144): mean=0.8527, p10=0.6336, p25=0.8066, median=0.9265
#   - Incorrect predictions (n=19): mean=0.5238, median=0.5394, p75=0.6754
#   - High confidence >= 0.68 (above p75=0.6754 of incorrect predictions)
#   - Medium confidence >= 0.35 (warn user to verify morphological traits)
#   - Low confidence < 0.35 or not_plant -> reject identification
ONNX_PLANT_PROVISIONAL_N = 163
ONNX_PLANT_HIGH_CONF_THRESHOLD = float(os.getenv("ONNX_PLANT_HIGH_CONF", "0.68"))
ONNX_PLANT_MEDIUM_CONF_THRESHOLD = float(os.getenv("ONNX_PLANT_MEDIUM_CONF", "0.35"))
ONNX_PLANT_MIN_TOP1_TOP2_MARGIN = float(os.getenv("ONNX_PLANT_MIN_MARGIN", "0.15"))
ONNX_NOT_PLANT_THRESHOLD = float(os.getenv("ONNX_NOT_PLANT_THRESHOLD", "0.45"))

# Secondary FloraSense thresholds (provisional, n=163):
FLORASENSE_HIGH_CONF_THRESHOLD = float(os.getenv("FLORASENSE_HIGH_CONF", "0.0250"))
FLORASENSE_MEDIUM_CONF_THRESHOLD = float(os.getenv("FLORASENSE_MEDIUM_CONF", "0.0040"))
FLORASENSE_MIN_TOP1_TOP2_RATIO = float(os.getenv("FLORASENSE_MIN_MARGIN_RATIO", "1.60"))

LOW_CONFIDENCE_PLANT_MESSAGE = (
    "Plant could not be identified reliably. Please upload a clearer image."
)


def load_onnx_plant_labels() -> Dict[str, Any]:
    """Load the 68 species + not_plant and 7 condition labels from models/plant_labels.json."""
    if not ONNX_PLANT_LABELS_PATH.exists():
        raise RuntimeError(f"Missing plant labels file at '{ONNX_PLANT_LABELS_PATH}'.")
    return json.loads(ONNX_PLANT_LABELS_PATH.read_text(encoding="utf-8"))


def load_plants_metadata() -> Dict[str, Dict[str, Any]]:
    """Load rich metadata (common name, scientific name, Hindi name, care, pv_crop) for all 68 species."""
    if not PLANTS_METADATA_PATH.exists():
        return {}
    raw = json.loads(PLANTS_METADATA_PATH.read_text(encoding="utf-8"))
    return {p["id"]: p for p in raw.get("plants", [])}


def load_conditions_metadata() -> Dict[str, Dict[str, Any]]:
    """Load rich metadata for the 7 general foliar condition classes from data/conditions_metadata.json."""
    if not CONDITIONS_METADATA_PATH.exists():
        return {}
    raw = json.loads(CONDITIONS_METADATA_PATH.read_text(encoding="utf-8"))
    if "conditions" in raw and isinstance(raw["conditions"], list):
        return {c["id"]: c for c in raw["conditions"]}
    return raw




SUPPORTED_PLANT_SUMMARY = {
    "onnx_convnext": {
        "model_name": "ConvNeXt-Tiny Dual-Head Plant Identifier (68 regional & crop species + non-plant rejection)",
        "license": "Apache-2.0 base (timm convnext_tiny) + CC BY-NC-SA 4.0 training corpus (iNaturalist attributions in models/inat_attribution.csv)",
        "total_raw_classes": 69,
        "unique_binomials": 68,
        "provisional_thresholds_n": ONNX_PLANT_PROVISIONAL_N,
    },
    "florasense": {
        "model_name": "Sisigoks/FloraSense (ViT-Base, 9,985 fine-grained botanical taxa — secondary suggestions only)",
        "license": "Apache-2.0 base (google/vit-base-patch16-224); fine-tuned checkpoint license unspecified on HF Hub",
        "total_raw_classes": 9985,
        "unique_binomials": 9481,
    },
}


# ==============================================================================
# 2. DISEASE DETECTION CONFIGURATION (Primary: ViT int8 ONNX `models/model.onnx`)
# ==============================================================================
DISEASE_BACKEND = os.getenv("PLANTVERSE_DISEASE_BACKEND", "onnx_vit").strip().lower()
ONNX_DISEASE_MODEL_PATH = Path(
    os.getenv("PLANTVERSE_ONNX_DISEASE_MODEL", str(MODELS_DIR / "model.onnx"))
)
ONNX_DISEASE_LABELS_PATH = Path(
    os.getenv("PLANTVERSE_ONNX_DISEASE_LABELS", str(MODELS_DIR / "labels.json"))
)
DISEASE_HF_MODEL_NAME = os.getenv(
    "PLANT_DISEASE_MODEL_NAME",
    "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
).strip()
EXTERNAL_API_URL = os.getenv("PLANT_DISEASE_API_URL", "").strip()
LOCAL_KERAS_PATH = os.getenv("PLANT_DISEASE_KERAS_MODEL_PATH", "trained_plant_disease_model.keras").strip()

# Image validation limits
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_IMAGE_DIMENSION_PX = 32

# ViT Disease ONNX preprocessing (verified against classifier.py & ViT config):
# - EXIF orientation transpose + RGB conversion
# - Squash resize to (224, 224) with BILINEAR interpolation
# - Scale and normalize: x / 127.5 - 1.0 (equivalent to rescale 1/255, mean=0.5, std=0.5)
DISEASE_INPUT_SIZE: Tuple[int, int] = (224, 224)
DISEASE_RESIZE_SHORTEST_EDGE: int = 256
DISEASE_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
DISEASE_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# Provisional data-driven confidence thresholds for Disease Detection (derived on PlantDoc tuning half, n=118):
#   - ViT Tuning Correct (n=87): mean=0.8201, p10=0.5512, p25=0.7734, median=0.8932
#   - High confidence >= 0.75 (near p25 of correct predictions) and top1-top2 margin >= 0.10
#   - Medium confidence >= 0.55 (near p10 of correct predictions; show with caution banner)
#   - Low confidence < 0.55 -> reject diagnosis as unreliable
DISEASE_PROVISIONAL_N = 118
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

# Mapping from canonical crop name to the exact subset of indices (0..37) in PLANTVILLAGE_38_RAW_CLASSES
CROP_TO_CLASS_INDICES: Dict[str, List[int]] = {
    "Apple": [0, 1, 2, 3],
    "Blueberry": [4],
    "Cherry": [5, 6],
    "Corn (Maize)": [7, 8, 9, 10],
    "Grape": [11, 12, 13, 14],
    "Orange / Citrus": [15],
    "Peach": [16, 17],
    "Bell Pepper": [18, 19],
    "Potato": [20, 21, 22],
    "Raspberry": [23],
    "Soybean": [24],
    "Squash": [25],
    "Strawberry": [26, 27],
    "Tomato": [28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
}

# Supported 14 crops in the 38-class PlantVillage disease model and their botanical/common/model ID aliases
SUPPORTED_DISEASE_CROPS: Dict[str, Set[str]] = {
    "Apple": {"apple", "malus", "malus domestica", "malus pumila", "malus sylvestris", "malus baccata", "malus prunifolia", "crabapple"},
    "Blueberry": {"blueberry", "vaccinium", "vaccinium corymbosum", "vaccinium angustifolium", "vaccinium myrtillus"},
    "Cherry": {"cherry", "cherry (including sour)", "prunus avium", "prunus cerasus", "prunus avium / cerasus", "prunus serrulata", "sweet cherry", "sour cherry"},
    "Corn (Maize)": {"corn", "corn (maize)", "maize (corn)", "maize", "zea", "zea mays"},
    "Grape": {"grape", "grapevine", "vitis", "vitis vinifera", "vitis labrusca", "vitis riparia"},
    "Orange / Citrus": {"orange", "orange / citrus", "lemon / citrus", "orange (fruit)", "citrus", "citrus sp.", "citrus sinensis", "citrus reticulata", "citrus limon", "citrus aurantium", "lemon", "lime", "mandarin"},
    "Peach": {"peach", "prunus persica", "nectarine"},
    "Bell Pepper": {"bell pepper", "pepper, bell", "pepper", "capsicum", "capsicum / chilli", "capsicum annuum", "capsicum frutescens", "capsicum chinense", "chilli", "sweet pepper"},
    "Potato": {"potato", "solanum tuberosum"},
    "Raspberry": {"raspberry", "rubus idaeus", "rubus occidentalis", "rubus strigosus"},
    "Soybean": {"soybean", "soya bean", "soyabean", "glycine max", "glycine soja"},
    "Squash": {"squash", "squash / pumpkin", "pumpkin", "zucchini", "cucurbita", "cucurbita sp.", "cucurbita pepo", "cucurbita maxima", "cucurbita moschata"},
    "Strawberry": {"strawberry", "fragaria", "fragaria ananassa", "fragaria x ananassa", "fragaria × ananassa", "fragaria vesca"},
    "Tomato": {"tomato", "solanum lycopersicum", "lycopersicon esculentum", "solanum pimpinellifolium"},
}


def resolve_supported_disease_crop(plant_name: Optional[str]) -> Optional[str]:
    """
    Given a plant name (common name, model ID, or Latin binomial), return the canonical
    supported crop name from SUPPORTED_DISEASE_CROPS if and only if it belongs
    to one of the 14 supported crops. Otherwise return None.
    """
    if not plant_name:
        return None
    clean = plant_name.strip().lower().replace("_", " ")
    if clean in ("unknown", "unknown plant", "unidentified plant", "not_plant", "not plant", "no plant"):
        return None

    for canonical_crop, aliases in SUPPORTED_DISEASE_CROPS.items():
        if clean == canonical_crop.lower() or clean in aliases:
            return canonical_crop
        for alias in aliases:
            if len(alias) >= 4 and (
                clean == alias
                or clean.startswith(alias + " ")
                or clean.startswith(alias + " (")
                or alias.startswith(clean + " ")
            ):
                return canonical_crop
    return None


# ==============================================================================
# 3. CLASSICAL CV LESION SEGMENTATION, SEVERITY & HEALTH SCORE CONFIG (TASKS 5-6)
# ==============================================================================
CLASSICAL_CV_METHOD_LABEL = (
    "Color-threshold segmentation (classical CV: HSV + Excess-Green index + border LAB background filter — not a trained segmentation model)"
)

CLASSICAL_CV_LIMITATIONS: List[str] = [
    "Cast shadows or dark specular glare on leaves can be misclassified as necrotic tissue.",
    "Soil, brown bark, or dead background foliage touching the leaf margin can inflate lesion area if chroma matches.",
    "Non-green, purple, or variegated leaves (e.g., Croton, Coleus, Money Plant, Snake Plant) trigger false color masks and are excluded.",
    "Diffuse viral mottling or mild nutrient chlorosis without discrete spots may be underestimated.",
]

SEVERITY_SCHEME_LABEL = (
    "Project-defined severity categories (Early < 5%, Moderate 5–20%, Severe >= 20% affected leaf area; not agricultural standards)"
)

DERIVED_HEALTH_SCORE_LABEL = "Derived metric (not scientifically validated)"

EXPERT_REVIEW_NOTICE = (
    "Agronomic Review Status: Preliminary reference entry — requires verification by a certified plant pathologist or local agricultural extension service before field application."
)

NO_RECOMMENDATION_MESSAGE = "No recommendation available for this disease."

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
    "money plant (pothos)",
    "epipremnum",
    "epipremnum aureum",
    "snake plant",
    "snake_plant",
    "sansevieria",
    "sansevieria trifasciata",
    "dracaena trifasciata",
    "spider plant",
    "spider_plant",
    "chlorophytum comosum",
    "aloe",
    "aloe vera",
    "jade",
    "jade plant",
    "crassula ovata",
}

# ==============================================================================
# 4. USER FEEDBACK LOGGING CONFIG (TASK 11)
# ==============================================================================
USER_FEEDBACK_LOG_PATH = DATA_DIR / "user_feedback_log.jsonl"
