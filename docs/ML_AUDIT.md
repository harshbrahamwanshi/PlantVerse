# Phase 0: Full ML Pipeline Audit (`docs/ML_AUDIT.md`)

**Date:** 2026-10-25  
**Branch:** `fix/ml-pipeline`  
**Project:** PlantVerse (`https://github.com/harshbrahamwanshi/PlantVerse.git`)  
**Reference Studied:** FloraCare (`https://github.com/harshbrahamwanshi/FloraCare.git` — Read-Only Reference)

---

## 1. End-to-End Pipeline Trace in PlantVerse

### A. Plant Identification Flow (`app_pages/identify.py` $\rightarrow$ `plant_identification.py`)

```
User Upload (app_pages/identify.py:76-81)
  │  st.file_uploader(type=["jpg", "jpeg", "png"])
  ▼
Image Opening (app_pages/identify.py:161)
  │  image = Image.open(img_file)  [NO EXIF orientation correction]
  ▼
Species Prediction (plant_identification.py:132-170, predict_species(image))
  ├─ RGB Conversion (lines 139-140): image.convert("RGB")
  ├─ Primary Model Inference (lines 145-152):
  │    classifier = load_classifier()  -> pipeline("image-classification", model="Sisigoks/FloraSense")
  │    preds = classifier(image)       -> default top_k=5
  │    raw_label = preds[0]["label"], score = float(preds[0]["score"])
  ├─ Label Cleaning (lines 105-117, 154):
  │    _clean_florasense_label(raw_label) -> strips botanical authority, returns "Genus species"
  ├─ SILENT CROSS-CHECK OVERRIDE (lines 158-169):
  │    if score < 0.15:
  │        dis_res = predict_disease(image)  # Calls 38-class PlantVillage MobileNetV2!
  │        if crop_conf >= 0.55:
  │            return _CROP_BOTANICAL_MAP.get(crop_name, crop_name), crop_conf
  ▼
Knowledge Graph Resolution (app_pages/identify.py:175-182)
  ├─ get_wikipedia_title(label) (plant_identification.py:183-250)
  ├─ get_taxonomy_from_wikidata(wiki_title) (plant_identification.py:278-474)
  ├─ get_medicinal_uses_from_wikipedia(wiki_title) (plant_identification.py:480-598)
  └─ PlantLocationFinder().get_wikipedia_summary(wiki_title) (ar_location_plant_map.py:83-107)
  ▼
Unconditional Disease Screening (app_pages/identify.py:188-209)
  │  if enable_disease_check (Default=True, line 99):
  │    disease_res = predict_disease(image)  [NO check if identified plant is a supported crop!]
  │    Displays disease badge and pathology tab on ANY plant (e.g., ferns, trees, houseplants).
```

### B. Disease Detection Flow (`app_pages/disease_detection.py` $\rightarrow$ `plant_disease_detection.py` / `ml_service/`)

```
User Upload or Sample Button (app_pages/disease_detection.py:60-100)
  ▼
Preprocessing (plant_disease_detection.py:77-145, preprocess_image(image_input))
  ├─ Validates size <= 10 MB, non-empty, Image.open().verify()
  ├─ Reopens image, converts to RGB (lines 133-137) [NO EXIF orientation correction]
  ├─ Re-encodes to JPEG quality=88 for session storage (lines 173-175)
  ▼
Model Dispatch (plant_disease_detection.py:177-242, predict_disease())
  ├─ Path 1: External Microservice (lines 178-189)
  │    Only if EXTERNAL_API_URL env var is set (default "" -> not used unless configured).
  │    Calls POST /api/disease/predict on ml_service/app.py.
  │    If request fails, catches exception and falls back to in-process inference.
  ├─ Path 2: Local Keras Model (lines 48-55, 213-236)
  │    Checks if "trained_plant_disease_model.keras" exists on disk.
  │    (File does NOT exist in repo, and tensorflow is NOT installed in venv).
  ├─ Path 3: Active Runtime Model — Hugging Face Transformers (lines 58-71, 198-211)
  │    Model: "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
  │    predictions = pipe(image, top_k=5)
  ├─ Path 4: SILENT FAKE FALLBACK (lines 72-74, 237-242; also ml_service/services/disease_predictor.py:71-72, 117-120)
  │    If model fails to load, returns {"type": "fallback"} and predicts:
  │    top_label = "Healthy Plant", top_score = 0.65 (or 0.70 in ml_service)!
  ▼
Post-Processing & Lookup (plant_disease_detection.py:244-282)
  ├─ info = get_disease_info(top_label) (disease_recommendations.py:788-818)
  ├─ is_uncertain = top_score < threshold (0.50)
  │    NOTE: Even when is_uncertain is True, the UI still displays the predicted crop,
  │    disease, pathogen, and treatment as a valid diagnosis with only a warning banner!
  ├─ Region / Lesion Segmentation: NOT IMPLEMENTED in PlantVerse.
  ├─ Dynamic Severity / Health Score: NOT IMPLEMENTED (severity is a static string from DISEASE_DATABASE).
  └─ Treatment: Static lookup in DISEASE_DATABASE (disease_recommendations.py:13-721).
```

---

## 2. Detailed Model Audit (With File Paths & Line Numbers)

### Model 1: Plant Identification Model

| Audit Dimension | Finding & Evidence |
| :--- | :--- |
| **Runtime Loader & Path** | Loaded in [`plant_identification.py`](file:///home/lumora/Desktop/PlantVerse/plant_identification.py#L93-L102) via `pipeline("image-classification", model="Sisigoks/FloraSense")`. Cached weights on disk: `/home/lumora/.cache/huggingface/hub/models--Sisigoks--FloraSense`. |
| **Architecture** | `ViTForImageClassification` (`google/vit-base-patch16-224` fine-tuned, hidden size 1024, patch size 16, image size $224 \times 224$). |
| **Silent Fallback / Model Reuse Bug** | **CRITICAL BUG** at [`plant_identification.py:158-169`](file:///home/lumora/Desktop/PlantVerse/plant_identification.py#L158-L169): Whenever FloraSense top-1 softmax score is `< 0.15`, `predict_species()` calls `from plant_disease_detection import predict_disease`! If the 38-class crop disease model has confidence $\ge 0.55$, `predict_species()` **overwrites the plant identification with the crop disease model's output**! |
| **Number of Classes & Class List** | **9,985 classes** defined in `config.json` (`id2label` indices `"0"` to `"9984"`). Format: Latin binomial + botanical authority with underscores (e.g., `0: "Abelia_chinensis_R_Br_"`, `183: "Adiantum_pedatum_L_"`). |
| **Duplicate / Noisy Classes in Training** | The 9,985 class list contains duplicate species split across slightly different punctuation of botanical authorities! Examples found in `config.json`:<br>• `185: "Adiantum_raddianum_C_Presl"` vs `186: "Adiantum_raddianum_C__Presl"`<br>• `1: "Abelia_grandiflora__Ravelli_ex_Andr___Rehder"` vs `4: "Abelia_x_grandiflora__Rovelli_ex_Andr___Rehder"`<br>• `11: "Abies_cephalonica_J_W_Loudon"` vs `12: "Abies_cephalonica_Loudon"`<br>• `13` & `14`: `Abies_concolor` duplicates.<br>• `Codiaeum_spp_`, `Codiaeum_variegatum__L___A_Juss_`, `Codiaeum_variegatum__L___A__Juss_` (3 classes for Croton!).<br>This splits probability mass across identical species! |
| **Inference Class Mapping** | HF `pipeline` maps index $\rightarrow$ `model.config.id2label[idx]`. Then [`_clean_florasense_label()`](file:///home/lumora/Desktop/PlantVerse/plant_identification.py#L105-L117) strips underscores and keeps the first two tokens (`Genus species`). However, it does **not** sum the probabilities of synonymous/duplicate classes before taking top-1! |
| **Preprocessing (Inference vs. Training)** | `ViTImageProcessor` in `Sisigoks/FloraSense`:<br>• `do_resize=True`, `size={"height": 224, "width": 224}`, `default_to_square=True` (squashes aspect ratio to $224 \times 224$, `resample=2` BILINEAR).<br>• `do_center_crop=None` (no center crop).<br>• `do_rescale=True` ($1/255$, `0.00392156862745098`).<br>• `do_normalize=True`, `image_mean=[0.5, 0.5, 0.5]`, `image_std=[0.5, 0.5, 0.5]`.<br>• **Missing EXIF orientation:** [`app_pages/identify.py:161`](file:///home/lumora/Desktop/PlantVerse/app_pages/identify.py#L161) opens phone camera images with `Image.open(img_file)` without `ImageOps.exif_transpose()`, so rotated phone photos are fed sideways/upside-down! |
| **Output Handling & Softmax** | HF `ImageClassificationPipeline` runs `model.eval()` and `torch.no_grad()`, computes `outputs.logits.softmax(-1)`, and returns top-5 dicts. **Why scores are tiny (`< 0.15`):** Because softmax is computed over $N = 9,985$ classes! In a 9,985-way softmax, even a strong logit of `5.25` (when uniform is $0.0001$) yields a probability of only `0.0166` ($1.66\%$) because $e^{5.25} = 190.5$ is divided by $\sum_{j=1}^{9985} e^{z_j} \approx 11,400$. |
| **Training Dataset & Fundamental Capability** | According to the official model card (`Sisigoks/FloraSense`), it was trained on `Sisigoks/Planter_GARDEN_EDITION` for 10 epochs and achieved an **Evaluation Loss of 4.2894** and **Evaluation Accuracy of only 35.46%**. Furthermore, on many common plants (e.g. `static/ginkgo.jpg` -> predicts `Abies alba` with $0.09\%$ confidence; `peepal.jpg` -> predicts `Populus tremuloides`; `money_plant.jpg` -> `Piper hispidum`), `Sisigoks/FloraSense` fails completely. |

---

### Model 2: Plant Disease Detection Model

| Audit Dimension | Finding & Evidence |
| :--- | :--- |
| **Runtime Loader & Path** | Loaded in [`plant_disease_detection.py:42-75`](file:///home/lumora/Desktop/PlantVerse/plant_disease_detection.py#L42-L75) and [`ml_service/services/disease_predictor.py:34-73`](file:///home/lumora/Desktop/PlantVerse/ml_service/services/disease_predictor.py#L34-L73).<br>1. Checks `LOCAL_KERAS_PATH` (`trained_plant_disease_model.keras`) — **file does not exist**.<br>2. Loads HF model `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` from `/home/lumora/.cache/huggingface/hub/models--linkanjarad--mobilenet_v2_1.0_224-plant-disease-identification`. |
| **Silent Fake Fallback Bug** | **CRITICAL VIOLATION OF GROUND RULE 3:** In [`plant_disease_detection.py:72-74, 237-242`](file:///home/lumora/Desktop/PlantVerse/plant_disease_detection.py#L237-L242) and [`ml_service/services/disease_predictor.py:71-72, 117-120`](file:///home/lumora/Desktop/PlantVerse/ml_service/services/disease_predictor.py#L117-L120), if the model fails to load, it silently returns a fake prediction: `top_label = "Healthy Plant"`, `top_score = 0.65` (or `0.70`) with `success: True`! |
| **Architecture** | `MobileNetV2ForImageClassification` (`google/mobilenet_v2_1.0_224` fine-tuned on 38 classes). |
| **Number of Classes & Mapping** | **38 classes** (14 crops: Apple, Blueberry, Cherry, Corn/Maize, Grape, Orange/Citrus, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato).<br>The HF model's `config.json` `id2label` uses human-readable names (`0: "Apple Scab"`, `1: "Apple with Black Rot"`, ..., `37: "Healthy Tomato Plant"`), which match the 38 keys of `DISEASE_DATABASE` and the alphabetical order of `RAW_LABEL_MAP` in [`disease_recommendations.py:725-764`](file:///home/lumora/Desktop/PlantVerse/disease_recommendations.py#L725-L764). |
| **Keras Code Path Bug** | In [`plant_disease_detection.py:216-219`](file:///home/lumora/Desktop/PlantVerse/plant_disease_detection.py#L216-L219), the dead Keras path resizes to `(128, 128)` with raw `[0, 255]` float32 pixels without normalization and assumes `preds` are probabilities. |
| **Preprocessing (Inference vs. Training)** | `MobileNetV2FeatureExtractor` in `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`:<br>• Resize shortest edge to `256` (`resample=2` BILINEAR)<br>• Center crop `224 x 224`<br>• Rescale $1/255$, Normalize `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`.<br>• **Missing EXIF orientation:** `preprocess_image()` in [`plant_disease_detection.py:129-139`](file:///home/lumora/Desktop/PlantVerse/plant_disease_detection.py#L129-L139) does not apply `ImageOps.exif_transpose()`.<br>• **No Leaf Isolation / Cropping:** Feeds the entire uncropped frame into center-crop, which on real field photos captures background soil/sky/fingers instead of the leaf. |
| **Training Data & Domain Mismatch** | Trained **exclusively on the Kaggle PlantVillage dataset** (single detached leaf on a uniform studio grey/black background).<br>• Reported accuracy on PlantVillage studio split: **95.41%**.<br>• **Severe Overconfidence & Domain Collapse on Non-PlantVillage Images:** Because it was trained in a closed-world 38-class studio regime with no out-of-distribution/background class and no field data (like PlantDoc), it produces **80–95% softmax confidence on completely unrelated plants**:<br>  - `static/fern-plate.jpg` (Fern) $\rightarrow$ **Healthy Raspberry Plant (86.89%)**<br>  - `peepal.jpg` (Sacred Fig) $\rightarrow$ **Grape Black Rot (94.60%)**<br>  - `money_plant.jpg` (Pothos) $\rightarrow$ **Grape Black Rot (80.70%)**<br>  - `curry_leaf.jpg` (Curry Leaf) $\rightarrow$ **Grape Black Rot (74.73%)**<br>  - `hibiscus.jpg` (Hibiscus) $\rightarrow$ **Tomato Late Blight (67.65%)**<br>  - `tulsi.jpg` (Holy Basil) $\rightarrow$ **Tomato Late Blight (62.60%)**<br>  - `static/sample_leaves/healthy_apple.jpg` (Real field photo of healthy apple leaf on tree) $\rightarrow$ **Healthy Blueberry Plant (48.23%)** |

---

## 3. Study of Reference Repository (`FloraCare`)

We audited `/tmp/FloraCare_ref` (`https://github.com/harshbrahamwanshi/FloraCare.git`) to understand its architecture, models, preprocessing, computer vision pipeline, and licensing.

1. **Model Serving Architecture (`backend/app/plant_id.py`, `classifier.py`, `diagnosis.py`):**
   - Uses **ONNX Runtime (`onnxruntime`)** with dynamic int8 quantized models (`plant_model.onnx` — 34 MB, and `model.onnx` — 86 MB), running in ~165–190 ms on CPU with ~5x lower RAM than PyTorch.
   - **Two-Stage Dual-Model Pipeline:**
     - **Model 1 (`plant_model.onnx`):** `ConvNeXt-Tiny` (`convnext_tiny.fb_in22k_ft_in1k`) with **two classification heads** trained on 24,630 train / 4,802 val / 4,808 test images from iNaturalist field photos, PlantVillage, PlantDoc, and Imagenette:
       - **Head 1 (`species_logits`):** **68 plant species** (55 common Indian/garden plants like Neem, Tulsi, Peepal, Banyan, Money Plant, Rose, Hibiscus, Aloe, Ginkgo/ornamentals + 13 agricultural crops) + **`not_plant` class** (69 total outputs). Achieves **90.5% top-1 / 98.2% top-5 accuracy** on held-out test set and **100% on `not_plant`**.
       - **Head 2 (`condition_logits`):** **7 general foliar symptom groups** (`healthy`, `leaf_spot_blight`, `powdery_mildew`, `rust`, `sooty_mould`, `mosaic_yellowing`, `pest_infestation`). Achieves **96.6% accuracy**.
     - **Model 2 (`model.onnx`):** Vision Transformer (`kimcomehome/plantvillage-vit-leaf-disease`, `ViT-Base`) **fine-tuned on PlantVillage + PlantDoc real field photos** (`training/train_field_model.ipynb`). Improves real-world field accuracy on PlantDoc from **37.7% $\rightarrow$ 76.7%** while keeping PlantVillage accuracy at **99.9%**!
2. **Preprocessing & Two-View Inference (`backend/app/plant_id.py:62-78`):**
   - `plant_id.py` runs a 2-image batch `[full_rgb, leaf_rgb]` resized to $224 \times 224$ (bilinear), scaled to $[0, 1]$, normalized with ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`, NCHW. Species probabilities are averaged across the full image and leaf crop; condition probabilities are taken from the leaf crop.
   - `classifier.py` (ViT disease model) resizes the leaf crop to $224 \times 224$ (bilinear), scales to $[-1, 1]$ (`x / 127.5 - 1.0`, equivalent to `mean=0.5, std=0.5`), NCHW.
3. **Crop-Consistent Disease Gating (`backend/app/diagnosis.py:114-135`):**
   - Checks if the identified plant has a corresponding PlantVillage crop (`pv_crop`).
   - Only crops with $P(\text{crop} \mid \text{plant model}) \ge 0.05$ are allowed to compete in the 38-class disease classifier: `fused[idx] = probs[idx] * p_crop`.
   - Impossible cross-crop combinations (e.g., Hibiscus $\rightarrow$ Tomato Late Blight, or Tomato $\rightarrow$ Apple Cedar Rust) are **mathematically prevented**.
4. **Classical CV Leaf & Lesion Segmentation (`backend/app/vision.py`):**
   - Uses OpenCV (`cv2`) HSV + Excess-Green ($2G - R - B > 25$) color segmentation to locate the dominant leaf core.
   - Builds a LAB color-space background model from the 6-pixel image border so brown tables/soil are not merged into the leaf.
   - Erodes the leaf mask slightly to avoid edge artifacts, then segments necrotic/chlorotic/powdery lesion contours (`tone | dark | white & ~green`), computing `lesion_area / leaf_area * 100` as affected percentage.
   - Disables color-based lesion severity on naturally variegated/colored plants (`croton`, `coleus`, `bougainvillea`, `money_plant`, `snake_plant`, `aloe`, `spider_plant`, `jade`).
5. **Severity & Health Score (`backend/app/diagnosis.py:24-39`):**
   - Maps affected area % to `none` (healthy), `early` ($< 5\%$), `moderate` ($5\text{–}20\%$), `severe` ($\ge 20\%$).
   - Computes a derived 0–100 health score using urgency penalty and affected area %.
6. **Licensing & Compatibility (`README.md:215-220`):**
   - Both repos (`PlantVerse` and `FloraCare`) are authored by the same user (`harshbrahamwanshi`).
   - `plant_model.onnx` is based on `timm` `convnext_tiny.fb_in22k_ft_in1k` (**Apache-2.0 license**), trained on CC0/CC-BY/CC-BY-SA iNaturalist, PlantVillage, PlantDoc, and Imagenette data.
   - `model.onnx` is fine-tuned from `kimcomehome/plantvillage-vit-leaf-disease` (**CC BY-SA 3.0**) on PlantVillage + PlantDoc.
   - Classical CV algorithms (`vision.py`) and JSON knowledge structures are 100% open-source and compatible with PlantVerse.

---

## 4. Ranked List of Root Causes & Evidence

### Rank 1 (CRITICAL): `predict_species()` Silently Hijacks Plant Identification with the 38-Class Crop Disease Model
- **Location:** [`plant_identification.py:158-169`](file:///home/lumora/Desktop/PlantVerse/plant_identification.py#L158-L169)
- **Mechanism:** Whenever `Sisigoks/FloraSense` outputs a top-1 softmax probability `< 0.15` (which happens on **almost every image** because softmax is divided across 9,985 classes!), `predict_species()` calls `predict_disease(image)`. Because the 38-class MobileNetV2 disease model is wildly overconfident on out-of-distribution images ($\ge 0.55$), `predict_species()` discards the plant model's prediction and returns one of the 14 agricultural crops!
- **Real Terminal Evidence:**
  - `hibiscus.jpg`: FloraSense top-1 is `Hibiscus_rosa_sinensis_L_` (`score=0.0166 < 0.15`). `predict_species()` discards `Hibiscus rosa-sinensis` and returns **`Tomato` (`0.6765`)**!
  - `tulsi.jpg`: FloraSense top-1 is `Ocimum_americanum_L_` (`score=0.0139 < 0.15`). `predict_species()` discards `Ocimum` and returns **`Tomato` (`0.6260`)**!
  - `static/fern-plate.jpg`: FloraSense top-1 is `Phegopteris_connectilis` (a fern, `score=0.0198 < 0.15`). `predict_species()` discards the fern and returns **`Raspberry` (`0.8689`)**!
  - `peepal.jpg`: `predict_species()` returns **`Grape` (`0.9460`)**!
  - `money_plant.jpg`: `predict_species()` returns **`Grape` (`0.8070`)**!
  - `curry_leaf.jpg`: `predict_species()` returns **`Grape` (`0.7473`)**!
  - `neem.jpg`: `predict_species()` returns **`Cucurbita` (`0.6068`)**!

### Rank 2 (CRITICAL): Fundamental Inadequacy & 9,985-Way Softmax Dilution of `Sisigoks/FloraSense`
- **Location:** [`plant_identification.py:94-102`](file:///home/lumora/Desktop/PlantVerse/plant_identification.py#L94-L102)
- **Mechanism:**
  1. `Sisigoks/FloraSense` has **9,985 classes** with an official evaluation accuracy of only **35.46%** (eval loss `4.2894`).
  2. Its label space has duplicate species entries with minor author-citation punctuation differences (e.g., `Adiantum_raddianum_C_Presl` vs `Adiantum_raddianum_C__Presl`, 3 duplicate `Codiaeum` classes), diluting softmax probabilities.
  3. Even when `Sisigoks/FloraSense` is not overridden by `predict_disease()`, it misidentifies common benchmark images right inside PlantVerse's own repo:
     - `static/ginkgo.jpg` (PlantVerse's own Ginkgo showcase image!) $\rightarrow$ predicts `Abies alba` (Silver Fir) with `0.0009` ($0.09\%$) confidence.
     - `static/sample_leaves/healthy_apple.jpg` (PlantVerse's own sample image!) $\rightarrow$ predicts `Salix lasiolepis` (Willow) with `0.0014` ($0.14\%$) confidence.
     - `static/herbarium.jpg` $\rightarrow$ predicts `Veronica teucrium` with `0.0006` ($0.06\%$) confidence.

### Rank 3 (CRITICAL): Zero Plant $\rightarrow$ Disease Consistency Gating
- **Location:** [`app_pages/identify.py:189-208`](file:///home/lumora/Desktop/PlantVerse/app_pages/identify.py#L189-L208) and [`plant_disease_detection.py:147-282`](file:///home/lumora/Desktop/PlantVerse/plant_disease_detection.py#L147-L282)
- **Mechanism:**
  1. In `app_pages/identify.py`, `predict_disease(image)` is run unconditionally on every uploaded plant image (even when the plant is a fern, Ginkgo, Snake Plant, or Hibiscus) and its result is displayed directly on the hero card (`app_pages/identify.py:260-265`) and pathology tab without checking if the identified species is even one of the 14 supported crops!
  2. In `plant_disease_detection.py`, the 38-class classifier is never filtered or validated against the host plant species. Thus:
     - `static/hero.jpg` identified as *Adiantum pedatum* (Northern Maidenhair Fern) is simultaneously diagnosed on the same card with **Healthy Raspberry Plant (28.4%)**!
     - `static/ginkgo.jpg` (Ginkgo biloba) is diagnosed with **Squash Powdery Mildew (46.3%)**!
     - `Potato_Late_blight_5305.jpg` (Potato leaf) is diagnosed as **Tomato Late Blight (99.4%)**!

### Rank 4 (HIGH): Domain Mismatch of Disease Model (`linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`) on Real-World Photos
- **Location:** [`plant_disease_detection.py:31-34`](file:///home/lumora/Desktop/PlantVerse/plant_disease_detection.py#L31-L34)
- **Mechanism:** `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` was trained strictly on studio PlantVillage images (single leaf on flat grey/black backdrop) without field photos (PlantDoc) and without leaf segmentation/cropping before center-crop.
- **Evidence:**
  - On PlantVerse's own `static/sample_leaves/healthy_apple.jpg` (a real outdoor photo of a healthy apple leaf with background greenery), `linkanjarad` fails and predicts **`Healthy Blueberry Plant` (`48.23%`)**!
  - On `corn_rust.jpg` (real field photo of maize common rust), `linkanjarad` misclassifies it as **`Corn (Maize) with Northern Leaf Blight` (`53.30%`)**.
  - On `late_blight.jpg` (real field photo of tomato late blight), `linkanjarad` misclassifies it as **`Tomato with Early Blight` (`98.40%`)**.
  - By contrast, FloraCare's `model.onnx` (ViT fine-tuned on PlantVillage + PlantDoc field photos) combined with leaf bounding-box cropping improves real field photo accuracy from **37.7% to 76.7%** while achieving **99.9%** on PlantVillage.

### Rank 5 (HIGH): Silent Fake Predictions on Model Load Failure & Unblocked Low-Confidence Predictions
- **Location:** [`plant_disease_detection.py:72-74, 237-242, 246-282`](file:///home/lumora/Desktop/PlantVerse/plant_disease_detection.py#L237-L242) & [`ml_service/services/disease_predictor.py:71-72, 117-120`](file:///home/lumora/Desktop/PlantVerse/ml_service/services/disease_predictor.py#L117-L120) & [`app_pages/identify.py:250-269`](file:///home/lumora/Desktop/PlantVerse/app_pages/identify.py#L250-L269)
- **Mechanism:**
  1. **Fake Fallback:** If the disease model fails to load, `predict_disease()` silently returns `top_label = "Healthy Plant"` with `65.0%` (or `70.0%`) confidence and `success = True`.
  2. **No Rejection Policy for Low Confidence:** In `app_pages/identify.py:251-257`, even if confidence is `0.06%` (`score = 0.0006`), the app still displays the species name as a valid result (labeled `"Tentative Candidate"`), queries Wikipedia/Wikidata for that random species, and adds it to the user's Plant Library! Similarly, in `app_pages/disease_detection.py:250-268`, when `is_uncertain` is True (`confidence < 50%`), it still displays the wrong crop, wrong disease, pathogen, and chemical treatment recommendations as the main headline.

### Rank 6 (MEDIUM): Missing EXIF Transpose, Scattered Constants, and Unchecked Non-Plant Inputs
- **Location:** [`plant_identification.py`](file:///home/lumora/Desktop/PlantVerse/plant_identification.py), [`plant_disease_detection.py`](file:///home/lumora/Desktop/PlantVerse/plant_disease_detection.py), [`app_pages/identify.py`](file:///home/lumora/Desktop/PlantVerse/app_pages/identify.py)
- **Mechanism:**
  1. Neither `plant_identification.py` nor `plant_disease_detection.py` calls `ImageOps.exif_transpose(img)`, causing portrait phone camera photos with EXIF orientation tags to be fed rotated $90^\circ$ or $180^\circ$ into the models.
  2. Neither model has an out-of-distribution / `not_plant` detector or leaf presence check, so uploading a photo of a desk, person, or blank wall produces a botanical species and crop disease diagnosis.
  3. Thresholds (`0.15`, `0.55`, `0.50`, `85`, `60`, `75`), model paths, and class mappings are scattered across `plant_identification.py`, `plant_disease_detection.py`, `app_pages/identify.py`, `app_pages/disease_detection.py`, and `ml_service/services/disease_predictor.py` instead of a single central config module.
