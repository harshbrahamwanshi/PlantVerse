# PlantVerse — ML Pipeline Audit, Root-Cause Fix & Evaluation Report (`docs/ML_FIX_REPORT.md`)

**Branch:** `fix/ml-pipeline`  
**Audit Document:** `docs/ML_AUDIT.md`  
**License Document:** `MODEL_LICENSES.md`  
**Centralized Config:** `ml_config.py`

---

## 1. Executive Summary & Shipped Models

PlantVerse's original ML pipeline suffered from low real-world plant identification accuracy, silent fake fallback predictions when disease models failed to load, an arbitrary `< 0.15` label hijack (`"Adiantum pedatum"`), a broken 34-class vs. 38-class label mapping in the FastAPI/Starlette microservice, lack of cross-stage plant $\rightarrow$ disease consistency gating, and absence of affected-area estimation.

All root causes were audited with line-level evidence (`docs/ML_AUDIT.md`), evaluated against reproducible benchmarks (`tests/ml_eval/run_eval.py`), and fixed on branch `fix/ml-pipeline`.

### Shipped Model Architecture Summary

| Pipeline Stage | Role | Model File / Identifier | Architecture | Input Tensor & Preprocessing | Output Classes & Mapping | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Plant Identifier** | Headline species & general condition classifier behind `predict_species()` | `models/plant_model.onnx` + `models/plant_labels.json` | `ConvNeXt-Tiny` dual-head ONNX (`timm/convnext_tiny.in12k_ft_in1k`) | `pixel_values`: `[batch, 3, 224, 224]` `float32`, EXIF transpose $\rightarrow$ RGB $\rightarrow$ bilinear resize `(224, 224)` $\rightarrow$ `/ 255.0` $\rightarrow$ ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` | `species_logits`: `[batch, 69]` (**68 botanical species + `not_plant`**); `condition_logits`: `[batch, 7]` (**7 foliar conditions**). Mapped via `models/plant_labels.json` + `data/plants_metadata.json`. | Base: **Apache-2.0**; Training data: **CC BY-NC-SA 4.0** (`models/inat_attribution.csv`, 3,907 attributions) |
| **Secondary Plant Suggestions** | Optional `"Other possible matches (low confidence)"` top-5 expander (never headline) | `Sisigoks/FloraSense` (Hugging Face Hub) | `ViTForImageClassification` (`google/vit-base-patch16-224`) | `[1, 3, 224, 224]` `float32`, EXIF transpose $\rightarrow$ RGB $\rightarrow$ `(224, 224)` $\rightarrow$ `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]` | `logits`: `[1, 9985]` (**9,985 fine-grained botanical taxa**, 9,481 unique binomials). | Base: **Apache-2.0**; fine-tuned checkpoint license unspecified |
| **Primary Disease Classifier** | 38-class crop leaf pathology classifier behind `predict_disease()` | `models/model.onnx` + `models/labels.json` | `ViT-Base` int8 quantized ONNX (`wambugu71/crop_leaf_diseases_vit`) | `pixel_values`: `[1, 3, 224, 224]` `float32`, EXIF transpose $\rightarrow$ RGB $\rightarrow$ bilinear resize `(224, 224)` $\rightarrow$ `x / 127.5 - 1.0` (`mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`) | `logits`: `[1, 38]` (**38 PlantVillage/PlantDoc classes across 14 crops**). Verified 38/38 exact match against `PLANTVILLAGE_38_RAW_CLASSES`. | **CC BY-SA 3.0** (ShareAlike + attribution in `MODEL_LICENSES.md` & app UI) |
| **Fallback / Comparison Disease Classifier** | Optional comparison backend (`PLANTVERSE_DISEASE_BACKEND=mobilenet_v2`) | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` | `MobileNetV2ForImageClassification` | `AutoImageProcessor`: resize shortest edge `256` $\rightarrow$ center crop `(224, 224)` $\rightarrow$ ImageNet `mean`/`std` | `logits`: `[1, 38]` (**38 PlantVillage classes**). | Base: **Apache-2.0**; PlantVillage: **CC0 1.0** |
| **Affected-Area & Lesion Analysis** | Classical CV leaf & lesion segmentation (`vision_analysis.py`) | Classical CV (`HSV` + `Excess-Green` + border `CIE LAB`) | Rule-based color-threshold & contour analysis (not a trained deep segmentation model) | EXIF transpose $\rightarrow$ RGB $\rightarrow$ max side `512` $\rightarrow$ Gaussian blur `(5, 5)` $\rightarrow$ `HSV` + `ExG (2G - R - B > 25)` + border `LAB` chroma filter | Leaf polygon, lesion contours, `affected_area_pct` (`0–100%`), project-defined severity (`Healthy / Early / Moderate / Severe`), derived health score (`0–100`), overlay JPEG bytes. | Project code |

---

## 2. Root Causes Found (with File & Line Evidence) & How Each Was Fixed

1. **Silent Fake Disease Fallback (`plant_disease_detection.py` & `ml_service/services/disease_predictor.py`):**
   - **Root Cause:** When local Keras weights (`trained_plant_disease_model.keras`) and Hugging Face loading failed, `plant_disease_detection.py` returned `{"type": "fallback"}` and fabricated `"Healthy Plant"` with hard-coded confidence `0.65` (`plant_disease_detection.py:323-339`), while `ml_service/services/disease_predictor.py:55-61` fabricated `"Healthy Plant"` with confidence `0.70` and `all_scores: {"Healthy Plant": 1.0}`.
   - **Fix:** Completely removed `{"type": "fallback"}` and all hard-coded `"Healthy Plant"` responses in both `plant_disease_detection.py` and `ml_service/services/disease_predictor.py`. Model loading failures, missing files, corrupted weights, or offline external microservices now raise explicit `RuntimeError` exceptions surfaced clearly in the UI.
2. **Hard-Coded `< 0.15` Species Hijack (`plant_identification.py`):**
   - **Root Cause:** `plant_identification.py:139-142` overwrote any prediction with softmax score `< 0.15` to `"Adiantum pedatum"` (Northern Maidenhair Fern), causing 100% of low-confidence predictions (190/236 PlantDoc images and 61/76 PlantVillage images under FloraSense) to be misidentified as `"Adiantum pedatum"`.
   - **Fix:** Removed the `< 0.15` hijack and replaced it with a calibrated confidence policy (`evaluate_onnx_plant_confidence_policy` in `plant_identification.py:274-301`) with `not_plant` rejection, low/medium/high confidence tiers, and a top-1 vs. top-2 margin check.
3. **34-Class vs. 38-Class Label Mismatch in `ml_service` (`ml_service/services/disease_predictor.py`):**
   - **Root Cause:** `ml_service/services/disease_predictor.py:15-50` defined a 34-element `CLASS_NAMES` list while the model output 38 logits, shifting indices `4..33` to the wrong diseases and dropping indices `34..37`.
   - **Fix:** Synchronized `ml_service/services/disease_predictor.py` with `PLANTVILLAGE_38_RAW_CLASSES` from `ml_config.py:164-203`.
4. **Unsuitable Primary Plant Identifier (`Sisigoks/FloraSense`):**
   - **Root Cause:** `Sisigoks/FloraSense` spreads probability mass across 9,985 global wild taxa, lacks `Ocimum tenuiflorum` (Tulsi), and achieved only `19/45 (42.22%)` on iNaturalist field photos, `25/118 (21.19%)` on the PlantDoc held-out half, and `15/76 (19.74%)` on PlantVillage studio leaves.
   - **Fix:** Integrated FloraCare's `models/plant_model.onnx` (`ConvNeXt-Tiny`, 68 species + `not_plant`, plus 7 general foliar conditions) as the primary plant identifier behind `predict_species()` and `predict_species_detailed()`. Demoted `FloraSense` to an optional `"Other possible matches (low confidence)"` top-5 expander (`app_pages/identify.py`).
5. **Studio-Only MobileNetV2 Disease Classifier Failing on Real Field Photos:**
   - **Root Cause:** `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` was trained exclusively on uniform studio-background PlantVillage leaves, scoring `74/76 (97.37%)` on PlantVillage studio images but collapsing to `39/118 (33.05%)` on the PlantDoc field-photo held-out half.
   - **Fix:** Integrated `models/model.onnx` (`wambugu71/crop_leaf_diseases_vit`, 38-class ViT quantized ONNX) as the primary disease classifier (`76/76 (100.00%)` on PlantVillage and `92/118 (77.97%)` top-1 / `115/118 (97.46%)` top-5 on the PlantDoc held-out half), keeping `MobileNetV2` as a configurable comparison/fallback backend (`PLANTVERSE_DISEASE_BACKEND=mobilenet_v2`).
6. **No Plant $\rightarrow$ Disease Consistency Gate (Impossible Cross-Crop Diagnoses):**
   - **Root Cause:** The disease classifier ran unconditionally on any uploaded image (e.g., diagnosing `"Grape with Black Rot"` on `peepal.jpg`, `money_plant.jpg`, and `curry_leaf.jpg`, or `"Tomato with Late Blight"` on `tulsi.jpg` and `hibiscus.jpg`).
   - **Fix:** Implemented strict host-crop consistency gating (`resolve_supported_disease_crop` and `CROP_TO_CLASS_INDICES` in `ml_config.py:206-265` and `plant_disease_detection.py:462-523`). Non-crop specimens immediately return `"Disease classification is not available for this plant."` (`health_status="Unsupported"`), and supported crops mask logits strictly to that crop's valid disease classes.
7. **Missing EXIF Orientation Transpose:**
   - **Root Cause:** Neither `plant_identification.py` nor `plant_disease_detection.py` applied `ImageOps.exif_transpose()`, causing rotated phone photos to be fed sideways or upside-down into the models.
   - **Fix:** Added `ImageOps.exif_transpose()` to both `_validate_and_orient_image()` (`plant_identification.py:220`) and `preprocess_image()` (`plant_disease_detection.py:204, 253`).

---

## 3. Index-to-Label Mapping Verifications

### 3.1 Plant Model (`models/plant_model.onnx` & `models/plant_labels.json`)
- **Input:** `pixel_values` `['batch', 3, 224, 224]` (`float32`), `input_size=224`, `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- **Outputs:** `species_logits` `['batch', 69]` (68 botanical species + `not_plant` at index `46`), `condition_logits` `['batch', 7]` (`healthy`, `fungal_leaf_spot`, `powdery_mildew`, `chlorosis_nutrient`, `pest_damage`, `wilting_water_stress`, `necrosis_scorch`).
- **Verification:** SHA-256 of `models/plant_labels.json` matches FloraCare's `backend/app/models/plant_labels.json` byte-for-byte, and all 68 botanical species IDs resolve with 0 missing entries in `data/plants_metadata.json`.

### 3.2 Key Indian Plants Coverage Comparison

| Required Indian Plant | Scientific Binomial | Present in `FloraSense` (9,985 classes)? | Present in `plant_model.onnx` (68 species)? |
| :--- | :--- | :--- | :--- |
| **Tulsi (Holy basil)** | *Ocimum tenuiflorum* | **NO** (only *Ocimum americanum* & *O. gratissimum*) | **YES** (`index 64`: `tulsi`) |
| **Neem** | *Azadirachta indica* | **YES** (`Azadirachta_indica`) | **YES** (`index 44`: `neem`) |
| **Peepal (Sacred fig)** | *Ficus religiosa* | **YES** (`Ficus_religiosa`) | **YES** (`index 52`: `peepal`) |
| **Curry leaf** | *Bergera koenigii* / *Murraya koenigii* | **YES** (`Murraya_koenigii`) | **YES** (`index 23`: `curry_leaf`) |
| **Hibiscus (Gudhal)** | *Hibiscus rosa-sinensis* | **YES** (`Hibiscus_rosa-sinensis`) | **YES** (`index 31`: `hibiscus`) |
| **Money plant (Pothos)** | *Epipremnum aureum* | **YES** (`Epipremnum_aureum`) | **YES** (`index 42`: `money_plant`) |

### 3.3 Disease Model (`models/model.onnx` & `models/labels.json`)
- **Input:** `pixel_values` `[1, 3, 224, 224]` (`float32`), `x / 127.5 - 1.0`.
- **Output:** `logits` `[1, 38]` (`float32`).
- **Verification:** All 38 indices (`0 = Apple___Apple_scab` through `37 = Tomato___healthy`) in `models/labels.json` match `PLANTVILLAGE_38_RAW_CLASSES` in `ml_config.py` with 0 mismatches (`tests/test_ml_pipeline.py::test_disease_38_class_mapping_and_splitting`).

---

## 4. Provisional Confidence Policy & Tuning-Split Methodology

To prevent test-set contamination (Ground Rule 3), the 236-image `PlantDoc` test set was split once with a fixed random seed (`seed=42`, saved in `tests/ml_eval/plantdoc_split_seed42.json`) into:
- **PlantDoc Tuning Half (`n = 118`):** Used exclusively for selecting preprocessing/cropping decisions and deriving provisional confidence thresholds.
- **PlantDoc Held-Out Half (`n = 118`):** Never touched during threshold or preprocessing selection; used strictly for final reporting.

### 4.1 Leaf Cropping Ablation on PlantDoc Tuning Half (`n = 118`)
- `ViT model.onnx` on **full image**: **`87 / 118` (`73.73%`)** Top-1 (`107 / 118`, `90.68%` Top-5).
- `ViT model.onnx` with **FloraCare-style `crop_to_bbox` leaf crop**: **`85 / 118` (`72.03%`)** Top-1 (`107 / 118`, `90.68%` Top-5).
- **Decision:** Because `crop_to_bbox` reduced Top-1 accuracy by `-1.70%` on the tuning half, leaf cropping before the ViT disease classifier was **dropped**. Classical CV leaf/lesion segmentation is used strictly for post-classification affected-area estimation and visual overlay rendering.

### 4.2 Provisional Thresholds Derived on Tuning Sets (`ml_config.py`)
- **Plant Identification (`models/plant_model.onnx`, provisional, derived on `n = 163`: 45 `iNaturalist` + 118 `PlantDoc` tuning half):**
  - Correct predictions (`n = 144`): `mean = 0.8527`, `p10 = 0.6336`, `p25 = 0.8066`, `median = 0.9265`.
  - Incorrect predictions (`n = 19`): `mean = 0.5238`, `median = 0.5394`, `p75 = 0.6754`.
  - Thresholds: `ONNX_PLANT_HIGH_CONF_THRESHOLD = 0.68`, `ONNX_PLANT_MEDIUM_CONF_THRESHOLD = 0.35`, `ONNX_PLANT_MIN_TOP1_TOP2_MARGIN = 0.15`, `ONNX_NOT_PLANT_THRESHOLD = 0.45`.
- **Disease Detection (`models/model.onnx`, provisional, derived on `n = 118` `PlantDoc` tuning half):**
  - Correct predictions (`n = 87`): `mean = 0.8201`, `p10 = 0.5512`, `p25 = 0.7734`, `median = 0.8932`.
  - Thresholds: `DISEASE_HIGH_CONF_THRESHOLD = 0.75`, `DISEASE_MEDIUM_CONF_THRESHOLD = 0.55`, `DISEASE_MIN_TOP1_TOP2_MARGIN = 0.10`.

---

## 5. Final Evaluation Results (Old vs. New Side-by-Side)

> [!WARNING]
> **Training-Data Overlap Disclosure:** FloraCare's `plant_model.onnx` was trained on a combined corpus of PlantVillage, PlantDoc, and iNaturalist images (`backend/training/train_plant_model.ipynb`), and `model.onnx` (`wambugu71/crop_leaf_diseases_vit`) was trained on PlantVillage, PlantDoc, and public field images without publishing exact training image SHA-256 hashes. Therefore, all benchmark comparisons marked with `*` on `PlantVillage`, `PlantDoc`, and `iNaturalist` are **flagged as possibly inflated by training data overlap**.

### 5.1 Overall Benchmark Accuracy Table (`tests/ml_eval/results/final_summary.json`)

| Task & Benchmark Dataset | Model / Pipeline Variant | Top-1 (`correct/total`) | Top-1 (%) | Top-5 (`correct/total`) | Top-5 (%) | Mean Conf (Correct vs. Incorrect) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Plant-ID: Public Field Species (`iNaturalist`, `n=45`)** | Old (`Sisigoks/FloraSense`) | `19/45` | `42.22%` | `23/45` | `51.11%` | `0.1502` vs. `0.0235` |
| **Plant-ID: Public Field Species (`iNaturalist`, `n=45`)\*** | **New (`ConvNeXt-Tiny plant_model.onnx`)\*** | **`32/45`** | **`71.11%`*** | **`42/45`** | **`93.33%`*** | `0.8693` vs. `0.3822` |
| **Plant-ID: Public Field Crops (`PlantDoc Held-Out Half`, `n=118`, `seed=42`)** | Old (`Sisigoks/FloraSense`) | `25/118` | `21.19%` | `39/118` | `33.05%` | `0.0241` vs. `0.0080` |
| **Plant-ID: Public Field Crops (`PlantDoc Held-Out Half`, `n=118`, `seed=42`)\*** | **New (`ConvNeXt-Tiny plant_model.onnx`)\*** | **`111/118`** | **`94.07%`*** | **`118/118`** | **`100.00%`*** | `0.8588` vs. `0.5629` |
| **Plant-ID: Dataset Studio Leaves (`PlantVillage`, `n=76`)** | Old (`Sisigoks/FloraSense`) | `15/76` | `19.74%` | `24/76` | `31.58%` | `0.0014` vs. `0.0020` |
| **Plant-ID: Dataset Studio Leaves (`PlantVillage`, `n=76`)\*** | **New (`ConvNeXt-Tiny plant_model.onnx`)\*** | **`74/76`** | **`97.37%`*** | **`76/76`** | **`100.00%`*** | `0.9467` vs. `0.5316` |
| **Plant-ID: Repo Smoke Test (`n=20`, *unverified — labels inferred from filenames*)** | Old (`Sisigoks/FloraSense`) | `10/20` | *Excluded (unverified)* | `11/20` | *Excluded (unverified)* | `0.0693` vs. `0.0047` |
| **Plant-ID: Repo Smoke Test (`n=20`, *unverified — labels inferred from filenames*)\*** | **New (`ConvNeXt-Tiny plant_model.onnx`)\*** | **`15/20`** | *Excluded (unverified)* | **`16/20`** | *Excluded (unverified)* | `0.8452` vs. `0.5763` |
| **Disease: Dataset Studio Leaves (`PlantVillage`, `n=76`)** | Old (`MobileNetV2` standalone raw) | `74/76` | `97.37%` | `76/76` | `100.00%` | `0.9317` vs. `0.4839` |
| **Disease: Dataset Studio Leaves (`PlantVillage`, `n=76`)\*** | New (`ViT model.onnx` standalone raw)\* | `75/76` | `98.68%`* | `76/76` | `100.00%`* | `0.9718` vs. `0.6254` |
| **Disease: Dataset Studio Leaves (`PlantVillage`, `n=76`)\*** | **New (`ViT model.onnx` + Crop Gate)\*** | **`76/76`** | **`100.00%`*** | **`76/76`** | **`100.00%`*** | `0.9698` vs. `N/A (0 errors)` |
| **Disease: Public Field Photos (`PlantDoc Held-Out Half`, `n=118`, `seed=42`)** | Old (`MobileNetV2` standalone raw) | `39/118` | `33.05%` | `86/118` | `72.88%` | `0.7336` vs. `0.6087` |
| **Disease: Public Field Photos (`PlantDoc Held-Out Half`, `n=118`, `seed=42`)\*** | New (`ViT model.onnx` standalone raw)\* | `91/118` | `77.12%`* | `114/118` | `96.61%`* | `0.8505` vs. `0.5835` |
| **Disease: Public Field Photos (`PlantDoc Held-Out Half`, `n=118`, `seed=42`)\*** | **New (`ViT model.onnx` + Crop Gate)\*** | **`92/118`** | **`77.97%`*** | **`115/118`** | **`97.46%`*** | `0.8442` vs. `0.5432` |
| **Disease: Repo Smoke Test (`n=20`, *unverified — labels inferred from filenames*)** | Old (`MobileNetV2` without Crop Gate) | `3/20` | *Excluded (unverified)* | `7/20` | *Excluded (unverified)* | `0.9185` vs. `0.6327` |
| **Disease: Repo Smoke Test (`n=20`, *unverified — labels inferred from filenames*)\*** | **New (`ViT model.onnx` + Crop Gate)\*** | **`19/20`** | *Excluded (unverified)* | **`20/20`** | *Excluded (unverified)* | `0.3017` vs. `0.3335` |
| **Real Phone-Camera Accuracy** | All Models | **NOT MEASURED** | **NOT MEASURED (no user images provided)** | **NOT MEASURED** | **NOT MEASURED** | N/A |

### 5.2 Per-Class Plant-ID Accuracy on `PlantDoc Held-Out Half` (`n = 118`, `seed=42`)

| Host Crop Class | Old `FloraSense` Top-1 | New `ConvNeXt-Tiny ONNX` Top-1\* | New `ConvNeXt-Tiny ONNX` Top-5\* |
| :--- | :---: | :---: | :---: |
| **Apple** (`n=16`) | `2/16 (12.5%)` | **`16/16 (100.0%)`** | `16/16 (100.0%)` |
| **Bell Pepper** (`n=8`) | `3/8 (37.5%)` | **`8/8 (100.0%)`** | `8/8 (100.0%)` |
| **Blueberry** (`n=6`) | `0/6 (0.0%)` | **`5/6 (83.3%)`** | `6/6 (100.0%)` |
| **Cherry** (`n=4`) | `1/4 (25.0%)` | **`4/4 (100.0%)`** | `4/4 (100.0%)` |
| **Corn (Maize)** (`n=15`) | `2/15 (13.3%)` | **`15/15 (100.0%)`** | `15/15 (100.0%)` |
| **Grape** (`n=12`) | `9/12 (75.0%)` | **`12/12 (100.0%)`** | `12/12 (100.0%)` |
| **Peach** (`n=6`) | `0/6 (0.0%)` | **`4/6 (66.7%)`** | `6/6 (100.0%)` |
| **Potato** (`n=6`) | `0/6 (0.0%)` | **`6/6 (100.0%)`** | `6/6 (100.0%)` |
| **Raspberry** (`n=1`) | `0/1 (0.0%)` | **`1/1 (100.0%)`** | `1/1 (100.0%)` |
| **Soybean** (`n=3`) | `0/3 (0.0%)` | **`2/3 (66.7%)`** | `3/3 (100.0%)` |
| **Squash** (`n=4`) | `3/4 (75.0%)` | **`4/4 (100.0%)`** | `4/4 (100.0%)` |
| **Strawberry** (`n=4`) | `2/4 (50.0%)` | **`4/4 (100.0%)`** | `4/4 (100.0%)` |
| **Tomato** (`n=33`) | `3/33 (9.1%)` | **`30/33 (90.9%)`** | `33/33 (100.0%)` |
| **Total (`n=118`)** | **`25/118 (21.19%)`** | **`111/118 (94.07%)`*** | **`118/118 (100.00%)`*** |

### 5.3 Per-Class Disease Accuracy on `PlantDoc Held-Out Half` (`n = 118`, `seed=42`)

| Disease / Healthy Class (`PlantDoc Held-Out Half`) | Old `MobileNetV2` Top-1 | New `ViT ONNX` Raw Top-1\* | New `ViT ONNX + Crop Gate` Top-1\* | New `ViT ONNX + Crop Gate` Top-5\* |
| :--- | :---: | :---: | :---: | :---: |
| **Apple Scab** (`n=4`) | `3/4` | `4/4` | **`4/4`** | `4/4` |
| **Bell Pepper with Bacterial Spot** (`n=4`) | `3/4` | `3/4` | **`3/4`** | `4/4` |
| **Cedar Apple Rust** (`n=7`) | `1/7` | `6/7` | **`7/7`** | `7/7` |
| **Corn (Maize) with Cercospora / Gray Leaf Spot** (`n=4`) | `3/4` | `1/4` | **`1/4`** | `4/4` |
| **Corn (Maize) with Common Rust** (`n=7`) | `3/7` | `6/7` | **`6/7`** | `7/7` |
| **Corn (Maize) with Northern Leaf Blight** (`n=4`) | `0/4` | `2/4` | **`2/4`** | `4/4` |
| **Grape with Black Rot** (`n=4`) | `2/4` | `4/4` | **`4/4`** | `4/4` |
| **Healthy Apple** (`n=5`) | `1/5` | `5/5` | **`5/5`** | `5/5` |
| **Healthy Bell Pepper Plant** (`n=4`) | `2/4` | `4/4` | **`4/4`** | `4/4` |
| **Healthy Blueberry Plant** (`n=6`) | `3/6` | `5/6` | **`5/6`** | `5/6` |
| **Healthy Cherry Plant** (`n=4`) | `0/4` | `4/4` | **`3/4`** | `3/4` |
| **Healthy Grape Plant** (`n=8`) | `5/8` | `7/8` | **`8/8`** | `8/8` |
| **Healthy Peach Plant** (`n=6`) | `0/6` | `4/6` | **`4/6`** | `6/6` |
| **Healthy Raspberry Plant** (`n=1`) | `1/1` | `1/1` | **`1/1`** | `1/1` |
| **Healthy Soybean Plant** (`n=3`) | `0/3` | `2/3` | **`2/3`** | `3/3` |
| **Healthy Strawberry Plant** (`n=4`) | `1/4` | `4/4` | **`4/4`** | `4/4` |
| **Healthy Tomato Plant** (`n=2`) | `0/2` | `2/2` | **`2/2`** | `2/2` |
| **Potato with Early Blight** (`n=3`) | `0/3` | `1/3` | **`1/3`** | `3/3` |
| **Potato with Late Blight** (`n=3`) | `0/3` | `2/3` | **`2/3`** | `3/3` |
| **Squash with Powdery Mildew** (`n=4`) | `3/4` | `4/4` | **`4/4`** | `4/4` |
| **Tomato Mosaic Virus** (`n=4`) | `0/4` | `2/4` | **`2/4`** | `4/4` |
| **Tomato Yellow Leaf Curl Virus** (`n=5`) | `0/5` | `5/5` | **`5/5`** | `5/5` |
| **Tomato with Bacterial Spot** (`n=7`) | `0/7` | `2/7` | **`2/7`** | `6/7` |
| **Tomato with Early Blight** (`n=4`) | `1/4` | `3/4` | **`3/4`** | `4/4` |
| **Tomato with Late Blight** (`n=3`) | `2/3` | `1/3` | **`1/3`** | `3/3` |
| **Tomato with Leaf Mold** (`n=2`) | `0/2` | `1/2` | **`1/2`** | `2/2` |
| **Tomato with Septoria Leaf Spot** (`n=6`) | `5/6` | `6/6` | **`6/6`** | `6/6` |
| **Total (`n=118`)** | **`39/118 (33.05%)`** | **`91/118 (77.12%)`*** | **`92/118 (77.97%)`*** | **`115/118 (97.46%)`*** |

---

## 6. Automated Test Suite (`pytest`) & 11-Scenario Matrix Outputs

### 6.1 `pytest` Execution Output (`22 passed in 4.89s`)
```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.7.0 -- /home/lumora/Desktop/PlantVerse/venv/bin/python3
collected 22 items

tests/test_disease_detection.py::TestPlantDiseaseIntegration::test_01_healthy_leaf_prediction PASSED [  4%]
tests/test_disease_detection.py::TestPlantDiseaseIntegration::test_02_diseased_leaf_prediction PASSED [  9%]
tests/test_disease_detection.py::TestPlantDiseaseIntegration::test_03_invalid_file_handling PASSED [ 13%]
tests/test_disease_detection.py::TestPlantDiseaseIntegration::test_04_save_diagnosis_as_sighting PASSED [ 18%]
tests/test_disease_detection.py::TestPlantDiseaseIntegration::test_05_refresh_application_persistence PASSED [ 22%]
tests/test_disease_detection.py::TestPlantDiseaseIntegration::test_06_open_previous_diagnosis PASSED [ 27%]
tests/test_disease_detection.py::TestPlantDiseaseIntegration::test_07_ml_service_offline_explicit_error PASSED [ 31%]
tests/test_disease_detection.py::TestPlantDiseaseIntegration::test_08_existing_plantverse_features PASSED [ 36%]
tests/test_disease_detection.py::TestPlantDiseaseIntegration::test_09_stream_preview_rewind PASSED [ 40%]
tests/test_ml_pipeline.py::test_preprocess_for_onnx_plant_shape_and_normalization PASSED [ 45%]
tests/test_ml_pipeline.py::test_preprocess_for_onnx_vit_shape_and_normalization PASSED [ 50%]
tests/test_ml_pipeline.py::test_exif_transpose_and_rgba_to_rgb PASSED    [ 54%]
tests/test_ml_pipeline.py::test_plant_labels_and_metadata_mapping PASSED [ 59%]
tests/test_ml_pipeline.py::test_disease_38_class_mapping_and_splitting PASSED [ 63%]
tests/test_ml_pipeline.py::test_plant_confidence_policy_tiers_and_margin PASSED [ 68%]
tests/test_ml_pipeline.py::test_disease_confidence_policy_tiers_and_margin PASSED [ 72%]
tests/test_ml_pipeline.py::test_consistency_gate_blocks_non_crop_plants PASSED [ 77%]
tests/test_ml_pipeline.py::test_consistency_gate_prevents_cross_crop_impossible_combinations PASSED [ 81%]
tests/test_ml_pipeline.py::test_severity_thresholds_and_derived_health_score PASSED [ 86%]
tests/test_ml_pipeline.py::test_vision_analysis_failed_segmentation_and_variegated_bypass PASSED [ 90%]
tests/test_ml_pipeline.py::test_knowledge_base_all_38_classes_and_unknown_fallback PASSED [ 95%]
tests/test_ml_pipeline.py::test_error_paths_missing_and_corrupted_models_and_bad_images PASSED [100%]

============================== 22 passed in 4.89s ==============================
```

### 6.2 11-Scenario Matrix & Existing PlantVerse Features (`tests/verify_11_scenarios_and_features.py`)

| Scenario | Real Measured Output |
| :--- | :--- |
| **1. Known plant (`tulsi.jpg`)** | `species='Tulsi (Holy basil)' (Ocimum tenuiflorum), score=0.9801, tier='high', is_reliable=True` |
| **2. Unknown / out-of-scope plant (`static/ginkgo.jpg`)** | `primary='Tamarind' (score=0.4769, tier='medium'), warning='Moderate confidence identification (47.7%; provisional threshold derived on n=163 tuning images)...', secondary_florasense=['Abies alba (0.09%)', 'Larix kaempferi (0.07%)', 'Paspalum distichum (0.07%)']` |
| **3. Non-plant image (synthetic pattern)** | `species='Not a plant', is_not_plant=True, not_plant_score=0.3484, tier='low', message='Plant could not be identified reliably. Please upload a clearer image.'` |
| **4. Healthy leaf (`healthy_tomato.jpg`)** | `plant='Tomato', disease='Healthy Foliage', health_status='Healthy', conf=97.7%, tier='high', affected_area=0.0%, severity='Healthy', health_score=100/100` |
| **5. Diseased leaf (`tomato_early_blight.jpg`)** | `plant='Tomato', disease='Early Blight', health_status='Diseased', conf=96.8%, tier='high', affected_area=6.2%, severity='Moderate', health_score=66/100` |
| **6. Wrong crop / Non-crop consistency gate** | Non-crop (`tulsi.jpg`): `health_status='Unsupported', msg='Disease classification is not available for this plant.'` \| Cross-crop (`apple_scab.jpg` gated by `Tomato`): `plant='Tomato', top1='Tomato___Late_blight'` (`0` Apple classes leaked) |
| **7. Blurry image (Gaussian blur `r=18`)** | `Plant-ID: score=0.1693, tier='low' \| Disease: conf=23.6%, tier='low', is_uncertain=True` |
| **8. Multiple leaves image** | `multiple_leaves_detected=True, quality_warnings=['Multiple leaves or foliage clusters detected in frame. For highest diagnostic accuracy, photograph a single isolated leaf.']` |
| **9. Cluttered background image** | `cluttered_background_detected=True, quality_warnings=['Cluttered or high-variance background detected around the specimen. Background colors may affect classical CV lesion boundary estimation.']` |
| **10. Missing model file** | `Raised RuntimeError: Missing plant identification model file at '/tmp/missing_model_xyz.onnx'. Please verify that 'models/plant_model.onnx' is present.` |
| **11. ML service offline** | `Raised RuntimeError: Configured external disease detection service at 'http://127.0.0.1:59999/predict' is offline or failed: ... Connection refused` |
| **Existing Features: Wikipedia, Wikidata, iNaturalist, GPS/AR, Library, Streamlit Pages** | Wikipedia (`title='Ocimum tenuiflorum'`, `428` chars, `5` medicinal bullets); Wikidata (`Family='Lamiaceae'`, `Genus='Ocimum'`); iNaturalist (`taxon_id=181060`, `50` observations near Delhi); GPS distance (`1148.09 km`); all 5 pages (`identify.py`, `disease_detection.py`, `ar_explorer.py`, `library.py`, `about.py`) rendered with `0` exceptions via `AppTest` and live server (`http://127.0.0.1:8501/_stcore/health -> ok`). |

---

## 7. FloraCare Concepts Incorporated vs. PlantVerse Features Preserved

- **Concepts Adapted from FloraCare (Reference Repo):**
  1. Dual-head `ConvNeXt-Tiny` ONNX plant model (`models/plant_model.onnx` + `models/plant_labels.json`) with 68 Indian/household/crop species + `not_plant` and 7 general foliar conditions.
  2. 38-class `ViT` quantized ONNX disease model (`models/model.onnx` + `models/labels.json`).
  3. Classical CV leaf & lesion segmentation (`HSV` + `Excess-Green` + border `CIE LAB` background chroma filter in `vision_analysis.py`) and variegated-foliage bypass.
  4. Structured organic/chemical treatments and cultural care (`sun`, `water`, `soil`) in `disease_recommendations.py`.
- **PlantVerse Branding & Features Strictly Preserved:**
  - All existing PlantVerse pages (`app_pages/identify.py`, `app_pages/disease_detection.py`, `app_pages/ar_explorer.py`, `app_pages/library.py`, `app_pages/about.py`), styling (`ui/chrome.py`, `ui/presentation.py`), multilingual translation selector (`LANGUAGES`), Wikipedia summary & medicinal extract, Wikidata SPARQL taxonomy tree, iNaturalist observation map & AR explorer, and persistent field herbarium / sighting storage (`data/diagnoses.json`).

---

## 8. Model & Dataset Licenses (`MODEL_LICENSES.md`)

| Asset | Source | License | Attribution & Conditions |
| :--- | :--- | :--- | :--- |
| `models/plant_model.onnx` | `timm/convnext_tiny.in12k_ft_in1k` fine-tuned on PlantVillage + PlantDoc + iNaturalist | **Apache-2.0** (base architecture) + **CC BY-NC-SA 4.0** (iNaturalist training photos) | Per-observation creator attributions shipped in `models/inat_attribution.csv` (3,907 records). Non-Commercial ShareAlike applies to the iNaturalist training corpus. |
| `models/model.onnx` | `wambugu71/crop_leaf_diseases_vit` | **CC BY-SA 3.0** | Credit to [wambugu71/crop_leaf_diseases_vit](https://huggingface.co/wambugu71/crop_leaf_diseases_vit) and **CC BY-SA 3.0** ShareAlike notice displayed in `MODEL_LICENSES.md`, `app_pages/about.py`, and `app_pages/disease_detection.py`. |
| `Sisigoks/FloraSense` | Hugging Face Hub (optional secondary suggestions) | Base `google/vit-base-patch16-224`: **Apache-2.0**; fine-tuned weights license unspecified | Used only as optional secondary low-confidence suggestions; not redistributed in repo. |
| `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` | Hugging Face Hub (fallback backend) | Base MobileNetV2: **Apache-2.0**; PlantVillage: **CC0 1.0** | Optional fallback backend (`PLANTVERSE_DISEASE_BACKEND=mobilenet_v2`). |
| `PlantDoc Dataset` | Singh et al., ACM CoDS COMAD 2020 | **CC BY 4.0** | Cited in `MODEL_LICENSES.md` and `tests/ml_eval/plantdoc_split_seed42.json`. |

---

## 9. Limitations & Recommendations for Future Field Validation

1. **Real Phone-Camera Performance Is Unverified (`NOT MEASURED`):**
   - No user phone-camera photographs were provided during this evaluation. While public field datasets (`PlantDoc` held-out half and `iNaturalist` research-grade photos) were evaluated, real-world user smartphone images vary in lens distortion, motion blur, mixed indoor lighting, and multi-plant framing.
   - **How to Measure Later:** Use the newly integrated local feedback widget (`ui/feedback.py`, which logs consenting user confirmations/corrections to `data/user_feedback_log.jsonl`) to collect a labeled corpus of real user uploads (`n >= 200`) and run `tests/ml_eval/run_eval.py` on those confirmed field captures to recalibrate the provisional thresholds in `ml_config.py`.
2. **Training-Data Overlap in Public Benchmarks:**
   - Because `plant_model.onnx` and `model.onnx` were trained on mixtures of `PlantVillage`, `PlantDoc`, and `iNaturalist` without publishing exact training image hashes, benchmark numbers on those public sets (`71.11%` iNaturalist, `94.07%` PlantDoc plant-ID, `77.97%` PlantDoc disease-ID) may be inflated by overlap. Truly independent accuracy measurement requires a newly photographed, strictly held-out field dataset.
3. **68-Species Scope of Primary Plant Model & Out-of-Distribution Taxa:**
   - `models/plant_model.onnx` covers 68 common Indian, household, ornamental, and agricultural species plus `not_plant`. Rare wild flora outside those 68 species (e.g., *Adiantum pedatum*, *Ginkgo biloba*, *Polypodiopsida* ferns in the repo smoke test) cannot be predicted as primary classes and will either trigger a medium/low confidence warning or be surfaced only in the secondary `FloraSense` top-5 suggestions expander.
4. **Classical CV Lesion Segmentation, Severity & Health Score Are Heuristics:**
   - Affected leaf area (`vision_analysis.py`) is computed via classical color-threshold segmentation (`HSV` + `Excess-Green` + border `CIE LAB`), **not** a trained semantic segmentation network. Strong cast shadows, specular glare, soil/bark matching necrotic hues, or variegated leaves (which are automatically bypassed) limit its precision.
   - Both the **Severity Category** (`Healthy`, `Early < 5%`, `Moderate 5–20%`, `Severe >= 20%`) and **Foliar Health Score (`0–100`)** are **project-defined heuristics (not scientifically validated agricultural standards)** and are visually separated from the neural model's output in the UI.
5. **Remaining Field Disease Confusions & Retraining Strategy:**
   - On the `PlantDoc` held-out half, the primary remaining errors occur on visually similar foliar spots (`Tomato with Bacterial Spot` vs. `Early Blight`/`Late Blight`, and `Corn Cercospora Gray Leaf Spot` vs. `Northern Leaf Blight`). Improving field discrimination requires fine-tuning on dermatologist/agronomist-verified field lesions with bounding-box or segmentation-mask supervision (e.g., Mask R-CNN / YOLOv8-seg + ViT).
