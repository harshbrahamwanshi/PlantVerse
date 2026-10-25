# PlantVerse — Model & Dataset Licenses and Attribution Notice

This document records the license terms, provenance, and attribution requirements for every machine learning model and dataset integrated into **PlantVerse**.

---

## 1. Primary Plant & Foliar Condition Classifier (`models/plant_model.onnx`)

- **Architecture:** ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`, exported to int8 ONNX, two classification heads: 68 plant species + `not_plant`, and 7 general foliar condition classes).
- **Base Pretrained Weights & Library (`timm`):** **Apache License 2.0** (Ross Wightman / Meta Platforms, Inc.).
- **Training Datasets & Fine-Tuned Weights Provenance:**
  1. **iNaturalist Research-Grade Observations:** Openly licensed field photographs (`CC0`, `CC BY`, `CC BY-SA`, `CC BY-NC`, `CC BY-NC-SA`). Individual photographer credits, observation URLs, and per-photo license tags for all 26,946 observations are preserved in [`models/inat_attribution.csv`](models/inat_attribution.csv).
  2. **PlantVillage Dataset** (Hughes & Salathé, 2015): **CC0 1.0 / CC BY-SA 3.0**.
  3. **PlantDoc Dataset** (Singh et al., 2020, *ACM CoDS COMAD*): **CC BY 4.0**.
  4. **Imagenette** (`not_plant` negative class): **BSD-3-Clause / ImageNet Research Terms**.
- **License & Redistribution Terms:** Because the training corpus includes `CC BY-NC` and `CC BY-NC-SA` iNaturalist observations alongside `CC BY-SA` datasets, the fine-tuned `models/plant_model.onnx` weights are distributed for **non-commercial, academic, and educational use under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**, with full photographer attribution in [`models/inat_attribution.csv`](models/inat_attribution.csv).

---

## 2. Primary Crop Disease Classifier (`models/model.onnx`)

- **Architecture:** Vision Transformer (`google/vit-base-patch16-224` fine-tuned on 38 PlantVillage & PlantDoc classes, quantized to int8 ONNX).
- **Upstream Base Checkpoint:** [`kimcomehome/plantvillage-vit-leaf-disease`](https://huggingface.co/kimcomehome/plantvillage-vit-leaf-disease), fine-tuned from `google/vit-base-patch16-224` (**Apache-2.0**) and further fine-tuned on PlantVillage + PlantDoc field photographs.
- **License:** **Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)** (https://creativecommons.org/licenses/by-sa/3.0/).
- **Attribution & ShareAlike Notice:**
  - *Original PlantVillage Dataset:* David P. Hughes and Marcel Salathé (2015), *"An open access repository of images on plant health to enable the development of mobile disease diagnostics"*, arXiv:1511.08060.
  - *Original PlantDoc Dataset:* Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, and Nipun Batra (2020), *"PlantDoc: A Dataset for Visual Plant Disease Detection"*, ACM CoDS COMAD 2020 (CC BY 4.0).
  - *ViT Checkpoint:* `kimcomehome/plantvillage-vit-leaf-disease` (CC BY-SA 3.0) and FloraCare / PhytoSense AR field-photo fine-tuned int8 ONNX export.
  - Any adaptations or redistributions of `models/model.onnx` must retain this attribution and be shared under **CC BY-SA 3.0** (or a compatible ShareAlike license).

---

## 3. Secondary / Comparison Models (Hugging Face Hub)

### A. `Sisigoks/FloraSense` (Optional Secondary Top-5 Suggestions Only)
- **Architecture:** `google/vit-base-patch16-224` (9,985 classes).
- **License Status:** Base architecture `google/vit-base-patch16-224` is **Apache-2.0**. The `Sisigoks/FloraSense` repository on Hugging Face Hub does not declare an explicit SPDX license tag on its fine-tuned weights; it is retained strictly as an optional secondary lookup via Hugging Face Hub and is not bundled as a primary binary artifact.

### B. `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (Comparison / Fallback)
- **Architecture:** `google/mobilenet_v2_1.0_224` (38 PlantVillage classes).
- **License Status:** Base model is **Apache-2.0**; trained on the **PlantVillage** dataset (**CC0 / CC BY-SA 3.0**). Retained only for benchmark comparison and fallback compatibility.

---

## 4. Evaluation Benchmark Datasets

1. **PlantVillage Test Split:** Hughes & Salathé (2015) — **CC0 / CC BY-SA 3.0**.
2. **PlantDoc Test Split (236 field images):** Singh et al. (2020), `https://github.com/pratikkayal/PlantDoc-Dataset` — **CC BY 4.0**.
3. **iNaturalist Evaluation Subset (45 field images):** Filtered strictly to **CC0** and **CC BY** research-grade observations via the public iNaturalist API.
