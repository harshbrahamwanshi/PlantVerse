"""
PlantVerse ML Evaluation Harness (Phase 1 Baseline & Task 9 Final Comparison)
=============================================================================
Runs the PlantVerse identification and disease detection pipelines across:
  1. Held-out PlantVillage dataset images (n=76 studio leaf domain across 38 classes)
  2. Public field-photo datasets:
     - PlantDoc held-out half (n=118 real field photos from fixed seed=42 split in
       tests/ml_eval/plantdoc_split_seed42.json; tuning half n=118 reported separately)
     - iNaturalist CC0/CC-BY research-grade species set (n=45 field photos across 15 species)
  3. Unverified smoke test on repository images (n=20: static/, static/sample_leaves/, and reference sample files)
     -- Kept strictly separate from accuracy metrics; filenames used ONLY as expected test labels.
  4. Real phone-camera accuracy: explicitly reported as "NOT MEASURED (no user images provided)".

Outputs:
  - CSV files and JSON summary in tests/ml_eval/results/
  - Printed tables:
      image | predicted plant | confidence | expected plant | correct?
      image | predicted disease | confidence | expected disease | correct?
  - Side-by-side Old vs. New Top-1 & Top-5 accuracy (correct/total and %), per-class accuracy,
    confusion summary, confidence distribution (correct vs. incorrect), and training-overlap flags.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from disease_recommendations import RAW_LABEL_MAP, normalize_disease_label
from ml_config import PLANTVILLAGE_38_RAW_CLASSES
from plant_disease_detection import (
    _compute_raw_38_probabilities,
    load_disease_classifier,
    predict_disease,
    preprocess_image,
)
from plant_identification import (
    get_florasense_secondary_suggestions,
    predict_species,
    predict_species_detailed,
)

CACHE_DIR = Path("/tmp/plantverse_eval_data")
RESULTS_DIR = CURRENT_DIR / "results"
SPLIT_MANIFEST_PATH = CURRENT_DIR / "plantdoc_split_seed42.json"
FLORACARE_REF_DIR = Path("/tmp/FloraCare_ref")


# Canonical equivalence aliases for comparing predicted plant species against expected plant names
PLANT_ALIASES = {
    "apple": {"apple", "malus domestica", "malus pumila", "malus"},
    "blueberry": {"blueberry", "vaccinium corymbosum", "vaccinium"},
    "cherry": {"cherry", "prunus avium", "prunus cerasus", "cherry (including sour)"},
    "corn": {"corn", "corn (maize)", "maize", "zea mays"},
    "grape": {"grape", "vitis vinifera", "vitis"},
    "orange": {"orange", "orange (fruit)", "orange / citrus", "citrus", "citrus sinensis"},
    "peach": {"peach", "prunus persica"},
    "bell pepper": {"bell pepper", "pepper, bell", "capsicum annuum", "capsicum", "capsicum / chilli"},
    "potato": {"potato", "solanum tuberosum"},
    "raspberry": {"raspberry", "rubus idaeus", "rubus"},
    "soybean": {"soybean", "glycine max"},
    "squash": {"squash", "cucurbita", "squash / pumpkin", "cucurbita pepo", "cucurbita moschata"},
    "strawberry": {"strawberry", "fragaria ananassa", "fragaria", "fragaria x ananassa"},
    "tomato": {"tomato", "solanum lycopersicum", "lycopersicon esculentum"},
    "tulsi": {"tulsi", "tulsi (holy basil)", "ocimum tenuiflorum", "ocimum sanctum"},
    "neem": {"neem", "azadirachta indica"},
    "peepal": {"peepal", "peepal (sacred fig)", "ficus religiosa"},
    "curry leaf": {"curry leaf", "bergera koenigii", "murraya koenigii"},
    "hibiscus": {"hibiscus", "hibiscus (gudhal)", "hibiscus rosa-sinensis", "hibiscus rosa sinensis"},
    "money plant": {"money plant", "money plant (pothos)", "epipremnum aureum", "epipremnum"},
    "banyan": {"banyan", "ficus benghalensis"},
    "rose": {"rose", "rosa", "rosa chinensis", "rosa indica", "rosa sp", "rosa moschata"},
    "aloe": {"aloe", "aloe vera"},
    "mango": {"mango", "mangifera indica"},
    "snake plant": {"snake plant", "sansevieria trifasciata", "dracaena trifasciata", "sansevieria"},
    "croton": {"croton", "codiaeum variegatum", "codiaeum", "codiaeum spp"},
    "ginkgo": {"ginkgo", "ginkgo biloba"},
    "adiantum": {"adiantum", "adiantum pedatum", "northern maidenhair fern"},
    "fern": {"fern", "polypodiopsida", "dryopteris", "phegopteris connectilis", "dryopteris filix-mas"},
}


def canonical_plant(name: str) -> str:
    if not name:
        return "unknown"
    clean = name.strip().lower().replace("_", " ")
    for canon, aliases in PLANT_ALIASES.items():
        if clean in aliases:
            return canon
        for a in aliases:
            if len(a) > 4 and (clean == a or clean.startswith(a + " ") or a.startswith(clean + " ")):
                return canon
    return clean


def is_plant_match(pred_plant: str, expected_plant: str, expected_sci: str = "") -> bool:
    c_pred = canonical_plant(pred_plant)
    c_exp = canonical_plant(expected_plant)
    c_sci = canonical_plant(expected_sci) if expected_sci else ""
    if c_pred == "unknown" or not c_pred:
        return False
    if c_pred == c_exp or (c_sci and c_pred == c_sci):
        return True
    if expected_sci:
        exp_genus = expected_sci.split()[0].lower()
        pred_first = pred_plant.strip().split()[0].lower()
        if len(exp_genus) >= 4 and exp_genus == pred_first and exp_genus not in ("solanum", "prunus"):
            return True
    return False


def is_disease_match(pred_raw_label: str, expected_raw_label: str) -> bool:
    if not pred_raw_label or not expected_raw_label:
        return False
    norm_pred = normalize_disease_label(pred_raw_label)
    norm_exp = normalize_disease_label(expected_raw_label)
    return norm_pred == norm_exp


def get_smoke_test_items() -> List[Dict[str, Any]]:
    """
    Build the unverified smoke test list from existing repository images + reference samples.
    Filenames are used ONLY as expected labels for evaluation, never during inference.
    """
    items = [
        {
            "image_path": str(PROJECT_ROOT / "static/hero.jpg"),
            "expected_plant": "Adiantum pedatum",
            "expected_scientific": "Adiantum pedatum",
            "expected_disease_raw": "UNSUPPORTED_NON_CROP",
        },
        {
            "image_path": str(PROJECT_ROOT / "static/ginkgo.jpg"),
            "expected_plant": "Ginkgo biloba",
            "expected_scientific": "Ginkgo biloba",
            "expected_disease_raw": "UNSUPPORTED_NON_CROP",
        },
        {
            "image_path": str(PROJECT_ROOT / "static/fern-plate.jpg"),
            "expected_plant": "Fern",
            "expected_scientific": "Polypodiopsida",
            "expected_disease_raw": "UNSUPPORTED_NON_CROP",
        },
        {
            "image_path": str(PROJECT_ROOT / "static/sample_leaves/tomato_early_blight.jpg"),
            "expected_plant": "Tomato",
            "expected_scientific": "Solanum lycopersicum",
            "expected_disease_raw": "Tomato___Early_blight",
        },
        {
            "image_path": str(PROJECT_ROOT / "static/sample_leaves/apple_scab.jpg"),
            "expected_plant": "Apple",
            "expected_scientific": "Malus domestica",
            "expected_disease_raw": "Apple___Apple_scab",
        },
        {
            "image_path": str(PROJECT_ROOT / "static/sample_leaves/healthy_tomato.jpg"),
            "expected_plant": "Tomato",
            "expected_scientific": "Solanum lycopersicum",
            "expected_disease_raw": "Tomato___healthy",
        },
        {
            "image_path": str(PROJECT_ROOT / "static/sample_leaves/healthy_apple.jpg"),
            "expected_plant": "Apple",
            "expected_scientific": "Malus domestica",
            "expected_disease_raw": "Apple___healthy",
        },
    ]

    ref_samples = [
        ("frontend/samples/hibiscus.jpg", "Hibiscus", "Hibiscus rosa-sinensis", "UNSUPPORTED_NON_CROP"),
        ("frontend/samples/tulsi.jpg", "Tulsi", "Ocimum tenuiflorum", "UNSUPPORTED_NON_CROP"),
        ("frontend/samples/neem.jpg", "Neem", "Azadirachta indica", "UNSUPPORTED_NON_CROP"),
        ("frontend/samples/peepal.jpg", "Peepal", "Ficus religiosa", "UNSUPPORTED_NON_CROP"),
        ("frontend/samples/money_plant.jpg", "Money plant", "Epipremnum aureum", "UNSUPPORTED_NON_CROP"),
        ("frontend/samples/curry_leaf.jpg", "Curry leaf", "Bergera koenigii", "UNSUPPORTED_NON_CROP"),
        ("frontend/samples/snake_plant.jpg", "Snake plant", "Dracaena trifasciata", "UNSUPPORTED_NON_CROP"),
        ("frontend/samples/croton.jpg", "Croton", "Codiaeum variegatum", "UNSUPPORTED_NON_CROP"),
        ("frontend/samples/rose_black_spot.jpg", "Rose", "Rosa", "UNSUPPORTED_NON_CROP"),
        ("frontend/samples/rose_powdery_mildew.jpg", "Rose", "Rosa", "UNSUPPORTED_NON_CROP"),
        ("frontend/samples/late_blight.jpg", "Tomato", "Solanum lycopersicum", "Tomato___Late_blight"),
        ("frontend/samples/corn_rust.jpg", "Corn (Maize)", "Zea mays", "Corn_(maize)___Common_rust_"),
        ("samples/Potato_Late_blight_5305.jpg", "Potato", "Solanum tuberosum", "Potato___Late_blight"),
    ]
    for rel, exp_p, exp_s, exp_d in ref_samples:
        p = FLORACARE_REF_DIR / rel
        if p.exists():
            items.append({
                "image_path": str(p),
                "expected_plant": exp_p,
                "expected_scientific": exp_s,
                "expected_disease_raw": exp_d,
            })
    return items


def evaluate_plant_id(
    records: List[Dict[str, Any]],
    dataset_name: str,
    csv_path: Path,
    backend: str = "onnx_convnext",
    print_limit: int = 30,
) -> Dict[str, Any]:
    rows = []
    top1_correct = 0
    top5_correct = 0
    per_class = defaultdict(lambda: [0, 0, 0])  # [top1_ok, top5_ok, total]
    confusions = Counter()
    correct_confs = []
    incorrect_confs = []

    if csv_path.exists() and csv_path.stat().st_size > 100:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cached_rows = list(reader)
        if len(cached_rows) == len(records):
            for r, cr in zip(records, cached_rows):
                exp_plant = r["expected_plant"]
                pred_label = cr["predicted_plant"]
                pred_conf = float(cr["confidence"])
                ok1 = cr["correct"].strip().lower() == "true"
                ok5 = cr.get("top5_correct", cr["correct"]).strip().lower() == "true"
                top1_correct += int(ok1)
                top5_correct += int(ok5)
                per_class[exp_plant][0] += int(ok1)
                per_class[exp_plant][1] += int(ok5)
                per_class[exp_plant][2] += 1
                if ok1:
                    correct_confs.append(pred_conf)
                else:
                    incorrect_confs.append(pred_conf)
                    confusions[(exp_plant, pred_label)] += 1
                rows.append({
                    "dataset": dataset_name,
                    "backend": backend,
                    "image": cr["image"],
                    "predicted_plant": pred_label,
                    "confidence": pred_conf,
                    "expected_plant": cr["expected_plant"],
                    "correct": ok1,
                    "top5_correct": ok5,
                })

    if not rows:
        for r in records:
            img_path = r["image_path"]
            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            exp_plant = r["expected_plant"]
            exp_sci = r.get("expected_scientific", "")

            if backend == "florasense":
                cands = get_florasense_secondary_suggestions(img, top_k=5)
                if cands:
                    pred_label = cands[0]["species"]
                    pred_conf = float(cands[0]["score"])
                    top5_labels = [c["species"] for c in cands]
                else:
                    pred_label = "Unknown"
                    pred_conf = 0.0
                    top5_labels = []
            else:
                det = predict_species_detailed(img, include_secondary_florasense=False)
                pred_label = det["species"]
                pred_conf = float(det["score"])
                top5_labels = []
                for c in det.get("top_candidates", [])[:5]:
                    top5_labels.append(c.get("species", ""))
                    if c.get("scientific_name"):
                        top5_labels.append(c["scientific_name"])

            ok1 = is_plant_match(pred_label, exp_plant, exp_sci)
            ok5 = ok1 or any(is_plant_match(lbl, exp_plant, exp_sci) for lbl in top5_labels)

            top1_correct += int(ok1)
            top5_correct += int(ok5)
            per_class[exp_plant][0] += int(ok1)
            per_class[exp_plant][1] += int(ok5)
            per_class[exp_plant][2] += 1

            if ok1:
                correct_confs.append(pred_conf)
            else:
                incorrect_confs.append(pred_conf)
                confusions[(exp_plant, pred_label)] += 1

            rows.append({
                "dataset": dataset_name,
                "backend": backend,
                "image": os.path.basename(img_path),
                "predicted_plant": pred_label,
                "confidence": round(pred_conf, 4),
                "expected_plant": f"{exp_plant} ({exp_sci})" if exp_sci and exp_sci != exp_plant else exp_plant,
                "correct": ok1,
                "top5_correct": ok5,
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "backend", "image", "predicted_plant", "confidence", "expected_plant", "correct", "top5_correct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 102}")
    print(f"PLANT IDENTIFICATION EVALUATION [{backend}]: {dataset_name} (n={len(rows)})")
    print(f"{'=' * 102}")
    print(f"{'image':<30} | {'predicted plant':<24} | {'conf':<7} | {'expected plant':<24} | {'correct?'}")
    print(f"{'-' * 102}")
    for row in rows[:print_limit]:
        print(
            f"{row['image'][:30]:<30} | {str(row['predicted_plant'])[:24]:<24} | "
            f"{row['confidence']:<7.4f} | {str(row['expected_plant'])[:24]:<24} | {row['correct']}"
        )
    if len(rows) > print_limit:
        print(f"... ({len(rows) - print_limit} more rows written to {csv_path})")

    acc1 = top1_correct / max(len(rows), 1)
    acc5 = top5_correct / max(len(rows), 1)
    c_mean = float(np.mean(correct_confs)) if correct_confs else 0.0
    i_mean = float(np.mean(incorrect_confs)) if incorrect_confs else 0.0
    print(f"{'-' * 102}")
    if "SMOKE TEST" not in dataset_name.upper():
        print(f"Top-1 Accuracy            : {top1_correct}/{len(rows)} ({acc1 * 100:.2f}%)")
        print(f"Top-5 Accuracy            : {top5_correct}/{len(rows)} ({acc5 * 100:.2f}%)")
    else:
        print(f"Smoke Test Matches        : Top-1 {top1_correct}/{len(rows)} | Top-5 {top5_correct}/{len(rows)} (Excluded from formal accuracy metrics)")
    print(f"Confidence Distribution   : Correct mean={c_mean:.4f} (n={len(correct_confs)}) | Incorrect mean={i_mean:.4f} (n={len(incorrect_confs)})")
    if confusions:
        top_conf = confusions.most_common(6)
        print("Top Confusions (expected -> predicted):", [f"{e} -> {p} ({n})" for (e, p), n in top_conf])

    return {
        "dataset": dataset_name,
        "backend": backend,
        "total": len(rows),
        "correct": top1_correct,
        "top5_correct": top5_correct,
        "accuracy": acc1 if "SMOKE TEST" not in dataset_name.upper() else None,
        "top5_accuracy": acc5 if "SMOKE TEST" not in dataset_name.upper() else None,
        "correct_conf_mean": c_mean,
        "incorrect_conf_mean": i_mean,
        "per_class": {
            k: {"correct": ok1, "top5_correct": ok5, "total": tot, "acc": ok1 / tot, "top5_acc": ok5 / tot}
            for k, (ok1, ok5, tot) in per_class.items()
        },
        "top_confusions": [
            {"expected": e, "predicted": p, "count": n} for (e, p), n in confusions.most_common(10)
        ],
    }


def evaluate_disease(
    records: List[Dict[str, Any]],
    dataset_name: str,
    csv_path: Path,
    backend: str = "onnx_vit",
    enforce_crop_gate: bool = True,
    print_limit: int = 30,
) -> Dict[str, Any]:
    rows = []
    top1_correct = 0
    top5_correct = 0
    per_class = defaultdict(lambda: [0, 0, 0])  # [top1_ok, top5_ok, total]
    confusions = Counter()
    correct_confs = []
    incorrect_confs = []

    old_env = os.environ.get("PLANTVERSE_DISEASE_BACKEND")
    os.environ["PLANTVERSE_DISEASE_BACKEND"] = backend

    try:
        for r in records:
            img_path = r["image_path"]
            img = preprocess_image(img_path)
            exp_raw = r["expected_disease_raw"]
            exp_norm = normalize_disease_label(exp_raw) if exp_raw != "UNSUPPORTED_NON_CROP" else "UNSUPPORTED_NON_CROP"

            if enforce_crop_gate:
                res = predict_disease(img, enforce_crop_gate=True)
                pred_raw = res.get("raw_label", "Unknown")
                pred_norm = (
                    normalize_disease_label(pred_raw)
                    if pred_raw not in ("Unknown", "UNSUPPORTED_PLANT", "UNCERTAIN")
                    else pred_raw
                )
                pred_conf = float(res.get("confidence", 0.0))
                top5_raw = [c.get("raw_label", "") for c in res.get("top_candidates", [])[:5]]
                if exp_raw == "UNSUPPORTED_NON_CROP":
                    ok1 = (
                        res.get("health_status") in ("Unsupported", "Uncertain", "Unknown")
                        or not res.get("is_supported_crop", True)
                        or pred_raw in ("UNSUPPORTED_PLANT", "UNCERTAIN")
                    )
                    ok5 = ok1
                else:
                    ok1 = is_disease_match(pred_raw, exp_raw) and res.get("health_status") not in ("Unsupported", "Uncertain")
                    ok5 = any(is_disease_match(c_raw, exp_raw) for c_raw in top5_raw) and res.get("health_status") not in ("Unsupported",)
            else:
                # Raw standalone classifier evaluation (no crop gate) for direct model comparison
                probs, _ = _compute_raw_38_probabilities(img)
                sorted_idx = np.argsort(probs)[::-1]
                top_idx = int(sorted_idx[0])
                pred_raw = PLANTVILLAGE_38_RAW_CLASSES[top_idx]
                pred_norm = normalize_disease_label(pred_raw)
                pred_conf = float(probs[top_idx])
                top5_raw = [PLANTVILLAGE_38_RAW_CLASSES[int(i)] for i in sorted_idx[:5]]
                if exp_raw == "UNSUPPORTED_NON_CROP":
                    ok1 = False
                    ok5 = False
                else:
                    ok1 = is_disease_match(pred_raw, exp_raw)
                    ok5 = any(is_disease_match(c_raw, exp_raw) for c_raw in top5_raw)

            top1_correct += int(ok1)
            top5_correct += int(ok5)
            per_class[exp_norm][0] += int(ok1)
            per_class[exp_norm][1] += int(ok5)
            per_class[exp_norm][2] += 1

            if ok1:
                correct_confs.append(pred_conf)
            else:
                incorrect_confs.append(pred_conf)
                confusions[(exp_norm, pred_norm)] += 1

            rows.append({
                "dataset": dataset_name,
                "backend": f"{backend}{'_gated' if enforce_crop_gate else '_raw'}",
                "image": os.path.basename(img_path),
                "predicted_disease": pred_norm,
                "confidence": round(pred_conf, 4),
                "expected_disease": exp_norm,
                "correct": ok1,
                "top5_correct": ok5,
            })
    finally:
        if old_env is None:
            os.environ.pop("PLANTVERSE_DISEASE_BACKEND", None)
        else:
            os.environ["PLANTVERSE_DISEASE_BACKEND"] = old_env

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "backend", "image", "predicted_disease", "confidence", "expected_disease", "correct", "top5_correct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    mode_str = f"{backend} ({'crop-gated' if enforce_crop_gate else 'standalone raw'})"
    print(f"\n{'=' * 108}")
    print(f"DISEASE DETECTION EVALUATION [{mode_str}]: {dataset_name} (n={len(rows)})")
    print(f"{'=' * 108}")
    print(f"{'image':<28} | {'predicted disease':<32} | {'conf':<7} | {'expected disease':<26} | {'correct?'}")
    print(f"{'-' * 108}")
    for row in rows[:print_limit]:
        print(
            f"{row['image'][:28]:<28} | {str(row['predicted_disease'])[:32]:<32} | "
            f"{row['confidence']:<7.4f} | {str(row['expected_disease'])[:26]:<26} | {row['correct']}"
        )
    if len(rows) > print_limit:
        print(f"... ({len(rows) - print_limit} more rows written to {csv_path})")

    acc1 = top1_correct / max(len(rows), 1)
    acc5 = top5_correct / max(len(rows), 1)
    c_mean = float(np.mean(correct_confs)) if correct_confs else 0.0
    i_mean = float(np.mean(incorrect_confs)) if incorrect_confs else 0.0
    print(f"{'-' * 108}")
    if "SMOKE TEST" not in dataset_name.upper():
        print(f"Top-1 Accuracy            : {top1_correct}/{len(rows)} ({acc1 * 100:.2f}%)")
        print(f"Top-5 Accuracy            : {top5_correct}/{len(rows)} ({acc5 * 100:.2f}%)")
    else:
        print(f"Smoke Test Outcomes       : Top-1 {top1_correct}/{len(rows)} | Top-5 {top5_correct}/{len(rows)} (Excluded from formal accuracy metrics)")
    print(f"Confidence Distribution   : Correct mean={c_mean:.4f} (n={len(correct_confs)}) | Incorrect mean={i_mean:.4f} (n={len(incorrect_confs)})")
    if confusions:
        top_conf = confusions.most_common(6)
        print("Top Confusions (expected -> predicted):", [f"{e} -> {p} ({n})" for (e, p), n in top_conf])

    return {
        "dataset": dataset_name,
        "backend": mode_str,
        "total": len(rows),
        "correct": top1_correct,
        "top5_correct": top5_correct,
        "accuracy": acc1 if "SMOKE TEST" not in dataset_name.upper() else None,
        "top5_accuracy": acc5 if "SMOKE TEST" not in dataset_name.upper() else None,
        "correct_conf_mean": c_mean,
        "incorrect_conf_mean": i_mean,
        "per_class": {
            k: {"correct": ok1, "top5_correct": ok5, "total": tot, "acc": ok1 / tot, "top5_acc": ok5 / tot}
            for k, (ok1, ok5, tot) in per_class.items()
        },
        "top_confusions": [
            {"expected": e, "predicted": p, "count": n} for (e, p), n in confusions.most_common(10)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run PlantVerse ML Evaluation Harness")
    parser.add_argument("--tag", type=str, default="final", help="Tag for output files (e.g. baseline, final)")
    args = parser.parse_args()

    pv_manifest = CACHE_DIR / "plantvillage_test" / "manifest.json"
    pd_manifest = CACHE_DIR / "plantdoc_test_manifest.json"
    inat_manifest = CACHE_DIR / "inat_species_test" / "manifest.json"

    if not (pv_manifest.exists() and pd_manifest.exists() and inat_manifest.exists()):
        from tests.ml_eval.build_datasets import build_inat_species_set, build_plantdoc_set, build_plantvillage_set
        build_plantvillage_set(76)
        build_plantdoc_set()
        build_inat_species_set(3)

    pv_records = json.loads(pv_manifest.read_text(encoding="utf-8"))
    inat_records = json.loads(inat_manifest.read_text(encoding="utf-8"))
    smoke_records = get_smoke_test_items()

    # Load fixed-seed PlantDoc split (seed=42: 118 tuning, 118 held-out)
    split_data = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    pd_tuning_records = split_data.get("tuning", split_data.get("tuning_records", []))
    pd_heldout_records = split_data.get("heldout", split_data.get("held_out_records", []))

    summary: Dict[str, Any] = {
        "tag": args.tag,
        "real_phone_camera_accuracy": "NOT MEASURED (no user images provided)",
        "training_overlap_warning": (
            "WARNING: FloraCare's plant_model.onnx (trained on PlantVillage + PlantDoc + iNaturalist) "
            "and model.onnx (wambugu71/crop_leaf_diseases_vit, trained on PlantVillage + PlantDoc + field images) "
            "do not publish exact training image hashes/IDs. Public benchmark numbers on PlantVillage, PlantDoc, "
            "and iNaturalist are flagged as POSSIBLY INFLATED by training data overlap."
        ),
        "plantdoc_split": {
            "seed": split_data.get("seed", 42),
            "tuning_count": len(pd_tuning_records),
            "held_out_count": len(pd_heldout_records),
        },
        "plant_id": {},
        "disease": {},
    }

    # ==========================================================================
    # 1. PLANT IDENTIFICATION: NEW PRIMARY (ConvNeXt-Tiny ONNX) vs OLD (FloraSense)
    # ==========================================================================
    summary["plant_id"]["new_onnx_inat"] = evaluate_plant_id(
        inat_records,
        "Public Field Species Set (iNaturalist 45, New ConvNeXt-Tiny ONNX) [POSSIBLY INFLATED BY OVERLAP]",
        RESULTS_DIR / f"{args.tag}_plant_id_onnx_inat.csv",
        backend="onnx_convnext",
        print_limit=20,
    )
    summary["plant_id"]["old_florasense_inat"] = evaluate_plant_id(
        inat_records,
        "Public Field Species Set (iNaturalist 45, Old FloraSense)",
        RESULTS_DIR / f"{args.tag}_plant_id_florasense_inat.csv",
        backend="florasense",
        print_limit=15,
    )

    summary["plant_id"]["new_onnx_plantdoc_heldout"] = evaluate_plant_id(
        pd_heldout_records,
        "PlantDoc HELD-OUT Half (n=118, seed=42, New ConvNeXt-Tiny ONNX) [POSSIBLY INFLATED BY OVERLAP]",
        RESULTS_DIR / f"{args.tag}_plant_id_onnx_plantdoc_heldout.csv",
        backend="onnx_convnext",
        print_limit=20,
    )
    summary["plant_id"]["old_florasense_plantdoc_heldout"] = evaluate_plant_id(
        pd_heldout_records,
        "PlantDoc HELD-OUT Half (n=118, seed=42, Old FloraSense)",
        RESULTS_DIR / f"{args.tag}_plant_id_florasense_plantdoc_heldout.csv",
        backend="florasense",
        print_limit=15,
    )

    summary["plant_id"]["new_onnx_plantvillage"] = evaluate_plant_id(
        pv_records,
        "PlantVillage Test Split (n=76, New ConvNeXt-Tiny ONNX) [POSSIBLY INFLATED BY OVERLAP]",
        RESULTS_DIR / f"{args.tag}_plant_id_onnx_plantvillage.csv",
        backend="onnx_convnext",
        print_limit=20,
    )
    summary["plant_id"]["old_florasense_plantvillage"] = evaluate_plant_id(
        pv_records,
        "PlantVillage Test Split (n=76, Old FloraSense)",
        RESULTS_DIR / f"{args.tag}_plant_id_florasense_plantvillage.csv",
        backend="florasense",
        print_limit=15,
    )

    summary["plant_id"]["new_onnx_smoke"] = evaluate_plant_id(
        smoke_records,
        "UNVERIFIED SMOKE TEST (n=20 Repo Images, New ConvNeXt-Tiny ONNX)",
        RESULTS_DIR / f"{args.tag}_plant_id_onnx_smoke.csv",
        backend="onnx_convnext",
        print_limit=20,
    )
    summary["plant_id"]["old_florasense_smoke"] = evaluate_plant_id(
        smoke_records,
        "UNVERIFIED SMOKE TEST (n=20 Repo Images, Old FloraSense)",
        RESULTS_DIR / f"{args.tag}_plant_id_florasense_smoke.csv",
        backend="florasense",
        print_limit=20,
    )

    # ==========================================================================
    # 2. DISEASE DETECTION: NEW (ViT ONNX Gated & Raw) vs OLD (MobileNetV2 Raw)
    # ==========================================================================
    summary["disease"]["new_vit_gated_plantvillage"] = evaluate_disease(
        pv_records,
        "PlantVillage (n=76, New ViT ONNX + Crop Gate) [POSSIBLY INFLATED BY OVERLAP]",
        RESULTS_DIR / f"{args.tag}_disease_vit_gated_plantvillage.csv",
        backend="onnx_vit",
        enforce_crop_gate=True,
        print_limit=20,
    )
    summary["disease"]["new_vit_raw_plantvillage"] = evaluate_disease(
        pv_records,
        "PlantVillage (n=76, New ViT ONNX Standalone Raw) [POSSIBLY INFLATED BY OVERLAP]",
        RESULTS_DIR / f"{args.tag}_disease_vit_raw_plantvillage.csv",
        backend="onnx_vit",
        enforce_crop_gate=False,
        print_limit=15,
    )
    summary["disease"]["old_mobilenet_raw_plantvillage"] = evaluate_disease(
        pv_records,
        "PlantVillage (n=76, Old MobileNetV2 Standalone Raw)",
        RESULTS_DIR / f"{args.tag}_disease_mobilenet_raw_plantvillage.csv",
        backend="mobilenet_v2",
        enforce_crop_gate=False,
        print_limit=15,
    )

    summary["disease"]["new_vit_gated_plantdoc_heldout"] = evaluate_disease(
        pd_heldout_records,
        "PlantDoc HELD-OUT Half (n=118, seed=42, New ViT ONNX + Crop Gate) [POSSIBLY INFLATED BY OVERLAP]",
        RESULTS_DIR / f"{args.tag}_disease_vit_gated_plantdoc_heldout.csv",
        backend="onnx_vit",
        enforce_crop_gate=True,
        print_limit=20,
    )
    summary["disease"]["new_vit_raw_plantdoc_heldout"] = evaluate_disease(
        pd_heldout_records,
        "PlantDoc HELD-OUT Half (n=118, seed=42, New ViT ONNX Standalone Raw) [POSSIBLY INFLATED BY OVERLAP]",
        RESULTS_DIR / f"{args.tag}_disease_vit_raw_plantdoc_heldout.csv",
        backend="onnx_vit",
        enforce_crop_gate=False,
        print_limit=15,
    )
    summary["disease"]["old_mobilenet_raw_plantdoc_heldout"] = evaluate_disease(
        pd_heldout_records,
        "PlantDoc HELD-OUT Half (n=118, seed=42, Old MobileNetV2 Standalone Raw)",
        RESULTS_DIR / f"{args.tag}_disease_mobilenet_raw_plantdoc_heldout.csv",
        backend="mobilenet_v2",
        enforce_crop_gate=False,
        print_limit=15,
    )

    summary["disease"]["new_vit_gated_smoke"] = evaluate_disease(
        smoke_records,
        "UNVERIFIED SMOKE TEST (n=20 Repo Images, New ViT ONNX + Crop Gate)",
        RESULTS_DIR / f"{args.tag}_disease_vit_gated_smoke.csv",
        backend="onnx_vit",
        enforce_crop_gate=True,
        print_limit=20,
    )
    summary["disease"]["old_mobilenet_raw_smoke"] = evaluate_disease(
        smoke_records,
        "UNVERIFIED SMOKE TEST (n=20 Repo Images, Old MobileNetV2 Raw without Gate)",
        RESULTS_DIR / f"{args.tag}_disease_mobilenet_raw_smoke.csv",
        backend="mobilenet_v2",
        enforce_crop_gate=False,
        print_limit=20,
    )

    print(f"\n{'=' * 112}")
    print(f"SIDE-BY-SIDE COMPARISON SUMMARY ({args.tag.upper()})")
    print(f"{'=' * 112}")
    print("WARNING:", summary["training_overlap_warning"])
    print(f"{'-' * 112}")
    print(f"{'Task / Benchmark Set':<48} | {'Model Variant':<28} | {'Top-1 (correct/total)':<21} | {'Top-5 (correct/total)'}")
    print(f"{'-' * 112}")

    def _fmt(d: Dict[str, Any], is_smoke: bool = False) -> Tuple[str, str]:
        c1, c5, tot = d["correct"], d["top5_correct"], d["total"]
        if is_smoke:
            return f"{c1}/{tot} (unverified)", f"{c5}/{tot} (unverified)"
        return f"{c1}/{tot} ({c1/tot*100:.2f}%)", f"{c5}/{tot} ({c5/tot*100:.2f}%)"

    rows_summary = [
        ("Plant-ID: iNaturalist Field (n=45)", "Old FloraSense", *(_fmt(summary["plant_id"]["old_florasense_inat"]))),
        ("Plant-ID: iNaturalist Field (n=45)*", "New ConvNeXt-Tiny ONNX*", *(_fmt(summary["plant_id"]["new_onnx_inat"]))),
        ("Plant-ID: PlantDoc Held-Out Half (n=118)", "Old FloraSense", *(_fmt(summary["plant_id"]["old_florasense_plantdoc_heldout"]))),
        ("Plant-ID: PlantDoc Held-Out Half (n=118)*", "New ConvNeXt-Tiny ONNX*", *(_fmt(summary["plant_id"]["new_onnx_plantdoc_heldout"]))),
        ("Plant-ID: PlantVillage Studio (n=76)", "Old FloraSense", *(_fmt(summary["plant_id"]["old_florasense_plantvillage"]))),
        ("Plant-ID: PlantVillage Studio (n=76)*", "New ConvNeXt-Tiny ONNX*", *(_fmt(summary["plant_id"]["new_onnx_plantvillage"]))),
        ("Plant-ID: Repo Smoke Test (n=20, unverified)", "Old FloraSense", *(_fmt(summary["plant_id"]["old_florasense_smoke"], True))),
        ("Plant-ID: Repo Smoke Test (n=20, unverified)*", "New ConvNeXt-Tiny ONNX*", *(_fmt(summary["plant_id"]["new_onnx_smoke"], True))),
        ("Disease: PlantVillage Studio (n=76)", "Old MobileNetV2 (raw)", *(_fmt(summary["disease"]["old_mobilenet_raw_plantvillage"]))),
        ("Disease: PlantVillage Studio (n=76)*", "New ViT ONNX (raw)*", *(_fmt(summary["disease"]["new_vit_raw_plantvillage"]))),
        ("Disease: PlantVillage Studio (n=76)*", "New ViT ONNX (+ Crop Gate)*", *(_fmt(summary["disease"]["new_vit_gated_plantvillage"]))),
        ("Disease: PlantDoc Held-Out Half (n=118)", "Old MobileNetV2 (raw)", *(_fmt(summary["disease"]["old_mobilenet_raw_plantdoc_heldout"]))),
        ("Disease: PlantDoc Held-Out Half (n=118)*", "New ViT ONNX (raw)*", *(_fmt(summary["disease"]["new_vit_raw_plantdoc_heldout"]))),
        ("Disease: PlantDoc Held-Out Half (n=118)*", "New ViT ONNX (+ Crop Gate)*", *(_fmt(summary["disease"]["new_vit_gated_plantdoc_heldout"]))),
        ("Disease: Repo Smoke Test (n=20, unverified)", "Old MobileNetV2 (no gate)", *(_fmt(summary["disease"]["old_mobilenet_raw_smoke"], True))),
        ("Disease: Repo Smoke Test (n=20, unverified)*", "New ViT ONNX (+ Crop Gate)*", *(_fmt(summary["disease"]["new_vit_gated_smoke"], True))),
    ]
    for b_name, m_var, t1_s, t5_s in rows_summary:
        print(f"{b_name:<48} | {m_var:<28} | {t1_s:<21} | {t5_s}")
    print(f"{'-' * 112}")
    print(f"Real Phone-Camera Accuracy: {summary['real_phone_camera_accuracy']}")
    print("* Flagged as POSSIBLY INFLATED by training data overlap (PlantVillage / PlantDoc / iNaturalist).")
    print(f"{'=' * 112}\n")

    summary_path = RESULTS_DIR / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved full JSON summary to {summary_path}")


if __name__ == "__main__":
    main()
