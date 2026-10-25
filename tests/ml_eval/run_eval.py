"""
PlantVerse ML Evaluation Harness (Phase 1 Baseline & Post-Fix Comparison)
=========================================================================
Runs the PlantVerse identification and disease detection pipelines across:
  1. Held-out PlantVillage dataset images (studio leaf domain)
  2. Public field-photo datasets:
     - PlantDoc test split (236 real field photos across 27 disease classes)
     - iNaturalist CC0/CC-BY research-grade species set (45 unseen field photos across 15 species)
  3. Unverified smoke test on repository images (static/, static/sample_leaves/, and reference sample files)
     -- Kept strictly separate from accuracy metrics; filenames used ONLY as expected test labels.
  4. Real phone-camera accuracy: explicitly reported as "NOT MEASURED (no user images provided)".

Outputs:
  - CSV files in tests/ml_eval/results/
  - Printed tables:
      image | predicted plant | confidence | expected plant | correct?
      image | predicted disease | confidence | expected disease | correct?
  - Overall accuracy, per-class accuracy, confusion summary, and confidence distribution (correct vs. incorrect).
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
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plant_identification import predict_species
from plant_disease_detection import predict_disease
from disease_recommendations import normalize_disease_label, RAW_LABEL_MAP

CACHE_DIR = Path("/tmp/plantverse_eval_data")
RESULTS_DIR = CURRENT_DIR / "results"
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
    # Also check genus match if expected_sci is provided
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


def evaluate_plant_id(records: List[Dict[str, Any]], dataset_name: str, csv_path: Path, print_limit: int = 30) -> Dict[str, Any]:
    rows = []
    correct_count = 0
    per_class = defaultdict(lambda: [0, 0])
    confusions = Counter()
    correct_confs = []
    incorrect_confs = []

    for r in records:
        img_path = r["image_path"]
        img = Image.open(img_path)
        pred_label, pred_conf = predict_species(img)
        exp_plant = r["expected_plant"]
        exp_sci = r.get("expected_scientific", "")
        ok = is_plant_match(pred_label, exp_plant, exp_sci)
        correct_count += int(ok)
        per_class[exp_plant][0] += int(ok)
        per_class[exp_plant][1] += 1
        if ok:
            correct_confs.append(float(pred_conf))
        else:
            incorrect_confs.append(float(pred_conf))
            confusions[(exp_plant, pred_label)] += 1

        rows.append({
            "dataset": dataset_name,
            "image": os.path.basename(img_path),
            "predicted_plant": pred_label,
            "confidence": round(float(pred_conf), 4),
            "expected_plant": f"{exp_plant} ({exp_sci})" if exp_sci and exp_sci != exp_plant else exp_plant,
            "correct": ok,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "image", "predicted_plant", "confidence", "expected_plant", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 95}")
    print(f"PLANT IDENTIFICATION EVALUATION: {dataset_name} (n={len(rows)})")
    print(f"{'=' * 95}")
    print(f"{'image':<32} | {'predicted plant':<24} | {'conf':<7} | {'expected plant':<24} | {'correct?'}")
    print(f"{'-' * 95}")
    for row in rows[:print_limit]:
        print(f"{row['image'][:32]:<32} | {str(row['predicted_plant'])[:24]:<24} | {row['confidence']:<7.4f} | {str(row['expected_plant'])[:24]:<24} | {row['correct']}")
    if len(rows) > print_limit:
        print(f"... ({len(rows) - print_limit} more rows written to {csv_path})")

    acc = correct_count / max(len(rows), 1)
    c_mean = float(np.mean(correct_confs)) if correct_confs else 0.0
    i_mean = float(np.mean(incorrect_confs)) if incorrect_confs else 0.0
    print(f"{'-' * 95}")
    if "SMOKE TEST" not in dataset_name.upper():
        print(f"Overall Plant-ID Accuracy : {correct_count}/{len(rows)} ({acc * 100:.2f}%)")
    else:
        print(f"Smoke Test Matches        : {correct_count}/{len(rows)} (Excluded from formal accuracy metrics)")
    print(f"Confidence Distribution   : Correct mean={c_mean:.4f} (n={len(correct_confs)}) | Incorrect mean={i_mean:.4f} (n={len(incorrect_confs)})")
    if confusions:
        top_conf = confusions.most_common(8)
        print("Top Confusions (expected -> predicted):", [f"{e} -> {p} ({n})" for (e, p), n in top_conf])

    return {
        "dataset": dataset_name,
        "total": len(rows),
        "correct": correct_count,
        "accuracy": acc if "SMOKE TEST" not in dataset_name.upper() else None,
        "correct_conf_mean": c_mean,
        "incorrect_conf_mean": i_mean,
        "per_class": {k: {"correct": ok, "total": tot, "acc": ok / tot} for k, (ok, tot) in per_class.items()},
        "top_confusions": [
            {"expected": e, "predicted": p, "count": n} for (e, p), n in confusions.most_common(10)
        ],
    }


def evaluate_disease(records: List[Dict[str, Any]], dataset_name: str, csv_path: Path, print_limit: int = 30) -> Dict[str, Any]:
    rows = []
    correct_count = 0
    per_class = defaultdict(lambda: [0, 0])
    confusions = Counter()
    correct_confs = []
    incorrect_confs = []

    for r in records:
        img_path = r["image_path"]
        img = Image.open(img_path)
        res = predict_disease(img)
        pred_raw = res.get("raw_label", "Unknown")
        pred_norm = normalize_disease_label(pred_raw) if pred_raw not in ("Unknown", "UNSUPPORTED_PLANT", "UNCERTAIN") else pred_raw
        pred_conf = float(res.get("confidence", 0.0))
        exp_raw = r["expected_disease_raw"]
        exp_norm = normalize_disease_label(exp_raw) if exp_raw != "UNSUPPORTED_NON_CROP" else "UNSUPPORTED_NON_CROP"

        if exp_raw == "UNSUPPORTED_NON_CROP":
            # For unsupported non-crop plants, the disease model is only correct if it refuses / marks unsupported
            ok = (
                res.get("health_status") in ("Unsupported", "Uncertain", "Unknown")
                or not res.get("is_supported_crop", True)
                or res.get("raw_label") in ("UNSUPPORTED_PLANT", "UNCERTAIN")
            )
        else:
            ok = is_disease_match(pred_raw, exp_raw) and res.get("health_status") not in ("Unsupported", "Uncertain")

        correct_count += int(ok)
        per_class[exp_norm][0] += int(ok)
        per_class[exp_norm][1] += 1
        if ok:
            correct_confs.append(pred_conf)
        else:
            incorrect_confs.append(pred_conf)
            confusions[(exp_norm, pred_norm)] += 1

        rows.append({
            "dataset": dataset_name,
            "image": os.path.basename(img_path),
            "predicted_disease": pred_norm,
            "confidence": round(pred_conf, 4),
            "expected_disease": exp_norm,
            "correct": ok,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "image", "predicted_disease", "confidence", "expected_disease", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 105}")
    print(f"DISEASE DETECTION EVALUATION: {dataset_name} (n={len(rows)})")
    print(f"{'=' * 105}")
    print(f"{'image':<28} | {'predicted disease':<32} | {'conf':<7} | {'expected disease':<26} | {'correct?'}")
    print(f"{'-' * 105}")
    for row in rows[:print_limit]:
        print(f"{row['image'][:28]:<28} | {str(row['predicted_disease'])[:32]:<32} | {row['confidence']:<7.4f} | {str(row['expected_disease'])[:26]:<26} | {row['correct']}")
    if len(rows) > print_limit:
        print(f"... ({len(rows) - print_limit} more rows written to {csv_path})")

    acc = correct_count / max(len(rows), 1)
    c_mean = float(np.mean(correct_confs)) if correct_confs else 0.0
    i_mean = float(np.mean(incorrect_confs)) if incorrect_confs else 0.0
    print(f"{'-' * 105}")
    if "SMOKE TEST" not in dataset_name.upper():
        print(f"Overall Disease Accuracy  : {correct_count}/{len(rows)} ({acc * 100:.2f}%)")
    else:
        print(f"Smoke Test Outcomes       : {correct_count}/{len(rows)} (Excluded from formal accuracy metrics)")
    print(f"Confidence Distribution   : Correct mean={c_mean:.4f} (n={len(correct_confs)}) | Incorrect mean={i_mean:.4f} (n={len(incorrect_confs)})")
    if confusions:
        top_conf = confusions.most_common(8)
        print("Top Confusions (expected -> predicted):", [f"{e} -> {p} ({n})" for (e, p), n in top_conf])

    return {
        "dataset": dataset_name,
        "total": len(rows),
        "correct": correct_count,
        "accuracy": acc if "SMOKE TEST" not in dataset_name.upper() else None,
        "correct_conf_mean": c_mean,
        "incorrect_conf_mean": i_mean,
        "per_class": {k: {"correct": ok, "total": tot, "acc": ok / tot} for k, (ok, tot) in per_class.items()},
        "top_confusions": [
            {"expected": e, "predicted": p, "count": n} for (e, p), n in confusions.most_common(10)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run PlantVerse ML Evaluation Harness")
    parser.add_argument("--tag", type=str, default="baseline", help="Tag for output files (e.g. baseline, phase2, phase5)")
    args = parser.parse_args()

    pv_manifest = CACHE_DIR / "plantvillage_test" / "manifest.json"
    pd_manifest = CACHE_DIR / "plantdoc_test_manifest.json"
    inat_manifest = CACHE_DIR / "inat_species_test" / "manifest.json"

    if not (pv_manifest.exists() and pd_manifest.exists() and inat_manifest.exists()):
        from tests.ml_eval.build_datasets import build_plantvillage_set, build_plantdoc_set, build_inat_species_set
        build_plantvillage_set(76)
        build_plantdoc_set()
        build_inat_species_set(3)

    pv_records = json.loads(pv_manifest.read_text(encoding="utf-8"))
    pd_records = json.loads(pd_manifest.read_text(encoding="utf-8"))
    inat_records = json.loads(inat_manifest.read_text(encoding="utf-8"))
    smoke_records = get_smoke_test_items()

    summary = {
        "tag": args.tag,
        "real_phone_camera_accuracy": "NOT MEASURED (no user images provided)",
        "plant_id": {},
        "disease": {},
    }

    # 1. Plant-ID Evaluation
    summary["plant_id"]["inat_field_species"] = evaluate_plant_id(
        inat_records,
        "Public Field Species Set (iNaturalist CC0/CC-BY, 15 species)",
        RESULTS_DIR / f"{args.tag}_plant_id_inat.csv",
        print_limit=45,
    )
    summary["plant_id"]["plantdoc_field_crops"] = evaluate_plant_id(
        pd_records,
        "Public Field Crops Set (PlantDoc Test Split)",
        RESULTS_DIR / f"{args.tag}_plant_id_plantdoc.csv",
        print_limit=25,
    )
    summary["plant_id"]["plantvillage_studio"] = evaluate_plant_id(
        pv_records,
        "Dataset-Image Set (PlantVillage Held-Out Test Split)",
        RESULTS_DIR / f"{args.tag}_plant_id_plantvillage.csv",
        print_limit=25,
    )
    summary["plant_id"]["smoke_test"] = evaluate_plant_id(
        smoke_records,
        "UNVERIFIED SMOKE TEST (Labels Inferred from Repo Filenames)",
        RESULTS_DIR / f"{args.tag}_plant_id_smoke.csv",
        print_limit=25,
    )

    # 2. Disease Detection Evaluation
    summary["disease"]["plantvillage_studio"] = evaluate_disease(
        pv_records,
        "Dataset-Image Set (PlantVillage Held-Out Test Split)",
        RESULTS_DIR / f"{args.tag}_disease_plantvillage.csv",
        print_limit=25,
    )
    summary["disease"]["plantdoc_field"] = evaluate_disease(
        pd_records,
        "Public Field-Photo Set (PlantDoc Test Split, 236 images)",
        RESULTS_DIR / f"{args.tag}_disease_plantdoc.csv",
        print_limit=30,
    )
    summary["disease"]["smoke_test"] = evaluate_disease(
        smoke_records,
        "UNVERIFIED SMOKE TEST (Labels Inferred from Repo Filenames)",
        RESULTS_DIR / f"{args.tag}_disease_smoke.csv",
        print_limit=25,
    )

    print(f"\n{'=' * 95}")
    print(f"SUMMARY REPORT ({args.tag.upper()})")
    print(f"{'=' * 95}")
    print(f"1. Plant-ID Public Field Accuracy (iNaturalist 45 imgs) : {summary['plant_id']['inat_field_species']['accuracy']*100:.2f}%")
    print(f"2. Plant-ID Public Field Crop Accuracy (PlantDoc 236)   : {summary['plant_id']['plantdoc_field_crops']['accuracy']*100:.2f}%")
    print(f"3. Plant-ID Dataset-Image Accuracy (PlantVillage 76)    : {summary['plant_id']['plantvillage_studio']['accuracy']*100:.2f}%")
    print(f"4. Disease Dataset-Image Accuracy (PlantVillage 76)     : {summary['disease']['plantvillage_studio']['accuracy']*100:.2f}%")
    print(f"5. Disease Public Field-Photo Accuracy (PlantDoc 236)   : {summary['disease']['plantdoc_field']['accuracy']*100:.2f}%")
    print(f"6. Smoke Test Outcomes (Repo Images, n={len(smoke_records)})             : Plant-ID {summary['plant_id']['smoke_test']['correct']}/{len(smoke_records)} | Disease {summary['disease']['smoke_test']['correct']}/{len(smoke_records)} (Unverified smoke test)")
    print(f"7. Real Phone-Camera Accuracy                           : {summary['real_phone_camera_accuracy']}")
    print(f"{'=' * 95}\n")

    summary_path = RESULTS_DIR / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved full JSON summary to {summary_path}")


if __name__ == "__main__":
    main()
