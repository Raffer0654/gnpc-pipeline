#!/usr/bin/env python
# Disease BAG prediction using protein + sex organ-age models.

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


DEBUG = True

MODEL_ROOT = Path("/home/jovyan/work/raffer pipelineScripts/results_imputation_miss30_sex")
DATA_PATH = Path("../GNPC Harmonised Dataset v1/Clinical/processed_data_v4_python/gnpc_processed_noncontrol.pkl")
ORG_MAP_PATH = Path("../GNPC Harmonised Dataset v1/Clinical/processed_data_v4_python/organ_system_mapping.json")

OUT_DIR = Path("/home/jovyan/work/raffer pipelineScripts/results_disease_comparison_sex")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_ALL = OUT_DIR / "all_disease_subjects"
OUT_DIR_SINGLE = OUT_DIR / "single_disease_subjects"
OUT_DIR_ALL.mkdir(parents=True, exist_ok=True)
OUT_DIR_SINGLE.mkdir(parents=True, exist_ok=True)

OUT_PATH_ALL = OUT_DIR_ALL / "disease_BAG_predictions_raw_ALLdiseases_imp_sex.csv"
OUT_PATH_SINGLE = OUT_DIR_SINGLE / "disease_BAG_predictions_raw_singleDisease_imp_sex.csv"

MODEL_SELECT_METRIC = "test_mae"
EXPECTED_MODELS = ("lasso", "en", "svr")


def looks_like_estimator(obj) -> bool:
    return hasattr(obj, "predict") and callable(getattr(obj, "predict"))


def get_model_label(model_obj):
    if isinstance(model_obj, Pipeline) and hasattr(model_obj, "named_steps") and "model" in model_obj.named_steps:
        est = model_obj.named_steps["model"]
        name = type(est).__name__.lower()
        if "svr" in name:
            return "svr"
        if "lasso" in name:
            return "lasso"
        if "elasticnet" in name or "elastic_net" in name:
            return "en"
        return name
    return type(model_obj).__name__.lower()


def safe_read_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def discover_candidate_models_for_organ(organ: str):
    out = []
    organ_dir = MODEL_ROOT / organ
    if not organ_dir.exists():
        return out
    for model_tag in EXPECTED_MODELS:
        md = organ_dir / model_tag
        if not md.exists():
            continue
        mpath = md / "best_model.joblib"
        metrics = safe_read_json(md / "metrics.json")
        out.append({
            "model_tag": model_tag,
            "model_dir": md,
            "best_model_path": mpath if mpath.exists() else None,
            "metrics": metrics,
        })
    return out


def pick_best_model_candidate(cands, metric_key: str):
    scored = []
    for c in cands:
        if c["best_model_path"] is None:
            continue
        metrics = c.get("metrics") or {}
        val = metrics.get(metric_key) or metrics.get("nested_cv_mae_mean")
        if val is None:
            continue
        try:
            scored.append((float(val), c))
        except Exception:
            continue
    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[0][1]
    for c in cands:
        if c["best_model_path"] is not None:
            return c
    return None


def subset_for_organ(base_df: pd.DataFrame, organ: str) -> pd.DataFrame:
    if "sex" in base_df.columns:
        if organ == "Female_Reproductive":
            return base_df[base_df["sex"] == 2].copy()
        if organ == "Male_Reproductive":
            return base_df[base_df["sex"] == 1].copy()
    return base_df.copy()


def load_preprocessing_artifact(organ: str) -> dict:
    path = MODEL_ROOT / organ / "preprocessing_artifacts.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing preprocessing artifact for {organ}: {path}\n"
            "Run 02_model_training_parallel_sex.py first."
        )
    with open(path, "r") as f:
        return json.load(f)


def prepare_X_and_rows(input_df: pd.DataFrame, organ: str):
    artifact = load_preprocessing_artifact(organ)
    feature_cols = artifact["feature_cols"]
    protein_cols = artifact.get("protein_cols", [c for c in feature_cols if c != "sex"])
    miss_thresh = float(artifact["miss_thresh"])
    medians = pd.Series(artifact.get("feature_medians", artifact["train_medians"]), dtype="float32")

    missing_feats = [f for f in feature_cols if f not in input_df.columns]
    if missing_feats:
        if DEBUG:
            print(f"Skipping {organ}: missing {len(missing_feats)} features (e.g., {missing_feats[:5]}).")
        return None, None

    work_df = input_df.copy()
    X_df = work_df[feature_cols].astype("float32")
    missing_rate = X_df[protein_cols].isna().mean(axis=1)
    keep_mask = missing_rate <= miss_thresh
    before_rows = len(work_df)
    work_df = work_df.loc[keep_mask].copy()
    X_df = X_df.loc[keep_mask].copy().fillna(medians).astype("float32")

    if DEBUG:
        print(f"  [{organ}] protein+sex preprocessing: {before_rows} -> {len(work_df)} rows")

    if work_df.empty:
        return None, None
    return work_df, X_df


def positive_diseases_row(row: pd.Series, disease_cols: list[str]) -> list[str]:
    return [d for d in disease_cols if int(row.get(d, 0)) == 1]


print("-" * 70)
print("LOADING DISEASE DATA + ORGAN MAP + PROTEIN+SEX MODELS")
print("-" * 70)

df = pd.read_pickle(DATA_PATH)
with open(ORG_MAP_PATH, "r") as f:
    organ_map = json.load(f)
if not MODEL_ROOT.exists():
    raise FileNotFoundError(f"MODEL_ROOT not found: {MODEL_ROOT}")

disease_cols = [
    "stroke", "tia", "tbi", "ad", "ftd", "pd", "als", "mci_sci",
    "cancer", "diabetes", "chf", "copd", "mi", "afib", "angina",
    "hyperlipidaemia", "hypertension", "depression", "anxiety",
]
disease_cols = [c for c in disease_cols if c in df.columns]

if "age" not in df.columns:
    raise KeyError("Disease dataframe missing required column: age")
if "sample_id" not in df.columns:
    raise KeyError("Disease dataframe missing required column: sample_id")
if "sex" not in df.columns:
    raise KeyError("Disease dataframe missing required column: sex")

df = df.copy()
df[disease_cols] = df[disease_cols].replace({-1: 0, 2: 0}).clip(0, 1).astype(int)
df["disease_count"] = df[disease_cols].sum(axis=1).astype(int)
df = df[df["age"].notna() & df["sex"].isin([1, 2]) & (df["disease_count"] >= 1)].copy()
df["sample_id"] = df["sample_id"].astype(str)
df["is_single_disease"] = (df["disease_count"] == 1).astype(int)

print(f"All-disease cohort: N={len(df):,}")
print(f"Single-disease subset: N={int(df['is_single_disease'].sum()):,}")

organ_best = {}
for organ in organ_map.keys():
    best = pick_best_model_candidate(discover_candidate_models_for_organ(organ), MODEL_SELECT_METRIC)
    if best is not None:
        organ_best[organ] = {
            "model_tag": best["model_tag"],
            "model_path": best["best_model_path"],
            "metrics": best.get("metrics"),
            "model_dir": best["model_dir"],
        }

print(f"\nSelected best protein+sex model for {len(organ_best)} organs.")

records_long = []
for organ in sorted(organ_best.keys()):
    target_df = subset_for_organ(df, organ)
    if target_df.empty:
        continue
    target_df, X_df = prepare_X_and_rows(target_df, organ)
    if target_df is None or X_df is None or target_df.empty:
        continue

    model_path = organ_best[organ]["model_path"]
    model_tag = organ_best[organ]["model_tag"]
    model = joblib.load(model_path)
    if not looks_like_estimator(model):
        raise ValueError(f"[{organ}] Loaded object is not a predictor: {model_path}")

    if DEBUG:
        print(
            f"  [DEBUG {organ}] X={X_df.shape} expects={getattr(model, 'n_features_in_', None)} "
            f"model={get_model_label(model)}"
        )

    X_np = X_df.to_numpy(dtype=np.float32)
    age_vec = target_df["age"].astype(float).to_numpy()
    sid_vec = target_df["sample_id"].to_numpy()
    y_pred = model.predict(X_np)
    bag = y_pred - age_vec

    for i in range(len(sid_vec)):
        row = target_df.iloc[i]
        for disease in positive_diseases_row(row, disease_cols):
            records_long.append([
                sid_vec[i], organ, model_tag, disease,
                int(row["disease_count"]), int(row["is_single_disease"]),
                float(age_vec[i]), float(y_pred[i]), float(bag[i]),
            ])

out_all = pd.DataFrame(records_long, columns=[
    "sample_id", "organ", "model", "Disease",
    "disease_count", "is_single_disease",
    "age_chron", "age_pred_raw", "BAG_raw",
])

if disease_cols and len(out_all):
    out_all = out_all.merge(
        df[["sample_id", "disease_count", "is_single_disease", "sex"] + disease_cols]
        .drop_duplicates("sample_id"),
        on=["sample_id", "disease_count", "is_single_disease"],
        how="left",
    )

print("\n" + "-" * 70)
print(f"Saving ALL-disease protein+sex predictions: {len(out_all):,}")
print(f"  -> {OUT_PATH_ALL}")
out_all.to_csv(OUT_PATH_ALL, index=False)

out_single = out_all[out_all["is_single_disease"] == 1].copy()
print(f"Saving SINGLE-disease protein+sex predictions: {len(out_single):,}")
print(f"  -> {OUT_PATH_SINGLE}")
out_single.to_csv(OUT_PATH_SINGLE, index=False)
print("Done.")
