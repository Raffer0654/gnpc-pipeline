#!/usr/bin/env python
# Disease-specific BAG age-bias correction using protein + sex organ-age models.
#
# For each Disease x organ:
#   1) Fit BAG_raw ~ age in all subjects positive for that disease, including comorbid subjects.
#   2) Apply those disease-derived alpha/beta parameters to subjects with only that disease
#      (disease_count == 1).
#
# This is not CN training correction. It is a disease-group independent correction
# targeted at single-disease outputs.

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


DEBUG = True

MODEL_ROOT = Path("results_imputation_miss30_sex")
DISEASE_DATA_PATH = Path("../GNPC Harmonised Dataset v1/Clinical/processed_data_v4_python/gnpc_processed_noncontrol.pkl")
ORG_MAP_PATH = Path("../GNPC Harmonised Dataset v1/Clinical/processed_data_v4_python/organ_system_mapping.json")

OUT_DIR = Path("results_disease_comparison_diseaseFitCorrection_singleDisease_sex")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH_LONG = OUT_DIR / "disease_BAG_predictions_diseaseFitCorrected_long_singleDisease_imp_sex.csv"
OUT_PATH_WIDE = OUT_DIR / "disease_BAG_predictions_diseaseFitCorrected_wide_singleDisease_imp_sex.csv"
PARAMS_OUT_PATH = OUT_DIR / "disease_fit_bias_params_by_disease_organ_imp_sex.csv"

MODEL_SELECT_METRIC = "test_mae"
MIN_N_TO_FIT_BIAS = 30
MIN_APPLY_N = 1
FALLBACK_IF_BAD_FIT = "raw"  # "raw" or "nan"
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


def fit_bias_params(age_vec: np.ndarray, y_pred_vec: np.ndarray):
    age_vec = np.asarray(age_vec, dtype=float)
    y_pred_vec = np.asarray(y_pred_vec, dtype=float)
    mask = np.isfinite(age_vec) & np.isfinite(y_pred_vec)
    age_fit = age_vec[mask]
    pred_fit = y_pred_vec[mask]

    if len(age_fit) < MIN_N_TO_FIT_BIAS:
        return np.nan, np.nan, len(age_fit), "too_few_samples"
    if np.nanstd(age_fit) == 0:
        return np.nan, np.nan, len(age_fit), "no_age_variation"

    try:
        bag_fit = pred_fit - age_fit
        alpha, beta = np.polyfit(age_fit, bag_fit, 1)
        return float(alpha), float(beta), len(age_fit), "fit_success"
    except Exception:
        return np.nan, np.nan, len(age_fit), "fit_failed"


def apply_bias_correction(age_vec: np.ndarray, y_pred_vec: np.ndarray, alpha: float, beta: float):
    age_vec = np.asarray(age_vec, dtype=float)
    y_pred_vec = np.asarray(y_pred_vec, dtype=float)
    bag_raw = y_pred_vec - age_vec
    return bag_raw - (alpha * age_vec + beta)


def positive_diseases_row(row: pd.Series, disease_cols: list[str]) -> list[str]:
    return [d for d in disease_cols if int(row.get(d, 0)) == 1]


print("-" * 70)
print("LOADING DISEASE DATA + ORGAN MAP + PROTEIN+SEX MODELS")
print("-" * 70)

df = pd.read_pickle(DISEASE_DATA_PATH)
with open(ORG_MAP_PATH, "r") as f:
    organ_map = json.load(f)

if not MODEL_ROOT.exists():
    raise FileNotFoundError(f"MODEL_ROOT not found: {MODEL_ROOT}")

for col in ["age", "sample_id", "sex"]:
    if col not in df.columns:
        raise KeyError(f"Disease dataframe missing required column: {col}")

df = df[df["age"].notna() & df["sex"].isin([1, 2])].copy()
df["sample_id"] = df["sample_id"].astype(str)

disease_cols = [
    "stroke", "tia", "tbi", "ad", "ftd", "pd", "als", "mci_sci",
    "cancer", "diabetes", "chf", "copd", "mi", "afib", "angina",
    "hyperlipidaemia", "hypertension", "depression", "anxiety",
]
disease_cols = [c for c in disease_cols if c in df.columns]
df[disease_cols] = df[disease_cols].replace({-1: 0, 2: 0}).clip(0, 1).astype(int)
df["disease_count"] = df[disease_cols].sum(axis=1).astype(int)

df_all_disease = df[df["disease_count"] >= 1].copy()
df_single_disease = df[df["disease_count"] == 1].copy()
df_all_disease["is_single_disease"] = (df_all_disease["disease_count"] == 1).astype(int)
df_single_disease["is_single_disease"] = 1

print(f"All disease-positive cohort for fitting: N={len(df_all_disease):,}")
print(f"Single-disease cohort for application:  N={len(df_single_disease):,}")

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

model_cache = {}
param_rows = []
long_records = []

print("\n" + "=" * 70)
print("FITTING DISEASE-SPECIFIC PARAMETERS ON ALL POSITIVES")
print("APPLYING TO SINGLE-DISEASE SUBJECTS")
print("=" * 70)

for disease in disease_cols:
    fit_base = df_all_disease[df_all_disease[disease] == 1].copy()
    apply_base = df_single_disease[df_single_disease[disease] == 1].copy()

    if len(fit_base) < MIN_N_TO_FIT_BIAS:
        if DEBUG:
            print(f"\n[{disease}] skipped fits: all-positive N={len(fit_base)} < {MIN_N_TO_FIT_BIAS}")
        continue
    if len(apply_base) < MIN_APPLY_N:
        if DEBUG:
            print(f"\n[{disease}] skipped apply: single-disease N={len(apply_base)} < {MIN_APPLY_N}")
        continue

    print(f"\n[{disease}] fit_N_all_positive={len(fit_base):,} | apply_N_single={len(apply_base):,}")

    for organ in sorted(organ_best.keys()):
        model_tag = organ_best[organ]["model_tag"]
        model_path = organ_best[organ]["model_path"]

        model = model_cache.get(organ)
        if model is None:
            model = joblib.load(model_path)
            if not looks_like_estimator(model):
                raise ValueError(f"[{organ}] Loaded object is not a predictor: {model_path}")
            model_cache[organ] = model

        fit_df_org = subset_for_organ(fit_base, organ)
        fit_df_org, X_fit_df = prepare_X_and_rows(fit_df_org, organ)

        param_row = {
            "Disease": disease,
            "organ": organ,
            "model": model_tag,
            "model_path": str(model_path),
            "alpha_disease_fit": np.nan,
            "beta_disease_fit": np.nan,
            "fit_status": "not_fit",
            "fit_N_all_positive": 0,
            "apply_N_single_disease": 0,
            "feature_set": "protein_plus_sex",
        }

        if fit_df_org is None or X_fit_df is None or fit_df_org.empty:
            param_row["fit_status"] = "no_fit_rows_after_preprocess"
            param_rows.append(param_row)
            continue

        fit_age = fit_df_org["age"].astype(float).to_numpy()
        fit_pred_raw = model.predict(X_fit_df.to_numpy(dtype=np.float32)).astype(float)
        fit_bag_raw = fit_pred_raw - fit_age
        alpha, beta, fit_n, fit_status = fit_bias_params(fit_age, fit_pred_raw)

        apply_df_org = subset_for_organ(apply_base, organ)
        apply_df_org, X_apply_df = prepare_X_and_rows(apply_df_org, organ)
        apply_n = 0 if apply_df_org is None else len(apply_df_org)

        param_row.update({
            "alpha_disease_fit": alpha,
            "beta_disease_fit": beta,
            "fit_status": fit_status,
            "fit_N_all_positive": fit_n,
            "apply_N_single_disease": apply_n,
            "fit_age_min": float(np.nanmin(fit_age)) if len(fit_age) else np.nan,
            "fit_age_max": float(np.nanmax(fit_age)) if len(fit_age) else np.nan,
            "fit_BAG_raw_mean": float(np.nanmean(fit_bag_raw)) if len(fit_bag_raw) else np.nan,
            "fit_BAG_raw_sd": float(np.nanstd(fit_bag_raw, ddof=1)) if len(fit_bag_raw) > 1 else np.nan,
        })
        param_rows.append(param_row)

        if apply_df_org is None or X_apply_df is None or apply_df_org.empty:
            continue

        apply_age = apply_df_org["age"].astype(float).to_numpy()
        apply_sid = apply_df_org["sample_id"].to_numpy()
        apply_pred_raw = model.predict(X_apply_df.to_numpy(dtype=np.float32)).astype(float)
        apply_bag_raw = apply_pred_raw - apply_age

        if np.isfinite(alpha) and np.isfinite(beta):
            apply_bag_corr = apply_bias_correction(apply_age, apply_pred_raw, alpha, beta)
            apply_pred_corr = apply_age + apply_bag_corr
            correction_status = "applied_disease_fit_params"
        else:
            correction_status = f"no_valid_disease_fit__{fit_status}"
            if FALLBACK_IF_BAD_FIT == "nan":
                apply_bag_corr = np.full_like(apply_bag_raw, np.nan, dtype=float)
                apply_pred_corr = np.full_like(apply_pred_raw, np.nan, dtype=float)
            else:
                apply_bag_corr = apply_bag_raw.copy()
                apply_pred_corr = apply_pred_raw.copy()
                correction_status += "__raw_fallback"

        for i in range(len(apply_sid)):
            row_i = apply_df_org.iloc[i]
            pos_dis = positive_diseases_row(row_i, disease_cols)
            rec = {
                "sample_id": str(apply_sid[i]),
                "Disease": disease,
                "organ": organ,
                "model": model_tag,
                "disease_count": int(row_i.get("disease_count", len(pos_dis))),
                "is_single_disease": int(row_i.get("is_single_disease", int(len(pos_dis) == 1))),
                "age_chron": float(apply_age[i]),
                "sex": int(row_i["sex"]),
                "age_pred_raw": float(apply_pred_raw[i]),
                "BAG_raw": float(apply_bag_raw[i]),
                "age_pred_corrected_disease_fit": float(apply_pred_corr[i]) if np.isfinite(apply_pred_corr[i]) else np.nan,
                "BAG_corrected_disease_fit": float(apply_bag_corr[i]) if np.isfinite(apply_bag_corr[i]) else np.nan,
                "alpha_disease_fit": alpha,
                "beta_disease_fit": beta,
                "fit_status": fit_status,
                "correction_status": correction_status,
                "fit_N_all_positive": fit_n,
                "feature_set": "protein_plus_sex",
            }
            for dc in disease_cols:
                rec[dc] = int(row_i[dc]) if dc in row_i.index else 0
            long_records.append(rec)

        if DEBUG:
            print(
                f"  {organ:20s} ({model_tag}) {correction_status} | "
                f"fit_N={fit_n}, apply_N={apply_n}, alpha={alpha:+.4f}, beta={beta:+.4f}"
            )

params_df = pd.DataFrame(param_rows)
params_df.to_csv(PARAMS_OUT_PATH, index=False)

out_long = pd.DataFrame(long_records)
if len(out_long) == 0:
    raise RuntimeError("No records were produced. Check cohort sizes, protein coverage, and model availability.")

print("\n" + "-" * 70)
print(f"Saving LONG single-disease disease-fit-corrected predictions:\n  {OUT_PATH_LONG}")
out_long.to_csv(OUT_PATH_LONG, index=False)

wide_bag = (
    out_long.pivot_table(
        index=["sample_id", "Disease", "age_chron"],
        columns="organ",
        values="BAG_corrected_disease_fit",
        aggfunc="first",
    )
    .reset_index()
)
non_organ_cols = {"sample_id", "Disease", "age_chron"}
wide_bag = wide_bag.rename(
    columns={c: f"{c}_BAG_corrected_disease_fit" for c in wide_bag.columns if c not in non_organ_cols}
)

merge_cols = ["sample_id", "sex", "disease_count", "is_single_disease"] + disease_cols
wide_out = wide_bag.merge(df_single_disease[merge_cols].drop_duplicates("sample_id"), on="sample_id", how="left")
ordered_cols = (
    ["sample_id", "Disease", "age_chron", "sex", "disease_count", "is_single_disease"]
    + sorted([c for c in wide_out.columns if c.endswith("_BAG_corrected_disease_fit")])
    + [c for c in disease_cols if c in wide_out.columns]
)
wide_out = wide_out[ordered_cols].copy()

print(f"Saving WIDE single-disease disease-fit-corrected predictions:\n  {OUT_PATH_WIDE}")
wide_out.to_csv(OUT_PATH_WIDE, index=False)

print(f"Saving disease-fit params:\n  {PARAMS_OUT_PATH}")
print("-" * 70)
print("Done.")
