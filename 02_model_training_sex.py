from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


SEX_COL = "sex"


def preprocess_gnpc_for_organ_age_prediction(
    df: pd.DataFrame,
    organ: str,
    organ_map: Dict[str, List[str]],
    *,
    id_col: str = "sample_id",
    target_col: str = "age",
    sex_col: str = SEX_COL,
    test_size: float = 0.20,
    seed: int = 42,
    return_ids: bool = False,
    miss_thresh: float = 0.30,
    median_fallback: float = 0.0,
) -> Dict[str, Any]:
    """
    Protein + sex organ-age preprocessing.

    Missingness filtering is based on organ proteins only. Remaining missing
    protein values are imputed from CN train medians. Sex is appended as a
    numeric covariate after restricting to valid sex values.
    """
    if organ not in organ_map:
        raise KeyError(f"Organ '{organ}' not found in mapping.")

    protein_cols = list(organ_map[organ])
    covariate_cols = [sex_col]
    feature_cols = protein_cols + covariate_cols

    needed_cols = [id_col, target_col] + protein_cols + covariate_cols
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns in df: {missing[:20]}" + (" ..." if len(missing) > 20 else "")
        )

    df_sub = df[needed_cols].copy()
    df_sub = df_sub[df_sub[target_col].notna()].copy()
    df_sub = df_sub[df_sub[sex_col].isin([1, 2])].copy()

    idx = np.arange(len(df_sub))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=seed)

    train_df = df_sub.iloc[idx_train].copy()
    test_df = df_sub.iloc[idx_test].copy()

    def add_missing_rate(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["_organ_missing_rate"] = d[protein_cols].isna().mean(axis=1)
        return d

    train_df = add_missing_rate(train_df)
    test_df = add_missing_rate(test_df)

    n_train_before = len(train_df)
    n_test_before = len(test_df)

    train_df = train_df.loc[train_df["_organ_missing_rate"] <= miss_thresh].copy()
    test_df = test_df.loc[test_df["_organ_missing_rate"] <= miss_thresh].copy()

    n_train_after = len(train_df)
    n_test_after = len(test_df)

    protein_medians = train_df[protein_cols].median(axis=0, skipna=True).fillna(median_fallback)
    train_df[protein_cols] = train_df[protein_cols].fillna(protein_medians)
    test_df[protein_cols] = test_df[protein_cols].fillna(protein_medians)

    train_df[sex_col] = pd.to_numeric(train_df[sex_col], errors="coerce")
    test_df[sex_col] = pd.to_numeric(test_df[sex_col], errors="coerce")

    feature_medians = train_df[feature_cols].median(axis=0, skipna=True).fillna(median_fallback)
    train_df[feature_cols] = train_df[feature_cols].fillna(feature_medians)
    test_df[feature_cols] = test_df[feature_cols].fillna(feature_medians)

    out: Dict[str, Any] = {
        "X_train": train_df[feature_cols].to_numpy(dtype=np.float32),
        "X_test": test_df[feature_cols].to_numpy(dtype=np.float32),
        "y_train": train_df[target_col].to_numpy(),
        "y_test": test_df[target_col].to_numpy(),
        "organ": organ,
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "protein_cols": protein_cols,
        "covariate_cols": covariate_cols,
        "seed": seed,
        "test_size": test_size,
        "miss_thresh": float(miss_thresh),
        "impute": "protein medians from train; sex appended as numeric covariate",
        "median_fallback": float(median_fallback),
        "train_medians": {k: float(v) for k, v in protein_medians.items()},
        "feature_medians": {k: float(v) for k, v in feature_medians.items()},
        "n_train_before": int(n_train_before),
        "n_train_after": int(n_train_after),
        "n_train_dropped_high_missing": int(n_train_before - n_train_after),
        "n_test_before": int(n_test_before),
        "n_test_after": int(n_test_after),
        "n_test_dropped_high_missing": int(n_test_before - n_test_after),
    }

    if return_ids:
        out["id_train"] = train_df[id_col].astype(str).to_numpy()
        out["id_test"] = test_df[id_col].astype(str).to_numpy()

    return out


def save_preprocessing_artifact(data: Dict[str, Any], output_dir: Union[str, Path], organ: str) -> None:
    artifact_dir = Path(output_dir) / organ
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "organ": organ,
        "feature_cols": list(data["feature_cols"]),
        "protein_cols": list(data["protein_cols"]),
        "covariate_cols": list(data["covariate_cols"]),
        "n_features": int(data["n_features"]),
        "miss_thresh": float(data["miss_thresh"]),
        "impute": data["impute"],
        "median_fallback": float(data["median_fallback"]),
        "train_medians": data["train_medians"],
        "feature_medians": data["feature_medians"],
        "seed": int(data["seed"]),
        "test_size": float(data["test_size"]),
        "n_train_before": int(data["n_train_before"]),
        "n_train_after": int(data["n_train_after"]),
        "n_train_dropped_high_missing": int(data["n_train_dropped_high_missing"]),
        "n_test_before": int(data["n_test_before"]),
        "n_test_after": int(data["n_test_after"]),
        "n_test_dropped_high_missing": int(data["n_test_dropped_high_missing"]),
    }

    if "id_train" in data:
        artifact["id_train"] = [str(x) for x in data["id_train"]]
    if "id_test" in data:
        artifact["id_test"] = [str(x) for x in data["id_test"]]

    with open(artifact_dir / "preprocessing_artifacts.json", "w") as f:
        json.dump(artifact, f, indent=2)

    pd.Series(data["train_medians"], name="train_median").to_csv(
        artifact_dir / "train_medians.csv",
        header=True,
    )
    pd.Series(data["feature_medians"], name="feature_median").to_csv(
        artifact_dir / "feature_medians.csv",
        header=True,
    )


def run_model_train_eval_save(
    data: Union[Dict[str, Any], Tuple[Any, Any, Any, Any]],
    model: str,
    organ: str,
    output_dir: Union[str, Path],
    *,
    seed: int = 42,
    outer_splits: int = 5,
    inner_splits: int = 5,
    scoring_for_tuning: str = "neg_mean_absolute_error",
    gs_n_jobs: int = 2,
) -> Dict[str, Any]:
    if isinstance(data, dict):
        X_train, X_test, y_train, y_test = (
            data["X_train"], data["X_test"], data["y_train"], data["y_test"]
        )
    else:
        X_train, X_test, y_train, y_test = data

    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    model_key = model.lower().strip()
    if model_key == "lasso":
        model_tag = "lasso"
    elif model_key in {"en", "elasticnet", "elastic_net", "elastic-net"}:
        model_tag = "en"
    else:
        model_tag = "svr"

    out_dir = Path(output_dir) / organ / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = [("scaler", StandardScaler())]
    if model_tag == "lasso":
        estimator = Lasso(max_iter=100_000, random_state=seed)
        param_grid = {"model__alpha": np.logspace(-4, 2, 60)}
    elif model_tag == "en":
        estimator = ElasticNet(max_iter=200_000, random_state=seed)
        param_grid = {
            "model__alpha": np.logspace(-4, 2, 40),
            "model__l1_ratio": np.linspace(0.05, 0.95, 19),
        }
    else:
        estimator = LinearSVR(random_state=seed, max_iter=1_000_000, tol=1e-3)
        param_grid = {
            "model__C": np.logspace(-3, 3, 15),
            "model__epsilon": [0.0, 0.05, 0.1, 0.2],
        }
    steps.append(("model", estimator))

    pipe = Pipeline(steps=steps)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring_for_tuning,
        cv=inner_cv,
        n_jobs=gs_n_jobs,
        refit=True,
        verbose=1,
    )

    nested = cross_validate(
        estimator=gs,
        X=X_train,
        y=y_train,
        cv=outer_cv,
        scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
        n_jobs=1,
        return_estimator=False,
    )

    nested_mae = -nested["test_mae"]
    nested_rmse = -nested["test_rmse"]

    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    best_params = gs.best_params_

    yhat_train = best_model.predict(X_train)
    yhat_test = best_model.predict(X_test)

    metrics = {
        "nested_cv_mae_mean": float(np.mean(nested_mae)),
        "nested_cv_mae_sd": float(np.std(nested_mae, ddof=1)) if len(nested_mae) > 1 else float("nan"),
        "nested_cv_rmse_mean": float(np.mean(nested_rmse)),
        "nested_cv_rmse_sd": float(np.std(nested_rmse, ddof=1)) if len(nested_rmse) > 1 else float("nan"),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "seed": seed,
        "tuning_scoring": scoring_for_tuning,
        "train_mae": float(mean_absolute_error(y_train, yhat_train)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, yhat_train))),
        "test_mae": float(mean_absolute_error(y_test, yhat_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, yhat_test))),
        "feature_set": "protein_plus_sex",
    }

    joblib.dump(gs, out_dir / "gridsearchcv.joblib")
    joblib.dump(best_model, out_dir / "best_model.joblib")

    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame({"y_true": y_train, "y_pred": yhat_train}).to_csv(out_dir / "pred_train.csv", index=False)
    pd.DataFrame({"y_true": y_test, "y_pred": yhat_test}).to_csv(out_dir / "pred_test.csv", index=False)

    return {
        "out_dir": str(out_dir),
        "best_model": best_model,
        "best_params": best_params,
        "metrics": metrics,
        "nested_mae": nested_mae,
        "nested_rmse": nested_rmse,
    }


def main():
    input_dir = Path("/home/jovyan/work/GNPC Harmonised Dataset v1/Clinical/processed_data_v4_python")
    seed = 42
    test_size = 0.20
    return_ids = True
    miss_thresh = 0.30
    output_dir = Path(f"results_imputation_miss{int(miss_thresh * 100)}_sex")

    with open(input_dir / "organ_system_mapping.json", "r") as f:
        organ_map: Dict[str, List[str]] = json.load(f)

    organs = list(organ_map.keys())
    print(f"Found {len(organs)} organ systems in mapping:")
    print(organs)

    print("\nLoading gnpc_processed.pkl once...")
    df = pd.read_pickle(input_dir / "gnpc_processed.pkl")
    print(f"Loaded df shape: {df.shape}")

    if SEX_COL not in df.columns:
        raise KeyError(f"CN dataframe missing required sex covariate column: {SEX_COL}")

    for organ in organs:
        print(f"\n=== Organ: {organ} | protein + sex ===")
        data = preprocess_gnpc_for_organ_age_prediction(
            df=df,
            organ=organ,
            organ_map=organ_map,
            test_size=test_size,
            seed=seed,
            return_ids=return_ids,
            miss_thresh=miss_thresh,
        )
        save_preprocessing_artifact(data, output_dir, organ)

        print(
            f"  Train kept: {data['n_train_after']}/{data['n_train_before']} "
            f"(dropped {data['n_train_dropped_high_missing']}) | "
            f"Test kept: {data['n_test_after']}/{data['n_test_before']} "
            f"(dropped {data['n_test_dropped_high_missing']})"
        )
        print(f"  Features: {data['n_features']} ({len(data['protein_cols'])} proteins + sex)")
        print(f"  Saved preprocessing artifact: {output_dir / organ / 'preprocessing_artifacts.json'}")

        try:
            res_lasso = run_model_train_eval_save(data, model="lasso", organ=organ, output_dir=output_dir, seed=seed)
            res_en = run_model_train_eval_save(data, model="en", organ=organ, output_dir=output_dir, seed=seed)
            res_svr = run_model_train_eval_save(data, model="svr", organ=organ, output_dir=output_dir, seed=seed)
            print(
                f"Done {organ} | "
                f"LASSO test_mae={res_lasso['metrics']['test_mae']:.3f}, "
                f"EN test_mae={res_en['metrics']['test_mae']:.3f}, "
                f"SVR test_mae={res_svr['metrics']['test_mae']:.3f}"
            )
        except Exception as e:
            print(f"!! Failed on organ={organ}: {type(e).__name__}: {e}")

    print("\nFinished all organs.")


if __name__ == "__main__":
    main()
