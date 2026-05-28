#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from joblib import Parallel, delayed


BASE_SCRIPT = Path(__file__).with_name("02_model_training_sex.py")
MODEL_PARALLEL_N_JOBS = 3
GRIDSEARCH_N_JOBS_PER_MODEL = 1
MODELS_TO_TRAIN = ("lasso", "en", "svr")


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing base training script: {BASE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("model_training_sex_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_one_model(base, data: Dict[str, Any], organ: str, model_tag: str, output_dir: Path, seed: int):
    print(f"    [{organ}] starting {model_tag}")
    res = base.run_model_train_eval_save(
        data,
        model=model_tag,
        organ=organ,
        output_dir=output_dir,
        seed=seed,
        gs_n_jobs=GRIDSEARCH_N_JOBS_PER_MODEL,
    )
    print(f"    [{organ}] done {model_tag}: test_mae={res['metrics']['test_mae']:.3f}")
    return model_tag, res["metrics"]


def main():
    base = load_base_module()

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

    all_results: Dict[str, Any] = {}

    for organ in organs:
        print(f"\n=== Organ: {organ} | protein + sex ===")
        data = base.preprocess_gnpc_for_organ_age_prediction(
            df=df,
            organ=organ,
            organ_map=organ_map,
            test_size=test_size,
            seed=seed,
            return_ids=return_ids,
            miss_thresh=miss_thresh,
        )
        base.save_preprocessing_artifact(data, output_dir, organ)

        print(
            f"  Train kept: {data['n_train_after']}/{data['n_train_before']} "
            f"(dropped {data['n_train_dropped_high_missing']}) | "
            f"Test kept: {data['n_test_after']}/{data['n_test_before']} "
            f"(dropped {data['n_test_dropped_high_missing']})"
        )
        print(f"  Features: {data['n_features']} ({len(data['protein_cols'])} proteins + sex)")
        print(f"  Saved preprocessing artifact: {output_dir / organ / 'preprocessing_artifacts.json'}")

        try:
            trained = Parallel(n_jobs=MODEL_PARALLEL_N_JOBS, backend="loky")(
                delayed(train_one_model)(base, data, organ, model_tag, output_dir, seed)
                for model_tag in MODELS_TO_TRAIN
            )
            all_results[organ] = {model_tag: metrics for model_tag, metrics in trained}
            print(
                f"Done {organ} | "
                f"LASSO test_mae={all_results[organ]['lasso']['test_mae']:.3f}, "
                f"EN test_mae={all_results[organ]['en']['test_mae']:.3f}, "
                f"SVR test_mae={all_results[organ]['svr']['test_mae']:.3f}"
            )
        except Exception as e:
            print(f"!! Failed on organ={organ}: {type(e).__name__}: {e}")
            all_results[organ] = {"error": f"{type(e).__name__}: {e}"}

    print("\nFinished all organs.")


if __name__ == "__main__":
    main()
