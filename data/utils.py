from typing import List, Set, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
import random
import json
import joblib
import datetime
import os
import xgboost as xgb
import torch

def split_data(data_path: str, ratio: Tuple[float, float, float] = (.8, .1 , .1), folds: int = 1): 
    df = pd.read_csv(data_path)
    indicies = list(range(len(df)))
    random.shuffle(indicies)
    if folds == 1:
        train_idx = int(len(df) * ratio[0])
        val_idx = int((ratio[0] + ratio[1]) * len(df))
        train = df.iloc[indicies[:train_idx]]
        val = df.iloc[indicies[train_idx: val_idx]]
        test = df.iloc[indicies[val_idx:]]
        return train, val, test
    
    df_folds = np.array_split(df, folds)
    return df_folds

def save_experiment(
    model,
    fold_mses,
    params,
    model_name,
    pred,
    y_test,
    base_path="results",
    extra_artifact=None,
    eval_metrics=None
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(base_path, f"{model_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), os.path.join(exp_dir, "model.pth"))
    elif isinstance(model, xgb.XGBRegressor):
        model.save_model(os.path.join(exp_dir, "model.json"))
    else:
        joblib.dump(model, os.path.join(exp_dir, "model.joblib"))
    
    if extra_artifact is not None:
        joblib.dump(extra_artifact, os.path.join(exp_dir, "scaler.joblib"))

    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "hyperparameters": params,
        "performance": {
            "fold_mses": [float(m) for m in fold_mses], # Convert to float for JSON
            "avg_mse": float(np.mean(fold_mses)),
            "std_mse": float(np.std(fold_mses))
        }
    }

    if eval_metrics is not None:
        metadata["performance"]["test_mse"] = float(eval_metrics["mse"])
        metadata["performance"]["test_r2"] = float(eval_metrics["r2"])
        metadata["performance"]["test_pearson"] = float(eval_metrics["pearson"])
    
    with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    results_df = pd.DataFrame({
        'y_test': y_test,
        'predictions': pred,
        'residuals': y_test - pred
    })
    results_df.to_csv(os.path.join(exp_dir, "predictions.csv"), index=False)


def save_summary_table(summary_df,base_path="results"):
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(base_path, f"summary_report_{timestamp}.csv")

    summary_df.to_csv(file_path, index=False)


def save_data_splits(train_df, val_df, test_df, source_path, base_path="results/data_splits"):
    """
    Persist the exact train/val/test split for reproducibility.
    """
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    split_dir = os.path.join(base_path, f"split_{timestamp}")
    os.makedirs(split_dir, exist_ok=True)

    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")
    test_path = os.path.join(split_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "timestamp": timestamp,
        "source_data_path": source_path,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "total_rows": int(len(train_df) + len(val_df) + len(test_df))
    }
    with open(os.path.join(split_dir, "split_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved data splits to: {split_dir}")
    return split_dir


def load_data_splits(split_dir):
    """
    Load previously saved train/val/test split CSVs.
    """
    train_path = os.path.join(split_dir, "train.csv")
    val_path = os.path.join(split_dir, "val.csv")
    test_path = os.path.join(split_dir, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f"Could not find split CSV files in '{split_dir}'. "
            "Expected train.csv, val.csv, and test.csv."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    print(f"Loaded data splits from: {split_dir}")
    return train_df, val_df, test_df


def load_feature_cache(feature_cache_path):
    if not os.path.exists(feature_cache_path):
        raise FileNotFoundError(f"Feature cache not found: {feature_cache_path}")
    data = np.load(feature_cache_path)
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Feature cache is missing keys: {missing}")
    print(f"Loaded feature cache from: {feature_cache_path}")
    return {k: data[k] for k in required}
