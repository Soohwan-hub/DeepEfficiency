import json
import argparse
import torch
import joblib
import datetime
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import data.utils as utils
import data.embedding as embedding
import architecture.FFNN as ffnn_m

from scipy.stats import pearsonr
#from sklearn.linear_model import Ridge as CPU_Ridge
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
from cuml.ensemble import RandomForestRegressor
from cuml.linear_model import Ridge
from sklearn.model_selection import ParameterSampler
import xgboost as xgb


def tune_train_XGBoost_from_encoded_folds(encoded_folds):
    param_grid = {
        'max_depth': [4, 6, 8, 10, 12],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0]
    }
    param_list = list(ParameterSampler(param_grid, n_iter=20, random_state=42))

    best_overall_mse = float('inf')
    best_overall_params = None
    best_overall_trees = 0
    best_fold_mses = []

    for _, params in enumerate(param_list):
        cv_mses = []
        cv_trees = []

        for i in range(len(encoded_folds)):
            X_val, y_val = encoded_folds[i]

            train_X_folds = [encoded_folds[j][0] for j in range(len(encoded_folds)) if j != i]
            train_y_folds = [encoded_folds[j][1] for j in range(len(encoded_folds)) if j != i]

            X_train = np.concatenate(train_X_folds, axis=0)
            y_train = np.concatenate(train_y_folds, axis=0)

            model = xgb.XGBRegressor(
                **params,
                n_estimators=2000,
                tree_method="hist",
                device="cuda",
                random_state=42,
                early_stopping_rounds=50
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )

            best_trees = model.best_iteration
            pred = model.predict(X_val)
            mse = mean_squared_error(y_val, pred)

            cv_trees.append(best_trees)
            cv_mses.append(mse)

        avg_mse = np.mean(cv_mses)
        avg_trees = int(np.round(np.mean(cv_trees)))

        if avg_mse < best_overall_mse:
            best_overall_mse = avg_mse
            best_overall_params = params
            best_overall_trees = avg_trees
            best_fold_mses = cv_mses

    X_master_train = np.concatenate([f[0] for f in encoded_folds], axis=0)
    y_master_train = np.concatenate([f[1] for f in encoded_folds], axis=0)

    master_model = xgb.XGBRegressor(
        **best_overall_params,
        n_estimators=best_overall_trees,
        tree_method="hist",
        device="cuda",
        random_state=42
    )
    master_model.fit(X_master_train, y_master_train)
    return master_model, best_fold_mses, best_overall_params


def tune_train_XGBoost(data_folds):
    encoded_folds = []
    for _, fold in enumerate(data_folds):
        X = embedding.concat_encoder(fold).astype(np.float32)
        y = fold["Log10_value"].to_numpy(dtype=np.float32)
        encoded_folds.append((X, y))
    return tune_train_XGBoost_from_encoded_folds(encoded_folds)


def tune_train_XGBoost_cached(X_trainval, y_trainval, n_folds=5):
    indices = np.arange(len(X_trainval))
    fold_indices = np.array_split(indices, n_folds)
    encoded_folds = []
    for fold_idx in fold_indices:
        encoded_folds.append((X_trainval[fold_idx], y_trainval[fold_idx]))
    return tune_train_XGBoost_from_encoded_folds(encoded_folds)


def tune_train_ExtraTrees(train_data):
    X_train = embedding.concat_encoder(train_data)
    y_train = train_data["Log10_value"].to_numpy(dtype=float)
    
    param_dist = {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [10, 15, 20, 30, None],
        'max_features': ['sqrt', 'log2', 1.0],
        'min_samples_split': [2, 5, 10]
    }

    base_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        refit=True
    )
    search.fit(X_train, y_train)

    results = search.cv_results_
    best_index = search.best_index_
    fold_mses = [
        -1 * results[f'split{i}_test_score'][best_index] for i in range(5)
    ]
    for i, mse in enumerate(fold_mses):
        print(f"Fold {i+1} MSE: {mse:.4f}")    

    print(f"Average CV MSE: {np.mean(fold_mses):.4f}")
    print(f"CV MSE Standard Deviation: {np.std(fold_mses):.4f}\n")

    return search.best_estimator_, fold_mses, search.best_params_


    # models = []
    # for i in range(len(data_folds)):
    #     model = ExtraTreesRegressor(
    #         n_estimators=500,
    #         max_depth=15,
    #         max_features='sqrt',
    #         n_jobs=-1,
    #         random_state=42
    #     )
    #     train = pd.concat([f for j, f in enumerate(data_folds) if j != i], ignore_index=True)
    #     test = data_folds[i]
    #     X_train = embedding.concat_encoder(train)
    #     y_train = train["Log10_value"].to_numpy(dtype=float)
    #     X_test = embedding.concat_encoder(test)
    #     y_test = test["Log10_value"].to_numpy(dtype=float)
    #     model.fit(X_train, y_train)

    #     pred = model.predict(X_test)
    #     mse = mean_squared_error(y_test, pred)
    #     r2 = r2_score(y_test, pred)
    #     models.append((model, mse, r2))
    # return models


def tune_train_RandomForest(train_data):
    X_train = embedding.concat_encoder(train_data).astype(np.float32)
    y_train = train_data["Log10_value"].to_numpy(dtype=np.float32)

    param_dist = {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [10, 15, 20, 30],
        'max_features': ['sqrt', 'log2', 1.0],
        'min_samples_split': [2, 5, 10]
    }

    base_model = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring='neg_mean_squared_error', 
        random_state=42,
        refit=True
    )

    search.fit(X_train, y_train)
    results = search.cv_results_
    best_index = search.best_index_
    
    fold_mses = [
        -1 * results[f'split{i}_test_score'][best_index] for i in range(5)
    ]
    
    for i, mse in enumerate(fold_mses):
        print(f"Fold {i+1} MSE: {mse:.4f}")
        
    print(f"Average CV MSE: {np.mean(fold_mses):.4f}")
    print(f"CV MSE Standard Deviation: {np.std(fold_mses):.4f}\n")
    
    return search.best_estimator_, fold_mses, search.best_params_

# def train_RandomForest(data_folds):
#     models = []
#     for i in range(len(data_folds)):
#         model = RandomForestRegressor(
#             n_estimators=500,
#             max_depth=15,
#             max_features='sqrt',
#             n_jobs=-1,
#             random_state=42
#         )

#         train = pd.concat([f for j, f in enumerate(data_folds) if j != i], ignore_index=True)
#         test = data_folds[i]
#         X_train = embedding.concat_encoder(train)
#         y_train = train["Log10_value"].to_numpy(dtype=float)
#         X_test = embedding.concat_encoder(test)
#         y_test = test["Log10_value"].to_numpy(dtype=float)

#         model.fit(X_train, y_train)
#         pred = model.predict(X_test)
#         mse = mean_squared_error(y_test, pred)
#         r2 = r2_score(y_test, pred)
#         models.append((model, mse, r2))

#     return models



def tune_train_Ridge(train_data):
    X_train = embedding.concat_encoder(train_data).astype(np.float32)
    y_train = train_data["Log10_value"].to_numpy(dtype=np.float32)

    param_dist = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'solver': ['eig', 'svd'] 
    }

    base_model = Ridge()

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=14,
        cv=5,
        scoring='neg_mean_squared_error', 
        random_state=42,
        refit=True
    )

    search.fit(X_train, y_train)
    results = search.cv_results_
    best_index = search.best_index_
    
    fold_mses = [
        -1 * results[f'split{i}_test_score'][best_index] for i in range(5)
    ]
    return search.best_estimator_, fold_mses, search.best_params_


def tune_train_ExtraTrees_cached(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [10, 15, 20, 30, None],
        'max_features': ['sqrt', 'log2', 1.0],
        'min_samples_split': [2, 5, 10]
    }

    base_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        refit=True
    )
    search.fit(X_train, y_train)

    results = search.cv_results_
    best_index = search.best_index_
    fold_mses = [
        -1 * results[f'split{i}_test_score'][best_index] for i in range(5)
    ]
    return search.best_estimator_, fold_mses, search.best_params_


def tune_train_RandomForest_cached(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 300, 500, 800],
        'max_depth': [10, 15, 20, 30],
        'max_features': ['sqrt', 'log2', 1.0],
        'min_samples_split': [2, 5, 10]
    }

    base_model = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        refit=True
    )
    search.fit(X_train, y_train)
    results = search.cv_results_
    best_index = search.best_index_

    fold_mses = [
        -1 * results[f'split{i}_test_score'][best_index] for i in range(5)
    ]
    return search.best_estimator_, fold_mses, search.best_params_


def tune_train_Ridge_cached(X_train, y_train):
    param_dist = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'solver': ['eig', 'svd']
    }

    base_model = Ridge()

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=14,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        refit=True
    )
    search.fit(X_train, y_train)
    results = search.cv_results_
    best_index = search.best_index_

    fold_mses = [
        -1 * results[f'split{i}_test_score'][best_index] for i in range(5)
    ]
    return search.best_estimator_, fold_mses, search.best_params_

# def train_Ridge(data_folds):
#     models = []
#     for i in range(len(data_folds)):
#         model = Ridge(alpha=1.0, random_state=42)

#         train = pd.concat([f for j, f in enumerate(data_folds) if j != i], ignore_index=True)
#         test = data_folds[i]
#         X_train = embedding.concat_encoder(train)
#         y_train = train["Log10_value"].to_numpy(dtype=float)
#         X_test = embedding.concat_encoder(test)
#         y_test = test["Log10_value"].to_numpy(dtype=float)

#         model.fit(X_train, y_train)
#         pred = model.predict(X_test)
#         mse = mean_squared_error(y_test, pred)
#         r2 = r2_score(y_test, pred)
#         models.append((model, mse, r2))

#     return models

def evaluate_model(frozen_model, test_set=None, scaler=None, X_test=None, y_test=None):
    if X_test is None or y_test is None:
        X_test = embedding.concat_encoder(test_set)
        y_test = test_set["Log10_value"].to_numpy(dtype=float)

    if isinstance(frozen_model, torch.nn.Module):
        frozen_model.eval()
        X_test_scaled = scaler.transform(X_test) if scaler else X_test
        X_test_T = torch.tensor(X_test_scaled, dtype=torch.float32).to(next(frozen_model.parameters()).device)
        with torch.no_grad():
            pred = frozen_model(X_test_T).cpu().numpy().flatten()
    else:
        pred = frozen_model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    pearson_corr, _ = pearsonr(y_test, pred)
        
    print(f"  -> MSE: {mse:.4f}")
    print(f"  -> R2 Score: {r2:.4f}")
    print(f"  -> Pearson r: {pearson_corr:.4f}\n")
    return mse, r2, pearson_corr, pred, y_test

def build_pipelines(train, val, df_folds, selected_models=None, feature_cache_data=None):
    """
    Build model pipeline list, optionally filtered by model names.
    """
    train_ = pd.concat([train, val], ignore_index=True)
    if feature_cache_data is None:
        all_pipelines = [
            ("XGBoost", tune_train_XGBoost, df_folds),
            ("ExtraTrees", tune_train_ExtraTrees, train_),
            ("RandomForest", tune_train_RandomForest, train_),
            ("Ridge", tune_train_Ridge, train_),
            ("FFNN", ffnn_m.tune_train_FFNN, (train, val))
        ]
    else:
        X_train = feature_cache_data["X_train"]
        y_train = feature_cache_data["y_train"]
        X_val = feature_cache_data["X_val"]
        y_val = feature_cache_data["y_val"]
        X_trainval = np.concatenate([X_train, X_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)

        all_pipelines = [
            ("XGBoost", tune_train_XGBoost_cached, (X_trainval, y_trainval)),
            ("ExtraTrees", tune_train_ExtraTrees_cached, (X_trainval, y_trainval)),
            ("RandomForest", tune_train_RandomForest_cached, (X_trainval, y_trainval)),
            ("Ridge", tune_train_Ridge_cached, (X_trainval, y_trainval)),
            ("FFNN", ffnn_m.tune_train_FFNN, (train, val))
        ]

    if not selected_models:
        return all_pipelines

    requested = {name.lower(): name for name in selected_models}
    available = {name.lower(): (name, func, data) for name, func, data in all_pipelines}
    unknown = sorted(set(requested.keys()) - set(available.keys()))
    if unknown:
        raise ValueError(
            "Unknown model(s): "
            + ", ".join(unknown)
            + ". Valid options are: XGBoost, ExtraTrees, RandomForest, Ridge, FFNN."
        )

    filtered = []
    for model_name in selected_models:
        key = model_name.lower()
        filtered.append(available[key])
    return filtered

def main(
    split_dir=None,
    save_splits=True,
    selected_models=None,
    feature_cache_path=None,
    results_root="results"
):
    data_path = "data/data_KCATKM.csv"
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(results_root, f"run_{run_timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Saving run artifacts to: {run_output_dir}")

    if split_dir:
        train, val, test = utils.load_data_splits(split_dir)
    else:
        train, val, test = utils.split_data(data_path=data_path, ratio=[.8,.1,.1])
        if save_splits:
            utils.save_data_splits(train, val, test, source_path=data_path)

    train_ = pd.concat([train, val], ignore_index=True)
    df_folds = np.array_split(train_, 5)
    feature_cache_data = None
    if feature_cache_path is not None:
        feature_cache_data = utils.load_feature_cache(feature_cache_path)

    pipelines = build_pipelines(
        train,
        val,
        df_folds,
        selected_models=selected_models,
        feature_cache_data=feature_cache_data
    )
    print(f"Running models: {', '.join(name for name, _, _ in pipelines)}")

    results_sum = []
    for name, func, data in pipelines:
        if isinstance(data, tuple):
            results = func(*data)
        else:
            results = func(data)

        if name == "FFNN":
            model, fold_mses, params, scaler = results
            mse, r2, pearson, pred, y_test = evaluate_model(model, test, scaler=scaler)
            utils.save_experiment(
                model,
                fold_mses,
                params,
                name,
                pred,
                y_test,
                base_path=run_output_dir,
                extra_artifact=scaler,
                eval_metrics={"mse": mse, "r2": r2, "pearson": pearson}
            )
        else:
            model, fold_mses, params = results
            if feature_cache_data is None:
                mse, r2, pearson, pred, y_test = evaluate_model(model, test, scaler=None)
            else:
                mse, r2, pearson, pred, y_test = evaluate_model(
                    model,
                    X_test=feature_cache_data["X_test"],
                    y_test=feature_cache_data["y_test"],
                    scaler=None
                )
            utils.save_experiment(
                model,
                fold_mses,
                params,
                name,
                pred,
                y_test,
                base_path=run_output_dir,
                extra_artifact=None,
                eval_metrics={"mse": mse, "r2": r2, "pearson": pearson}
            )
        
        
        results_sum.append({
            "model": name,
            "avg_cv_mse": np.mean(fold_mses),
            "vault_mse": mse,
            "test_r2": r2,
            "test_pearson": pearson
        })
    
    summary_df = pd.DataFrame(results_sum)
    print(summary_df.to_string(index=False))
    utils.save_summary_table(summary_df, base_path=run_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate enzyme efficiency models.")
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Path to a saved split directory containing train.csv, val.csv, and test.csv."
    )
    parser.add_argument(
        "--no-save-splits",
        action="store_true",
        help="Do not save newly generated train/val/test splits."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model(s) to run. Options: XGBoost ExtraTrees RandomForest Ridge FFNN"
    )
    parser.add_argument(
        "--feature-cache",
        type=str,
        default=None,
        help="Path to precomputed feature cache (.npz)."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory where a timestamped run folder will be created."
    )
    args = parser.parse_args()
    main(
        split_dir=args.split_dir,
        save_splits=not args.no_save_splits,
        selected_models=args.models,
        feature_cache_path=args.feature_cache,
        results_root=args.results_root
    )
