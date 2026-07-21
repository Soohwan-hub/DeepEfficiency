import os
import subprocess
import sys
import numpy as np
import pandas as pd
import data.utils as utils


def _extract_prediction_column(pred_df):
    # Prefer explicit prediction column names if present.
    preferred = [
        "pred_log10[kcat/Km(s^-1mM^-1)]",
        "pred_log10[kcat(s^-1)]",
        "pred_log10[Km(mM)]",
        "prediction",
        "pred",
        "pred_label",
        "Pre_label",
        "kcat_km_pred",
        "kcat_km",
    ]
    for col in preferred:
        if col in pred_df.columns:
            return pred_df[col]

    ignored = {"Enzyme_id", "type", "sequence", "smiles", "Unnamed: 0"}
    numeric_cols = [c for c in pred_df.columns if c not in ignored and pd.api.types.is_numeric_dtype(pred_df[c])]
    if numeric_cols:
        return pred_df[numeric_cols[0]]

    raise ValueError(
        "Could not identify prediction column in CataPro output. "
        f"Columns were: {list(pred_df.columns)}"
    )


def _resolve_catapro_paths(catapro_dir):
    catapro_root = os.path.abspath(os.path.expanduser(catapro_dir))
    root_predict = os.path.join(catapro_root, "predict.py")
    inference_predict = os.path.join(catapro_root, "inference", "predict.py")

    if os.path.isfile(root_predict):
        predict_script = root_predict
    elif os.path.isfile(inference_predict):
        predict_script = inference_predict
    else:
        raise FileNotFoundError(
            "Could not find CataPro inference entrypoint. Expected one of: "
            f"{root_predict} or {inference_predict}."
        )

    model_dir = os.path.join(catapro_root, "models")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Could not find models directory at {model_dir}. "
            "Download or place CataPro models there before inference."
        )

    sample_dir = os.path.join(catapro_root, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    return catapro_root, predict_script, model_dir, sample_dir


def _mse(y_true, y_pred):
    diff = y_true - y_pred
    return float(np.mean(diff * diff))


def _r2(y_true, y_pred):
    residual = y_true - y_pred
    ss_res = float(np.sum(residual * residual))
    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(centered * centered))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _pearson(y_true, y_pred):
    y_true_centered = y_true - float(np.mean(y_true))
    y_pred_centered = y_pred - float(np.mean(y_pred))
    denom = float(np.linalg.norm(y_true_centered) * np.linalg.norm(y_pred_centered))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(y_true_centered, y_pred_centered) / denom)


def CataPro(
    split_dir=None,
    catapro_dir="CataPro",
    batch_size=64,
    device="cuda:0",
    out_fname="catapro_prediction.csv",
    pred_save_name="CataPro_predicted_label.csv",
    metrics_save_name="CataPro_metrics.csv"
):
    if split_dir is None:
        raise ValueError("split_dir is required.")
    catapro_root, predict_script, model_dir, sample_dir = _resolve_catapro_paths(catapro_dir)

    train, val, test = utils.load_data_splits(split_dir)
    df = pd.DataFrame({
        "Enzyme_id": test["Protein_ID"],
        "type": test["EnzymeType"],
        "sequence": test["Sequence"],
        "smiles": test["SMILES"]
    })
    del train, val

    input_path = os.path.join(sample_dir, "sample_inp.csv")
    # CataPro's inference code reads CSV with index_col=0, so keep a dedicated
    # index column to avoid shifting "Enzyme_id" out of the dataframe columns.
    df.to_csv(input_path, index=True)
    output_path = os.path.join(catapro_root, out_fname)

    cmd = [
        sys.executable, predict_script,
        "-inp_fpath", input_path,
        "-model_dpath", model_dir,
        "-batch_size", str(batch_size),
        "-device", device,
        "-out_fpath", output_path
    ]

    subprocess.run(cmd, cwd=os.path.dirname(predict_script), check=True)
    pred_df = pd.read_csv(output_path)
    pred_values = _extract_prediction_column(pred_df).to_numpy()
    y_true = test["Log10_value"].to_numpy()

    if len(pred_values) != len(y_true):
        raise ValueError(
            f"Prediction row count ({len(pred_values)}) does not match test row count ({len(y_true)})."
        )

    mse = _mse(y_true, pred_values)
    r2 = _r2(y_true, pred_values)
    pearson = _pearson(y_true, pred_values)

    pred_out = pd.DataFrame({
        "Sequence": test["Sequence"].to_numpy(),
        "SMILES": test["SMILES"].to_numpy(),
        "y_test": y_true,
        "predictions": pred_values
    })
    pred_out_path = os.path.join(catapro_root, pred_save_name)
    pred_out.to_csv(pred_out_path, index=False)

    metrics = pd.DataFrame([{
        "model": "CataPro",
        "mse": mse,
        "r2": r2,
        "pearson": pearson
    }])
    metrics_out_path = os.path.join(catapro_root, metrics_save_name)
    metrics.to_csv(metrics_out_path, index=False)

    print(f"CataPro prediction saved to: {output_path}")
    print(f"Saved aligned predictions to: {pred_out_path}")
    print(f"Saved metrics to: {metrics_out_path}")
    return output_path, pred_out_path, metrics_out_path

if __name__ == "__main__":
    CataPro()