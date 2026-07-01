import argparse
import os
import numpy as np
import pandas as pd

import data.embedding as embedding


def load_splits(split_dir):
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
    return train_df, val_df, test_df


def build_xy(df, enzyme_batch_size, substrate_batch_size):
    X = embedding.concat_encoder(
        df,
        enzyme_batch_size=enzyme_batch_size,
        substrate_batch_size=substrate_batch_size
    ).astype(np.float32)
    y = df["Log10_value"].to_numpy(dtype=np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Precompute and cache embeddings/features for saved data splits.")
    parser.add_argument(
        "--split-dir",
        required=True,
        help="Directory containing train.csv, val.csv, and test.csv."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz file path. Default: <split-dir>/feature_cache/features.npz"
    )
    parser.add_argument(
        "--enzyme-batch-size",
        type=int,
        default=2,
        help="Batch size used for enzyme encoder."
    )
    parser.add_argument(
        "--substrate-batch-size",
        type=int,
        default=64,
        help="Batch size used for substrate encoder."
    )
    args = parser.parse_args()

    train_df, val_df, test_df = load_splits(args.split_dir)

    X_train, y_train = build_xy(train_df, args.enzyme_batch_size, args.substrate_batch_size)
    X_val, y_val = build_xy(val_df, args.enzyme_batch_size, args.substrate_batch_size)
    X_test, y_test = build_xy(test_df, args.enzyme_batch_size, args.substrate_batch_size)

    output_path = args.output
    if output_path is None:
        output_dir = os.path.join(args.split_dir, "feature_cache")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "features.npz")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    print(f"Saved feature cache to: {output_path}")


if __name__ == "__main__":
    main()
