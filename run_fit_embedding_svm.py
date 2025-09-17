import argparse
import numpy as np
import h5py
from pathlib import Path
import pickle

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


TIME_AVG_EMB_PATH = "/om2/user/imgriff/projects/torch_2_aud_attn/acts_for_RDM_analysis/word_task_v10_main_feature_gain_config/word_task_v10_main_feature_gain_config_model_activations_0dB_time_avg.h5"
FULL_TIME_EMB_PATH = "/om2/user/imgriff/projects/torch_2_aud_attn/binaural_unit_activations/word_task_v10_main_feature_gain_config/word_task_v10_main_feature_gain_config_model_activations_0dB.h5"

LAYERS = [
    "norm_coch_rep",
    "conv_block_0_relu",
    "conv_block_1_relu",
    "conv_block_2_relu",
    "conv_block_3_relu",
    "conv_block_4_relu",
    "conv_block_5_relu",
    "conv_block_6_relu",
    "relufc",
]

SIGNALS = [
    "cue",
    "target",
    "same_sex_sig",
    "diff_sex_sig",
    "nat_scene_sig",
    "mixture_same_sex",
    "mixture_diff_sex",
    "mixture_nat_scene",
]

ATTN_TYPES = ["single_source", "cued"]


def _walk_keys(h5):
    keys = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append(name)

    h5.visititems(visitor)
    return keys


def find_embeddings_dataset(h5, layer, signal, attn):
    candidates = _walk_keys(h5)
    # Prefer exact containment of all three identifiers in path
    ranked = [k for k in candidates if layer in k and signal in k and attn in k]
    if not ranked:
        # Try any two of the three
        two_match = [
            k for k in candidates if sum(s in k for s in (layer, signal, attn)) >= 2
        ]
        ranked = two_match
    if not ranked and candidates:
        # Fallback: any dataset under layer
        ranked = [k for k in candidates if layer in k] or candidates
    for key in ranked:
        ds = h5[key]
        if isinstance(ds, h5py.Dataset) and np.issubdtype(ds.dtype, np.number):
            return key
    raise RuntimeError(
        "Could not locate numeric embeddings dataset matching identifiers."
    )


def load_embeddings(h5, layer, signal, attn):
    key = find_embeddings_dataset(h5, layer, signal, attn)
    X = np.array(h5[key])
    # If time dimension present, average over it to produce fixed-size vectors
    if X.ndim > 2:
        # heuristics: average over axis with largest size if 3D and one axis equals feature dim
        time_axis = 1 if X.shape[1] >= X.shape[2] else 0
        X = X.mean(axis=time_axis)
    if X.ndim == 1:
        X = X[:, None]
    return X, key


def load_labels(h5, target, num_f0_bins=8):
    def get_any(keys):
        for k in keys:
            if k in h5:
                return np.array(h5[k])
        # search nested
        for path in _walk_keys(h5):
            if any(k in path for k in keys):
                return np.array(h5[path])
        return None

    if target == "location":
        loc = get_any(["target_loc"])
        if loc is None:
            raise RuntimeError("No location labels found for location.")
        loc = np.asarray(loc)
        # Map unique (azimuth, elevation) tuples or single-value locations to integer classes
        if loc.ndim >= 2 and loc.shape[-1] == 2:
            loc_2d = loc.reshape(-1, 2)
            _, y = np.unique(loc_2d, axis=0, return_inverse=True)
        else:
            _, y = np.unique(loc, return_inverse=True)
        return y.astype(int)

    if target == "f0_bin":
        f0 = get_any(["target_f0"])
        if f0 is None:
            raise RuntimeError("No f0 labels found for f0_bin.")
        f0 = np.asarray(f0).astype(float)
        # Bin into equal-count bins (quantile-based binning)
        quantiles = np.linspace(0, 1, num_f0_bins + 1)
        edges = np.quantile(f0, quantiles)
        # np.digitize returns 1..len(edges)-1; shift to 0-based
        y = np.digitize(f0, edges[1:-1], right=True).astype(int)
        return y

    if target == "word_class":
        word = get_any(["target_word_int"])
        if word is None:
            raise RuntimeError("No word class labels found.")
        word = np.array(word)
        # Ensure integer classes
        if not np.issubdtype(word.dtype, np.integer):
            # Try to map to categorical codes
            _, y = np.unique(word, return_inverse=True)
            return y.astype(int)
        return word.astype(int)

    raise ValueError(f"Unknown target: {target}")


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def cross_validate_svm(X_train, y_train, cv=5, use_scaling=True):
    # Use stratified CV to ensure balanced folds
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Create pipeline with optional scaling
    if use_scaling:
        # StandardScaler helps with SVM convergence and can prevent overfitting
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "estimator",
                    OneVsRestClassifier(
                        LinearSVC(
                            max_iter=5000,  # Increased iterations for convergence
                            dual=True,
                            random_state=0,
                            tol=1e-4,  # Tighter tolerance for better convergence
                        )
                    ),
                ),
            ]
        )
        # Parameter grid for pipeline with scaling
        param_grid = {
            "estimator__estimator__C": [
                0.01,
                0.1,
                1,
                10,
                100,
            ],  # More C values including stronger regularization
            "estimator__estimator__class_weight": [
                None,
                "balanced",
            ],  # Handle class imbalance
        }
    else:
        pipeline = OneVsRestClassifier(
            LinearSVC(max_iter=5000, dual=True, random_state=0, tol=1e-4)
        )
        # Parameter grid for direct OneVsRestClassifier
        param_grid = {
            "estimator__C": [
                0.01,
                0.1,
                1,
                10,
                100,
            ],  # More C values including stronger regularization
            "estimator__class_weight": [None, "balanced"],  # Handle class imbalance
        }

    # Use stratified CV for both grid search and final evaluation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_strategy,
        n_jobs=-1,
        scoring="accuracy",
        return_train_score=True,  # Track training scores to detect overfitting
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Cross-validate with the same CV strategy
    cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=cv_strategy, scoring="accuracy", n_jobs=-1
    )

    # Calculate train vs validation gap to detect overfitting
    train_scores = grid_search.cv_results_["mean_train_score"][grid_search.best_index_]
    val_scores = grid_search.cv_results_["mean_test_score"][grid_search.best_index_]
    overfitting_gap = train_scores - val_scores

    return (
        best_model,
        grid_search.best_params_,
        float(np.mean(cv_scores)),
        float(1.0 - np.mean(cv_scores)),
        float(np.std(cv_scores)),
        float(overfitting_gap),
        float(train_scores),
        float(val_scores),
    )


def run_one_vs_rest_svm(X, y, use_scaling=True):
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Add data validation
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class distribution: {np.bincount(y_train)}")

    (
        best_model,
        best_params,
        mean_acc,
        mean_err,
        std_acc,
        overfitting_gap,
        train_acc,
        val_acc,
    ) = cross_validate_svm(X_train, y_train, use_scaling=use_scaling)

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred)

    return {
        "model": best_model,
        "cv_mean_accuracy": mean_acc,
        "cv_std_accuracy": std_acc,
        "cv_mean_error": mean_err,
        "test_accuracy": acc,
        "classification_report": report,
        "best_params": best_params,
        "overfitting_gap": overfitting_gap,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Fit OVR LinearSVC on NN embeddings.")
    p.add_argument("--embedding_type", choices=["time_avg", "full_time"], required=True)
    p.add_argument("--layer_idx", type=int, required=True)
    p.add_argument("--signal_idx", type=int, required=True)
    p.add_argument("--attn_idx", type=int, required=True)
    p.add_argument(
        "--target", choices=["location", "f0_bin", "word_class"], required=True
    )
    p.add_argument("--time_avg_path", default=TIME_AVG_EMB_PATH)
    p.add_argument("--full_time_path", default=FULL_TIME_EMB_PATH)
    p.add_argument("--num_f0_bins", type=int, default=14)
    p.add_argument("--output_model", type=str, default="")
    p.add_argument("--no_scaling", action="store_true", help="Disable feature scaling")
    p.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
    return p.parse_args()


def main():
    args = parse_args()

    layer = LAYERS[args.layer_idx]
    signal = SIGNALS[args.signal_idx]
    attn = ATTN_TYPES[args.attn_idx]

    h5_path = (
        args.time_avg_path if args.embedding_type == "time_avg" else args.full_time_path
    )
    with h5py.File(h5_path, "r") as h5:
        X, emb_key = load_embeddings(h5, layer, signal, attn)
        y = load_labels(h5, args.target, num_f0_bins=args.num_f0_bins)

    # Align X and y lengths
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]

    # Drop NaNs/inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    results = run_one_vs_rest_svm(X, y, use_scaling=not args.no_scaling)

    print(f"Embeddings: {h5_path}")
    print(f"Dataset key: {emb_key}")
    print(f"Layer: {layer} | Signal: {signal} | Attn: {attn} | Target: {args.target}")
    print(f"Best Params: {results['best_params']}")
    print(
        f"CV Mean Acc: {results['cv_mean_accuracy']:.4f} +- {results['cv_std_accuracy']:.4f}"
    )
    print(f"CV Mean Error: {results['cv_mean_error']:.4f}")
    print(f"Train Acc: {results['train_accuracy']:.4f}")
    print(f"Val Acc: {results['val_accuracy']:.4f}")
    print(f"Overfitting Gap: {results['overfitting_gap']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print("Classification Report:\n" + results["classification_report"])

    if args.output_model:
        out_path = Path(args.output_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "model": results["model"],
                    "layer": layer,
                    "signal": signal,
                    "attn": attn,
                    "target": args.target,
                    "embedding_type": args.embedding_type,
                    "embeddings_dataset_key": emb_key,
                    "metrics": {k: v for k, v in results.items() if k != "model"},
                },
                f,
            )
        print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
