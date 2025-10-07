import h5py
import numpy as np
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

# ========== 1. PCA DIMENSIONALITY REDUCTION ==========
def apply_pca(X, n_components=512, target_variance=0.8, random_state=42):
    """
    Apply PCA dimensionality reduction to features with adaptive component selection.
    
    The function uses a two-step approach:
    1. Fit PCA with min(n_components, n_features, n_samples) components
    2. If more than n_components would be used, find minimum components for target_variance
    
    Parameters:
    -----------
    X : numpy array
        Feature matrix
    n_components : int
        Minimum number of components (default: 512)
    target_variance : float
        Target explained variance ratio when n_components would be exceeded (default: 0.8)
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    X_reduced : numpy array
        PCA-transformed features
    pca : PCA object
        Fitted PCA transformer
    explained_variance : float
        Actual explained variance ratio
    """
    n_samples, n_features = X.shape
    max_components = min(n_features, n_samples)
    
    print(f"\nApplying PCA:")
    print(f"  Original dimensions: {n_features}")
    print(f"  Minimum components requested: {n_components}")
    print(f"  Target variance (if > {n_components} components): {target_variance}")
    
    # Determine effective number of components
    if max_components <= n_components:
        # Use all available components
        effective_n_components = max_components
        print(f"  Using all available components: {effective_n_components}")
    else:
        # First, fit PCA with max_components to analyze variance
        print(f"  Fitting initial PCA to analyze variance...")
        pca_temp = PCA(n_components=max_components, random_state=random_state)
        pca_temp.fit(X)
        
        # Calculate cumulative explained variance
        cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        
        # Find number of components needed for target variance
        components_for_target = np.argmax(cumsum_var >= target_variance) + 1
        
        print(f"  Components for {target_variance*100}% variance: {components_for_target}")
        print(f"  Variance at {n_components} components: {cumsum_var[n_components-1]:.4f}")
        
        # Use the maximum of n_components and components_for_target
        effective_n_components = max(n_components, components_for_target)
        
        if effective_n_components > n_components:
            print(f"  ⚠ Need {effective_n_components} components to reach {target_variance*100}% variance")
            print(f"  Using {effective_n_components} components (exceeds minimum of {n_components})")
        else:
            print(f"  Using minimum {n_components} components")
    
    # Fit final PCA with determined number of components
    pca = PCA(n_components=effective_n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"  Final components: {effective_n_components}")
    print(f"  Explained variance: {explained_variance:.4f}")
    print(f"  Reduced shape: {X_reduced.shape}")
    print(f"  Memory footprint: {X_reduced.nbytes / (1024**2):.2f} MB")
    
    return X_reduced, pca, explained_variance


# ========== 2. SPLIT DATA ==========
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# ========== 3. HYPERPARAMETER TUNING ==========
def cross_validate_svm(X_train, y_train, cv=5, param_grid=None, max_iter=1000, dual=True, 
                       random_state=0):
    """
    Perform cross-validation for SVM hyperparameters (C and kernel).
    X_train is already PCA-transformed.
    """
    if param_grid is None:
        param_grid = {
            'estimator__C': [0.1, 1, 10],
        }
    
    # dual=True when n_samples < n_features 
    ovr = OneVsRestClassifier(svm.LinearSVC(max_iter=max_iter, dual=dual,
                                            random_state=random_state))

    grid_search = GridSearchCV(ovr, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Cross-validate best model to compute average accuracy
    best_model = grid_search.best_estimator_
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)

    return best_model, grid_search.best_params_, mean_accuracy, std_accuracy


# ========== 4. SINGLE RUN FUNCTION ==========
def run_one_vs_rest_svm(X, y, param_grid=None, max_iter=1000, dual=True, 
                        random_state=0):
    # Step 1: Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 2: Cross-validation
    print("Running cross-validation and hyperparameter tuning...")
    best_model, best_params, mean_acc, std_acc = cross_validate_svm(
        X_train, y_train, param_grid=param_grid, max_iter=max_iter, 
        dual=dual, random_state=random_state
    )

    print(f"\nBest Parameters: {best_params}")
    print(f"Cross-Validation Mean Accuracy: {mean_acc:.4f}")
    print(f"Cross-Validation Std Dev: {std_acc:.4f}")

    # Step 3: Train best model on full training set and evaluate
    print("\nEvaluating on test set...")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return {
        'model': best_model,
        'cv_mean_accuracy': mean_acc,
        'cv_std_accuracy': std_acc,
        'test_accuracy': acc,
        'classification_report': report,
        'best_params': best_params
    }


# ========== 5. K-FOLD FUNCTION ==========
def run_kfold_svm(X, y, k=10, param_grid=None, max_iter=1000, dual=True, 
                  random_state=42, cv_inner=5):
    """
    Run K-fold cross-validation on SVM.
    X is already PCA-transformed before calling this function.
    Returns results for each fold and the best model across all folds.
    
    Parameters:
    -----------
    X : numpy array
        Feature matrix (already PCA-transformed)
    y : numpy array
        Labels
    k : int
        Number of folds for outer cross-validation
    param_grid : dict
        Parameter grid for GridSearchCV
    max_iter : int
        Maximum iterations for LinearSVC
    dual : bool
        Use dual formulation for LinearSVC
    random_state : int
        Random state for reproducibility
    cv_inner : int
        Number of folds for inner cross-validation (hyperparameter tuning)
    """
    if param_grid is None:
        param_grid = {
            'estimator__C': [0.1, 1, 10],
        }
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    fold_results = []
    best_fold_acc = 0
    best_fold_model = None
    best_fold_idx = -1
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx + 1}/{k}")
        print(f"{'='*60}")
        
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Run hyperparameter tuning on training fold
        ovr = OneVsRestClassifier(svm.LinearSVC(max_iter=max_iter, dual=dual, 
                                                random_state=random_state))
        grid_search = GridSearchCV(ovr, param_grid, cv=cv_inner, n_jobs=-1, 
                                  scoring='accuracy')
        grid_search.fit(X_train_fold, y_train_fold)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Evaluate on test fold
        y_pred = best_model.predict(X_test_fold)
        fold_acc = accuracy_score(y_test_fold, y_pred)
        fold_report = classification_report(y_test_fold, y_pred, output_dict=True)
        
        print(f"Fold {fold_idx + 1} Test Accuracy: {fold_acc:.4f}")
        print(f"Best Params: {best_params}")
        
        fold_result = {
            'fold': fold_idx + 1,
            'best_params': best_params,
            'test_accuracy': fold_acc,
            'classification_report': fold_report,
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        }
        fold_results.append(fold_result)
        
        # Track best fold
        if fold_acc > best_fold_acc:
            best_fold_acc = fold_acc
            best_fold_model = best_model
            best_fold_idx = fold_idx + 1
    
    # Compute aggregate statistics
    accuracies = [fr['test_accuracy'] for fr in fold_results]
    
    summary = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'best_fold_idx': best_fold_idx,
        'best_fold_accuracy': best_fold_acc
    }
    
    print(f"\n{'='*60}")
    print(f"K-Fold Summary (k={k})")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Best Fold: {best_fold_idx} (Accuracy: {best_fold_acc:.4f})")
    
    return {
        'fold_results': fold_results,
        'summary': summary,
        'best_model': best_fold_model,
        'best_fold_idx': best_fold_idx
    }


# ========== 6. LOAD H5 DATA ==========
def load_h5_layer_data(h5_path, layer_name, label_key='target_word_int'):
    """
    Load activations and labels from H5 file for a specific layer.
    
    Parameters:
    -----------
    h5_path : str
        Path to H5 file
    layer_name : str
        Name of the layer group to load
    label_key : str
        Key for labels dataset
    
    Returns:
    --------
    X : numpy array
        Activations
    y : numpy array
        Labels
    """
    with h5py.File(h5_path, 'r') as f:
        if layer_name not in f:
            raise ValueError(f"Layer '{layer_name}' not found in H5 file. Available: {list(f.keys())}")
        
        if label_key not in f:
            raise ValueError(f"Label key '{label_key}' not found in H5 file. Available: {list(f.keys())}")
        
        X = f[layer_name][:]
        y = f[label_key][:]
        
        print(f"Loaded layer '{layer_name}':")
        print(f"  Activations shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Memory footprint: {X.nbytes / (1024**2):.2f} MB")
        
    return X, y


def get_layer_names(h5_path, label_key='target_word_int', exclude_keys=None):
    """
    Get list of layer names from H5 file, excluding specified keys.
    
    Parameters:
    -----------
    h5_path : str
        Path to H5 file
    label_key : str
        Key for labels to exclude
    exclude_keys : list
        Additional keys to exclude (e.g., metadata keys)
    
    Returns:
    --------
    list : Sorted list of valid layer names
    """
    if exclude_keys is None:
        # Default exclusions: labels and metadata keys
        exclude_keys = [label_key, 'layer_names']
    else:
        # Ensure label_key is in exclusions
        if label_key not in exclude_keys:
            exclude_keys.append(label_key)
    
    with h5py.File(h5_path, 'r') as f:
        # Get all keys and filter out excluded ones
        all_keys = list(f.keys())
        layers = [key for key in all_keys if key not in exclude_keys]
        
        # Sort layers to ensure consistent ordering
        layers.sort()
        
    return layers


# ========== 7. SAVE RESULTS ==========
def save_results(results, output_path):
    """
    Save results dictionary to JSON file.
    Note: Scikit-learn models are not JSON serializable, so we exclude them.
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a serializable copy
    serializable_results = {}
    for key, value in results.items():
        if key in ['best_model', 'model']:
            # Skip model objects
            continue
        elif key == 'fold_results':
            # Process fold results
            serializable_results[key] = value
        else:
            serializable_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


# ========== 8. PARSE PARAMETER GRID ==========
def parse_param_grid(c_values_str):
    """
    Parse C values string into parameter grid.
    
    Parameters:
    -----------
    c_values_str : str
        Comma-separated list of C values, e.g., "0.1,1,10"
    
    Returns:
    --------
    dict : Parameter grid for GridSearchCV
    """
    c_values = [float(c.strip()) for c in c_values_str.split(',')]
    return {'estimator__C': c_values}


# ========== 9. MAIN FUNCTION ==========
def main():
    parser = argparse.ArgumentParser(
        description='Run OneVsRest SVM on H5 layer activations with K-fold CV and Adaptive PCA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to H5 file containing layer activations')
    parser.add_argument('--layer_idx', type=int, required=True,
                        help='Layer index (from SLURM job array, 0-indexed)')
    
    # Data arguments
    parser.add_argument('--label_key', type=str, default='target_word_int',
                        help='Key for labels in H5 file')
    parser.add_argument('--exclude_keys', type=str, default='layer_names',
                        help='Comma-separated list of additional keys to exclude from layer processing')
    
    # Cross-validation arguments
    parser.add_argument('--k_folds', type=int, default=10,
                        help='Number of folds for outer K-fold cross-validation')
    parser.add_argument('--cv_inner', type=int, default=5,
                        help='Number of folds for inner cross-validation (hyperparameter tuning)')
    
    # SVM hyperparameters
    parser.add_argument('--c_values', type=str, default='0.1,1,10',
                        help='Comma-separated list of C values to try (e.g., "0.01,0.1,1,10,100")')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum number of iterations for LinearSVC')
    parser.add_argument('--dual', action='store_true', default=True,
                        help='Use dual formulation (set True when n_samples < n_features)')
    parser.add_argument('--no_dual', dest='dual', action='store_false',
                        help='Use primal formulation (set when n_samples > n_features)')
    
    # PCA arguments
    parser.add_argument('--n_components', type=int, default=512,
                        help='Minimum number of PCA components (default: 512)')
    parser.add_argument('--target_variance', type=float, default=0.8,
                        help='Target explained variance when more than n_components needed (default: 0.8)')
    parser.add_argument('--skip_pca', action='store_true',
                        help='Skip PCA dimensionality reduction')
    
    # Random state
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./svm_results',
                        help='Directory to save results')
    parser.add_argument('--output_prefix', type=str, default='svm_results',
                        help='Prefix for output filename')
    
    # Verbose
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse parameter grid
    param_grid = parse_param_grid(args.c_values)
    
    # Parse exclude keys
    exclude_keys = [key.strip() for key in args.exclude_keys.split(',') if key.strip()]
    if args.label_key not in exclude_keys:
        exclude_keys.append(args.label_key)
    
    if args.verbose:
        print(f"\nConfiguration:")
        print(f"  H5 Path: {args.h5_path}")
        print(f"  Layer Index: {args.layer_idx}")
        print(f"  Label Key: {args.label_key}")
        print(f"  Exclude Keys: {exclude_keys}")
        print(f"  K-Folds (outer): {args.k_folds}")
        print(f"  CV (inner): {args.cv_inner}")
        print(f"  C values: {param_grid['estimator__C']}")
        print(f"  Max iterations: {args.max_iter}")
        print(f"  Dual formulation: {args.dual}")
        print(f"  PCA components (min): {args.n_components}")
        print(f"  PCA target variance: {args.target_variance}")
        print(f"  Skip PCA: {args.skip_pca}")
        print(f"  Random state: {args.random_state}")
    
    # Get all layer names
    layer_names = get_layer_names(args.h5_path, args.label_key, exclude_keys)
    print(f"\nAvailable layers ({len(layer_names)}): {layer_names}")
    
    # Get layer name from index
    if args.layer_idx < 0 or args.layer_idx >= len(layer_names):
        raise ValueError(f"Layer index {args.layer_idx} out of range [0, {len(layer_names)-1}]")
    
    layer_name = layer_names[args.layer_idx]
    print(f"\nProcessing layer index {args.layer_idx}: '{layer_name}'")
    
    # Load data
    print("\nLoading data from H5 file...")
    X_original, y = load_h5_layer_data(args.h5_path, layer_name, args.label_key)
    original_shape = X_original.shape
    
    # Apply PCA if not skipped
    pca_info = None
    if not args.skip_pca:
        X, pca_model, explained_variance = apply_pca(
            X_original, 
            n_components=args.n_components,
            target_variance=args.target_variance,
            random_state=args.random_state
        )
        pca_info = {
            'n_components_requested': args.n_components,
            'target_variance': args.target_variance,
            'effective_n_components': X.shape[1],
            'explained_variance': float(explained_variance)
        }
    else:
        print("\nSkipping PCA (--skip_pca flag set)")
        X = X_original
    
    # Check dual vs primal recommendation
    n_samples, n_features = X.shape
    if args.verbose:
        if args.dual and n_samples > n_features:
            print(f"\nWarning: Using dual=True but n_samples ({n_samples}) > n_features ({n_features})")
            print("Consider using --no_dual for better performance")
        elif not args.dual and n_samples < n_features:
            print(f"\nWarning: Using dual=False but n_samples ({n_samples}) < n_features ({n_features})")
            print("Consider using --dual for better performance")
    
    # Run K-fold SVM
    print(f"\nRunning {args.k_folds}-fold cross-validation...")
    kfold_results = run_kfold_svm(
        X, y, 
        k=args.k_folds,
        param_grid=param_grid,
        max_iter=args.max_iter,
        dual=args.dual,
        random_state=args.random_state,
        cv_inner=args.cv_inner
    )
    
    # Prepare results for saving
    results_to_save = {
        'layer_name': layer_name,
        'layer_idx': args.layer_idx,
        'h5_path': args.h5_path,
        'timestamp': datetime.now().isoformat(),
        'original_data_shape': list(original_shape),
        'data_shape': list(X.shape),
        'n_samples': int(n_samples),
        'n_features': int(n_features),
        'n_classes': int(len(np.unique(y))),
        'pca_applied': not args.skip_pca,
        'pca_info': pca_info,
        'k_folds': args.k_folds,
        'cv_inner': args.cv_inner,
        'hyperparameters': {
            'c_values': param_grid['estimator__C'],
            'max_iter': args.max_iter,
            'dual': args.dual,
            'random_state': args.random_state
        },
        'fold_results': kfold_results['fold_results'],
        'summary': kfold_results['summary']
    }
    
    # Save results with model name in path
    h5_parent = Path(args.h5_path).parent.name
    output_filename = f"{args.output_prefix}_{layer_name}.json"
    output_path = Path(args.output_dir) / h5_parent / output_filename
    save_results(results_to_save, output_path)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
    print(f"Best fold: {kfold_results['best_fold_idx']} "
          f"(Accuracy: {kfold_results['summary']['best_fold_accuracy']:.4f})")
    print(f"Mean accuracy: {kfold_results['summary']['mean_accuracy']:.4f} "
          f"± {kfold_results['summary']['std_accuracy']:.4f}")


if __name__ == "__main__":
    main()