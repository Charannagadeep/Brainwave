"""
baselines.py
------------
Classical BCI baselines:
  1. CSP + SVM   (RBF kernel)
  2. CSP + Random Forest

CSP (Common Spatial Patterns) finds spatial filters that maximize the
variance ratio between classes. We extract log-variance features from
the filtered signals and feed them to sklearn classifiers.

For 4 classes we use a one-vs-rest (OVR) CSP decomposition:
one set of filters per class pair, then concatenate features.

Usage:
    python baselines.py                     # all subjects, both classifiers
    python baselines.py --subjects 1 2 3
    python baselines.py --classifier svm
"""

import argparse
import csv
import os
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALL_SUBJECTS, CSP_COMPONENTS, SEED, RESULTS_DIR,
    TRAIN_RATIO,
)
from dataset import load_subject_data


# ─────────────────────────────────────────────────────────────────────────────
# Manual CSP implementation (no sklearn dependency for the filters themselves)
# ─────────────────────────────────────────────────────────────────────────────

def compute_covariance(X: np.ndarray) -> np.ndarray:
    """
    Compute normalized covariance matrix for a set of epochs.

    Args:
        X : [N, C, T]
    Returns:
        cov : [C, C]  averaged normalized covariance
    """
    covs = []
    for trial in X:
        # trial: [C, T]
        c = trial @ trial.T
        c /= np.trace(c) + 1e-10
        covs.append(c)
    return np.mean(covs, axis=0)


def csp_filters(X_a: np.ndarray, X_b: np.ndarray, n: int = 4) -> np.ndarray:
    """
    Compute CSP filters for a binary classification problem.

    Args:
        X_a, X_b : [N, C, T] arrays for the two classes
        n        : number of filters per class (total returned = 2*n)
    Returns:
        W : [2n, C] spatial filter matrix (rows are filters)
    """
    cov_a = compute_covariance(X_a)
    cov_b = compute_covariance(X_b)
    cov_total = cov_a + cov_b

    # Eigendecomposition of the composite covariance
    eigenvalues, eigenvectors = np.linalg.eigh(cov_total)
    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Whitening matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
    P = D_inv_sqrt @ eigenvectors.T  # [C, C] whitening

    # Project and solve generalized eigenvalue problem in whitened space
    S_a = P @ cov_a @ P.T
    eigenvalues_a, W_a = np.linalg.eigh(S_a)
    order_a = np.argsort(eigenvalues_a)[::-1]
    W_a     = W_a[:, order_a]

    # CSP filters in original space: W^T = W_a^T @ P
    W = (W_a.T @ P)  # [C, C]

    # Select n filters from each end (most discriminative)
    selected = np.vstack([W[:n, :], W[-n:, :]])  # [2n, C]
    return selected


def extract_csp_features(X: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """
    Apply spatial filters and compute log-variance features.

    Args:
        X       : [N, C, T]
        filters : [n_filters, C]
    Returns:
        features : [N, n_filters]
    """
    # Apply filters: [N, n_filters, T]
    filtered = np.einsum("fc,nct->nft", filters, X)
    # Log-variance: [N, n_filters]
    var = np.var(filtered, axis=2)
    var = np.log(var + 1e-10)
    return var


class MulticlassCSP:
    """
    One-vs-rest CSP for multi-class problems.

    For each class c, computes CSP filters that discriminate c from all others.
    Feature vector is the concatenation of all OVR filter outputs.
    """
    def __init__(self, n_components: int = CSP_COMPONENTS):
        self.n_components  = n_components
        self.filters_list_ = []
        self.classes_      = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X : [N, C, T]
            y : [N]
        """
        self.classes_      = np.unique(y)
        self.filters_list_ = []
        n = self.n_components // 2  # n from each end

        for c in self.classes_:
            X_c     = X[y == c]
            X_other = X[y != c]
            if len(X_c) < 2 or len(X_other) < 2:
                # Degenerate case: use identity-like filter
                self.filters_list_.append(np.eye(X.shape[1])[:self.n_components])
            else:
                W = csp_filters(X_c, X_other, n=n)
                self.filters_list_.append(W)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Returns [N, n_classes * n_components] feature matrix."""
        feats = [extract_csp_features(X, W) for W in self.filters_list_]
        return np.hstack(feats)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_subject(
    subject_id:  int,
    classifier:  str = "both",
) -> list[dict]:
    """
    Run CSP + SVM and/or CSP + RF on one subject's data.
    Returns list of result dicts.
    """
    X, y = load_subject_data(subject_id)
    if X is None or len(X) < 20:
        return []

    # Train/test split (stratified, 80/20)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - TRAIN_RATIO, random_state=SEED)
    train_idx, test_idx = next(sss.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Fit CSP on training data
    csp = MulticlassCSP(n_components=CSP_COMPONENTS)
    F_train = csp.fit_transform(X_train, y_train)
    F_test  = csp.transform(X_test)

    results = []
    classifiers = []

    if classifier in ("svm", "both"):
        classifiers.append(("csp_svm", Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
        ])))

    if classifier in ("rf", "both"):
        classifiers.append(("csp_rf", Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, max_depth=20,
                random_state=SEED, n_jobs=-1,
            )),
        ])))

    for name, pipe in classifiers:
        pipe.fit(F_train, y_train)
        preds = pipe.predict(F_test)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="macro", zero_division=0)

        results.append({
            "subject":  subject_id,
            "model":    name,
            "mode":     "subj_dep",
            "n_train":  len(X_train),
            "n_test":   len(X_test),
            "test_acc": round(acc, 4),
            "macro_f1": round(f1, 4),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Run classical BCI baselines")
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--classifier", choices=["svm", "rf", "both"], default="both")
    args = parser.parse_args()

    from config import PROCESSED_DIR
    subjects = args.subjects if args.subjects else ALL_SUBJECTS
    subjects = [s for s in subjects
                if os.path.exists(os.path.join(PROCESSED_DIR, f"S{s:03d}_X.npy"))]

    print(f"Running CSP baselines on {len(subjects)} subjects ...")

    all_results = []
    for subj in subjects:
        res = run_baseline_subject(subj, args.classifier)
        if res:
            for r in res:
                print(f"  S{subj:03d} | {r['model']:8s} | "
                      f"acc={r['test_acc']:.3f} | f1={r['macro_f1']:.3f}")
            all_results.extend(res)

    if not all_results:
        print("No results — make sure preprocessing has been run first.")
        return

    # Print summary
    for model_name in ["csp_svm", "csp_rf"]:
        accs = [r["test_acc"] for r in all_results if r["model"] == model_name]
        f1s  = [r["macro_f1"] for r in all_results if r["model"] == model_name]
        if accs:
            print(f"\n{model_name}: "
                  f"acc={np.mean(accs):.3f}±{np.std(accs):.3f} | "
                  f"f1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")

    # Save
    out_path = os.path.join(RESULTS_DIR, "baselines.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
