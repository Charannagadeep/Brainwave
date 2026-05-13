"""
evaluate.py
-----------
Reads all CSV result files from results/ and produces:
  - A summary table comparing all models
  - Per-subject accuracy distributions
  - Confusion matrix (averaged across subjects for BrainWave)
  - Transformer depth ablation table
  - Inference speed measurement

Run this after all training and baseline scripts have completed.

Usage:
    python evaluate.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALL_SUBJECTS, DEVICE, RESULTS_DIR, CHECKPOINT_DIR,
    CLASS_NAMES, N_LAYERS,
)
from dataset import load_subject_data, make_loader, subject_dependent_split
from models import get_model


# ─────────────────────────────────────────────────────────────────────────────
# Load and summarize CSV results
# ─────────────────────────────────────────────────────────────────────────────

def load_all_results() -> pd.DataFrame:
    """Load all result CSVs from RESULTS_DIR and concatenate."""
    dfs = []
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".csv"):
            path = os.path.join(RESULTS_DIR, fname)
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception:
                pass
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def print_summary_table(df: pd.DataFrame):
    """Print the main model comparison table."""
    if df.empty:
        print("No results found. Run training scripts first.")
        return

    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)

    model_order = ["csp_svm", "csp_rf", "cnn_only", "cnn_bilstm", "brainwave"]
    model_names = {
        "csp_svm":    "CSP + SVM",
        "csp_rf":     "CSP + Random Forest",
        "cnn_only":   "Spatial CNN Only",
        "cnn_bilstm": "CNN + BiLSTM",
        "brainwave":  "BrainWave (CNN+Tf)",
    }

    # Subject-dependent results
    print(f"\n{'Model':<25} {'Subj-Dep Acc':>14} {'Subj-Ind Acc':>14} {'Macro F1':>10}")
    print("-"*70)

    for mname in model_order:
        dep_rows = df[(df["model"] == mname) & (df["mode"] == "subj_dep")]
        ind_rows = df[(df["model"] == mname) & (df["mode"] == "subj_ind")]

        dep_acc = f"{dep_rows['test_acc'].mean():.1%} ± {dep_rows['test_acc'].std():.1%}" \
                  if len(dep_rows) > 0 else "N/A"
        dep_f1  = dep_rows["macro_f1"].mean() if len(dep_rows) > 0 else float("nan")

        ind_acc = f"{ind_rows['test_acc'].mean():.1%} ± {ind_rows['test_acc'].std():.1%}" \
                  if len(ind_rows) > 0 else "N/A"

        disp_name = model_names.get(mname, mname)
        print(f"  {disp_name:<23} {dep_acc:>14} {ind_acc:>14} {dep_f1:>10.3f}")

    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix (averaged across subjects for BrainWave subject-dependent)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mean_confusion_matrix() -> np.ndarray:
    """
    Load saved BrainWave models and compute normalized confusion matrix
    averaged across all subjects with checkpoints available.
    """
    from sklearn.metrics import confusion_matrix
    criterion = torch.nn.CrossEntropyLoss()

    cms = []
    subjects_with_ckpt = [
        s for s in ALL_SUBJECTS
        if os.path.exists(os.path.join(CHECKPOINT_DIR, f"brainwave_S{s:03d}.pt"))
    ]

    if not subjects_with_ckpt:
        print("No BrainWave checkpoints found for confusion matrix.")
        return None

    for subj in subjects_with_ckpt[:20]:  # cap at 20 for speed
        X, y = load_subject_data(subj)
        if X is None:
            continue
        _, _, _, _, X_test, y_test = subject_dependent_split(X, y)
        if len(X_test) == 0:
            continue

        model = get_model("brainwave")
        ckpt  = os.path.join(CHECKPOINT_DIR, f"brainwave_S{subj:03d}.pt")
        try:
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        except Exception:
            continue
        model.eval().to(DEVICE)

        preds, trues = [], []
        test_loader = make_loader(X_test, y_test, shuffle=False)
        with torch.no_grad():
            for xb, yb in test_loader:
                p = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
                preds.extend(p.tolist())
                trues.extend(yb.numpy().tolist())

        cm = confusion_matrix(trues, preds, labels=[0, 1, 2, 3])
        # Normalize per row
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        cms.append(cm_norm)

    if not cms:
        return None

    mean_cm = np.mean(cms, axis=0)

    print("\nBrainWave Normalized Confusion Matrix (mean across subjects):")
    print(f"{'':>12}", end="")
    for name in CLASS_NAMES:
        print(f"{name:>12}", end="")
    print()
    for i, true_name in enumerate(CLASS_NAMES):
        print(f"  {true_name:<10}", end="")
        for j in range(4):
            print(f"  {mean_cm[i, j]:.3f}    ", end="")
        print()

    return mean_cm


# ─────────────────────────────────────────────────────────────────────────────
# Depth ablation summary
# ─────────────────────────────────────────────────────────────────────────────

def print_depth_ablation(df: pd.DataFrame):
    """Print accuracy vs Transformer depth from ablation results."""
    ablation_rows = df[df["model"] == "brainwave"]
    if "n_layers" not in ablation_rows.columns:
        return

    ablation = ablation_rows.groupby("n_layers")["test_acc"].agg(["mean", "std"])
    if ablation.empty:
        return

    print("\nTransformer Depth Ablation (subject-dependent):")
    print(f"  {'Layers':>8} {'Acc Mean':>12} {'Acc Std':>10}")
    for L, row in ablation.iterrows():
        print(f"  {int(L):>8} {row['mean']:>11.1%} {row['std']:>9.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference speed
# ─────────────────────────────────────────────────────────────────────────────

def measure_inference_speed(n_runs: int = 100):
    """Measure time to classify one 3-second EEG epoch with BrainWave."""
    model = get_model("brainwave").to(DEVICE)
    model.eval()

    # Warmup
    dummy = torch.randn(1, 64, 480).to(DEVICE)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(dummy)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_epoch = (elapsed / n_runs) * 1000
    print(f"\nBrainWave inference speed: {ms_per_epoch:.2f} ms/epoch  "
          f"(avg over {n_runs} runs on {DEVICE})")
    print(f"Real-time BCI threshold: 100 ms → "
          f"{'PASSES ✓' if ms_per_epoch < 100 else 'FAILS ✗'}")

    return ms_per_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = load_all_results()
    print_summary_table(df)
    print_depth_ablation(df)
    compute_mean_confusion_matrix()
    measure_inference_speed()

    # Save consolidated results
    if not df.empty:
        out = os.path.join(RESULTS_DIR, "all_results.csv")
        df.to_csv(out, index=False)
        print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
