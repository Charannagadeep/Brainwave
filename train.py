"""
train.py
--------
Training loop for BrainWave and CNN-only / CNN-BiLSTM variants.

Two modes:
  --mode subj_dep   Subject-dependent: train one model per subject (80/20 split)
  --mode subj_ind   Subject-independent: 5-fold CV across subjects

Results are saved to results/train_log_{mode}.csv.

Usage:
    python train.py --mode subj_dep --model brainwave
    python train.py --mode subj_dep --model cnn_only
    python train.py --mode subj_ind --model brainwave --folds 5
    python train.py --mode subj_dep --subjects 1 2 3   # run specific subjects
    python train.py --mode subj_dep --layers 1          # depth ablation
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALL_SUBJECTS, DEVICE, SEED,
    LR, WEIGHT_DECAY, N_EPOCHS, PATIENCE, BATCH_SIZE,
    CHECKPOINT_DIR, RESULTS_DIR,
    N_LAYERS, EMBED_DIM, N_HEADS, FF_DIM, DROPOUT, N_CLASSES,
)
from dataset import (
    load_subject_data, subject_dependent_split,
    subject_independent_folds, load_and_concat,
    make_loader,
)
from models import get_model, count_parameters

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Core training / evaluation functions
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """One epoch of training. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Returns (loss, accuracy) on a DataLoader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss   = criterion(logits, yb)
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(dim=1) == yb).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total


def train_model(
    model:      nn.Module,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    checkpoint_path: str = None,
    device:     str = DEVICE,
    lr:         float = LR,
    weight_decay: float = WEIGHT_DECAY,
    n_epochs:   int = N_EPOCHS,
    patience:   int = PATIENCE,
    verbose:    bool = False,
) -> dict:
    """
    Full training loop with:
    - Adam optimizer
    - Cosine annealing LR schedule
    - Early stopping on validation loss
    - Best-weight checkpointing

    Returns dict with training history and best validation accuracy.
    """
    model = model.to(device)
    optimizer  = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler  = CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion  = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)

    best_val_loss = float("inf")
    best_weights  = None
    no_improve    = 0
    history       = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if verbose:
            print(f"  Epoch {epoch:3d} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.3f} | "
                  f"{time.time()-t0:.1f}s")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
            if checkpoint_path:
                torch.save(best_weights, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {"history": history, "best_val_loss": best_val_loss}


# ─────────────────────────────────────────────────────────────────────────────
# Subject-dependent evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_subject_dependent(
    model_name: str,
    subjects:   list,
    n_layers:   int = N_LAYERS,
    verbose:    bool = False,
) -> list:
    """
    Train and evaluate one model per subject using 80/10/10 split.
    Returns list of per-subject result dicts.
    """
    from sklearn.metrics import f1_score, confusion_matrix
    criterion = nn.CrossEntropyLoss()
    results   = []

    for subj in subjects:
        X, y = load_subject_data(subj)
        if X is None or len(X) < 20:
            print(f"  [SKIP] S{subj:03d}: not enough data")
            continue

        X_train, y_train, X_val, y_val, X_test, y_test = \
            subject_dependent_split(X, y)

        model = get_model(model_name, n_layers=n_layers) \
                if model_name == "brainwave" else get_model(model_name)

        ckpt = os.path.join(CHECKPOINT_DIR, f"{model_name}_S{subj:03d}.pt")
        info = train_model(
            model, X_train, y_train, X_val, y_val,
            checkpoint_path=ckpt,
            verbose=verbose,
        )

        # Evaluate on test set
        test_loader = make_loader(X_test, y_test, shuffle=False)
        _, test_acc = evaluate_loader(model, test_loader, criterion, DEVICE)

        # Get predictions for F1 and confusion matrix
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(yb.numpy().tolist())

        f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
        cm  = confusion_matrix(all_true, all_preds, labels=[0, 1, 2, 3])

        res = {
            "subject":    subj,
            "model":      model_name,
            "mode":       "subj_dep",
            "n_layers":   n_layers,
            "n_train":    len(X_train),
            "n_test":     len(X_test),
            "test_acc":   round(test_acc, 4),
            "macro_f1":   round(f1, 4),
            "n_params":   count_parameters(model),
        }
        results.append(res)
        print(f"  S{subj:03d} | acc={test_acc:.3f} | f1={f1:.3f} | "
              f"n_train={len(X_train)} | n_test={len(X_test)}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Subject-independent evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_subject_independent(
    model_name: str,
    n_folds:    int = 5,
    verbose:    bool = False,
) -> list:
    """
    5-fold cross-validation across subjects.
    Each fold holds out ~20 subjects entirely.
    Returns list of per-fold result dicts.
    """
    from sklearn.metrics import f1_score
    criterion = nn.CrossEntropyLoss()
    folds     = subject_independent_folds(n_folds=n_folds)
    results   = []

    for fold_idx, (train_subj, val_subj, test_subj) in enumerate(folds):
        print(f"\nFold {fold_idx+1}/{n_folds} | "
              f"train={len(train_subj)} subj | "
              f"val={len(val_subj)} subj | "
              f"test={len(test_subj)} subj")

        X_train, y_train = load_and_concat(train_subj)
        X_val,   y_val   = load_and_concat(val_subj)
        X_test,  y_test  = load_and_concat(test_subj)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  [SKIP] fold {fold_idx+1}: no data")
            continue

        model = get_model(model_name)
        ckpt  = os.path.join(CHECKPOINT_DIR, f"{model_name}_fold{fold_idx+1}.pt")

        train_model(
            model, X_train, y_train, X_val, y_val,
            checkpoint_path=ckpt, verbose=verbose,
        )

        test_loader = make_loader(X_test, y_test, shuffle=False)
        _, test_acc = evaluate_loader(model, test_loader, criterion, DEVICE)

        all_preds, all_true = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(yb.numpy().tolist())

        f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

        res = {
            "fold":      fold_idx + 1,
            "model":     model_name,
            "mode":      "subj_ind",
            "n_train":   len(X_train),
            "n_test":    len(X_test),
            "test_acc":  round(test_acc, 4),
            "macro_f1":  round(f1, 4),
        }
        results.append(res)
        print(f"  Fold {fold_idx+1} | acc={test_acc:.3f} | f1={f1:.3f}")

    if results:
        accs = [r["test_acc"] for r in results]
        f1s  = [r["macro_f1"] for r in results]
        print(f"\n{model_name} subject-independent: "
              f"acc={np.mean(accs):.3f}±{np.std(accs):.3f} | "
              f"f1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: list, path: str):
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train BrainWave models")
    parser.add_argument("--mode",  choices=["subj_dep", "subj_ind"], required=True)
    parser.add_argument("--model", choices=["brainwave", "cnn_only", "cnn_bilstm"],
                        default="brainwave")
    parser.add_argument("--subjects", nargs="+", type=int, default=None,
                        help="Specific subjects (default: all 103)")
    parser.add_argument("--folds",   type=int, default=5)
    parser.add_argument("--layers",  type=int, default=N_LAYERS,
                        help="Number of Transformer layers (ablation)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Model:  {args.model}")
    print(f"Mode:   {args.mode}")

    if args.mode == "subj_dep":
        subjects = args.subjects if args.subjects else ALL_SUBJECTS
        # Keep only subjects with preprocessed files
        subjects = [s for s in subjects
                    if os.path.exists(
                        os.path.join(
                            os.path.join(os.path.dirname(__file__), "data", "processed"),
                            f"S{s:03d}_X.npy"
                        )
                    )]
        print(f"Subjects with data: {len(subjects)}")
        results = run_subject_dependent(
            args.model, subjects,
            n_layers=args.layers,
            verbose=args.verbose,
        )
        out_path = os.path.join(
            RESULTS_DIR, f"subj_dep_{args.model}_L{args.layers}.csv"
        )

    else:  # subj_ind
        results = run_subject_independent(
            args.model, n_folds=args.folds, verbose=args.verbose
        )
        out_path = os.path.join(RESULTS_DIR, f"subj_ind_{args.model}.csv")

    save_results(results, out_path)


if __name__ == "__main__":
    main()
