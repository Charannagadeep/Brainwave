"""
dataset.py
----------
PyTorch Dataset wrapper around the preprocessed .npy files.
Also provides train/val/test split utilities and the 5-fold
subject-independent cross-validation splitter.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALL_SUBJECTS, PROCESSED_DIR,
    TRAIN_RATIO, VAL_RATIO, N_FOLDS,
    BATCH_SIZE, SEED,
)


class EEGDataset(Dataset):
    """
    Simple wrapper around pre-computed (X, y) arrays.

    Args:
        X : np.ndarray [N, 64, 480] float32
        y : np.ndarray [N]          int64
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
    """Wrap arrays into a DataLoader."""
    ds = EEGDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def load_subject_data(subject_id: int) -> tuple | tuple:
    """Load preprocessed .npy arrays for one subject."""
    x_path = os.path.join(PROCESSED_DIR, f"S{subject_id:03d}_X.npy")
    y_path = os.path.join(PROCESSED_DIR, f"S{subject_id:03d}_y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None, None
    return np.load(x_path), np.load(y_path)


def subject_dependent_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed:        int   = SEED,
) -> tuple:
    """
    Split one subject's data into train, val, test sets.

    Uses stratified sampling to preserve class proportions.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)

    n_test  = max(1, int(n * (1 - train_ratio)))
    n_val   = max(1, int((n - n_test) * val_ratio))
    n_train = n - n_test - n_val

    # Stratified: collect per-class indices, split each proportionally
    classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []

    for c in classes:
        c_idx = rng.permutation(np.where(y == idx)[0]) if False else \
                rng.permutation(idx[y[idx] == c])

        n_c        = len(c_idx)
        n_c_test   = max(1, int(n_c * (1 - train_ratio)))
        n_c_val    = max(1, int((n_c - n_c_test) * val_ratio))
        n_c_train  = n_c - n_c_test - n_c_val

        train_idx.extend(c_idx[:n_c_train].tolist())
        val_idx.extend(c_idx[n_c_train:n_c_train + n_c_val].tolist())
        test_idx.extend(c_idx[n_c_train + n_c_val:].tolist())

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
    )


def subject_independent_folds(
    subjects: list = None,
    n_folds:  int       = N_FOLDS,
    seed:     int       = SEED,
) -> list[tuple[list, list, list]]:
    """
    Split subjects into N_FOLDS groups for subject-independent evaluation.

    Each fold returns (train_subjects, val_subjects, test_subjects).
    All epochs from test subjects are held out completely.

    Returns:
        List of (train_subj_list, val_subj_list, test_subj_list) tuples.
    """
    if subjects is None:
        # Only include subjects whose preprocessed files exist
        subjects = [s for s in ALL_SUBJECTS
                    if os.path.exists(os.path.join(PROCESSED_DIR, f"S{s:03d}_X.npy"))]

    rng = np.random.default_rng(seed)
    subjects = rng.permutation(subjects).tolist()

    fold_size = len(subjects) // n_folds
    folds = []

    for i in range(n_folds):
        test_start = i * fold_size
        test_end   = test_start + fold_size if i < n_folds - 1 else len(subjects)
        test_subj  = subjects[test_start:test_end]
        rest_subj  = subjects[:test_start] + subjects[test_end:]

        # 10% of training subjects → validation
        n_val   = max(1, int(len(rest_subj) * 0.1))
        val_subj   = rest_subj[:n_val]
        train_subj = rest_subj[n_val:]

        folds.append((train_subj, val_subj, test_subj))

    return folds


def load_and_concat(subject_ids: list) -> tuple:
    """Load and concatenate data from multiple subjects."""
    xs, ys = [], []
    for sid in subject_ids:
        X, y = load_subject_data(sid)
        if X is not None and len(X) > 0:
            xs.append(X)
            ys.append(y)
    if not xs:
        return np.empty((0, 64, 480), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.concatenate(xs), np.concatenate(ys)
