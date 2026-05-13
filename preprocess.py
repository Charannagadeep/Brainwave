"""
preprocess.py
-------------
Full MNE preprocessing pipeline. Compatible with Python 3.9+.

Output per subject:
    processed/S{id:03d}_X.npy  shape [N_epochs, 64, 480]  float32
    processed/S{id:03d}_y.npy  shape [N_epochs]            int64

Pipeline:
    1. Load raw EDF files
    2. Bandpass filter 4-40 Hz
    3. Extract 3-second epochs at MI cue onset
    4. Baseline correction
    5. Artifact rejection (>100uV peak-to-peak)
    6. Z-score normalization per channel
    7. Label mapping (run-dependent T1/T2 codes)

Usage:
    python3 preprocess.py
    python3 preprocess.py --subjects 1 2 3
    python3 preprocess.py --force
"""

import argparse
import os
import sys
import warnings
from typing import Optional, Tuple

import mne
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALL_SUBJECTS, HAND_RUNS, FEET_RUNS,
    DATA_DIR, PROCESSED_DIR,
    L_FREQ, H_FREQ, TMIN, TMAX, BASELINE,
    ARTIFACT_THRESH, N_CHANNELS, N_TIMES, SFREQ,
)
from download_data import download_subject


def _load_edf(fpath):
    """Load one EDF file. Returns raw or None."""
    try:
        return mne.io.read_raw_edf(fpath, preload=True, verbose=False)
    except Exception as e:
        tqdm.write(f"  [WARN] Cannot read {fpath}: {e}")
        return None


def _process_runs(run_files, label_map):
    """
    Process a list of EDF run files with a given label map.
    label_map: dict { annotation_substring -> class_int }
               e.g. {'T0': 0, 'T1': 1, 'T2': 2}

    Returns (X_list, y_list) of numpy arrays.
    """
    xs, ys = [], []

    for fpath in run_files:
        raw = _load_edf(fpath)
        if raw is None:
            continue

        # Step 1: bandpass filter
        raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)

        # Step 2: get events from annotations
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
        except Exception:
            continue

        # Build id->class map using the annotation keys
        id_to_class = {}
        for key, val in event_id.items():
            for ann_substr, class_int in label_map.items():
                if ann_substr in key:
                    id_to_class[val] = class_int
                    break

        if not id_to_class:
            continue

        # Step 3: epoch extraction
        try:
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            epochs = mne.Epochs(
                raw, events,
                event_id=event_id,
                tmin=TMIN,
                tmax=TMAX - 1.0 / SFREQ,
                proj=False,
                picks=picks,
                baseline=None,
                preload=True,
                verbose=False,
            )
        except Exception:
            continue

        # Step 4: baseline correction
        epochs.apply_baseline(baseline=BASELINE, verbose=False)

        # Step 5: artifact rejection
        epochs.drop_bad(reject={"eeg": ARTIFACT_THRESH}, verbose=False)

        # Get data for 0-3s window
        data = epochs.get_data(tmin=0.0, tmax=3.0 - 1.0 / SFREQ)

        if data.shape[2] > N_TIMES:
            data = data[:, :, :N_TIMES]
        if data.shape[2] != N_TIMES or data.shape[1] != N_CHANNELS:
            continue

        # Step 6: z-score normalization per channel per epoch
        mean = data.mean(axis=2, keepdims=True)
        std  = data.std(axis=2, keepdims=True)
        std[std < 1e-8] = 1e-8
        data = (data - mean) / std

        # Step 7: map labels
        raw_labels = epochs.events[:, 2]
        mapped = np.array([id_to_class.get(lbl, -1) for lbl in raw_labels])
        valid  = mapped >= 0
        data   = data[valid].astype(np.float32)
        mapped = mapped[valid]

        if len(data) > 0:
            xs.append(data)
            ys.append(mapped)

    return xs, ys


def preprocess_subject(subject_id):
    """
    Full preprocessing for one subject.
    Returns (X, y) or (None, None) on failure.
    """
    # Get file paths — download if not already cached
    hand_files = download_subject(subject_id, DATA_DIR)
    # Filter to only hand runs
    hand_files = [f for f in hand_files
                  if any(f"R{r:02d}" in os.path.basename(f) for r in HAND_RUNS)]

    feet_files = download_subject(subject_id, DATA_DIR)
    feet_files = [f for f in feet_files
                  if any(f"R{r:02d}" in os.path.basename(f) for r in FEET_RUNS)]

    # Hand runs: T0=rest, T1=left hand, T2=right hand
    xs_hand, ys_hand = _process_runs(
        hand_files,
        label_map={"T0": 0, "T1": 1, "T2": 2}
    )

    # Feet runs: T0=rest, T2=both feet (T1=both fists — skip by not including it)
    xs_feet, ys_feet = _process_runs(
        feet_files,
        label_map={"T0": 0, "T2": 3}
    )

    all_x = xs_hand + xs_feet
    all_y = ys_hand + ys_feet

    if not all_x:
        return None, None

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


def save_subject(subject_id, X, y):
    np.save(os.path.join(PROCESSED_DIR, f"S{subject_id:03d}_X.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, f"S{subject_id:03d}_y.npy"), y)


def load_subject(subject_id):
    x_path = os.path.join(PROCESSED_DIR, f"S{subject_id:03d}_X.npy")
    y_path = os.path.join(PROCESSED_DIR, f"S{subject_id:03d}_y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None, None
    return np.load(x_path), np.load(y_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else ALL_SUBJECTS
    print(f"Preprocessing {len(subjects)} subjects -> {PROCESSED_DIR}")

    success = skipped = failed = 0

    for subj in tqdm(subjects, desc="Subjects"):
        x_path = os.path.join(PROCESSED_DIR, f"S{subj:03d}_X.npy")
        if os.path.exists(x_path) and not args.force:
            skipped += 1
            continue

        X, y = preprocess_subject(subj)

        if X is None or len(X) == 0:
            tqdm.write(f"  [SKIP] S{subj:03d}: no valid epochs")
            failed += 1
            continue

        save_subject(subj, X, y)
        success += 1

        if success <= 3:
            unique, counts = np.unique(y, return_counts=True)
            tqdm.write(f"  S{subj:03d}: {len(X)} epochs | classes: {dict(zip(unique.tolist(), counts.tolist()))}")

    print(f"\nDone. Processed: {success} | Skipped (cached): {skipped} | Failed: {failed}")


if __name__ == "__main__":
    main()
