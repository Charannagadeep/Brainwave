"""
visualize.py
------------
Generates two interpretability plots for BrainWave:

  1. Temporal attention plot
     For each of the 4 classes, extract the CLS token's attention weights
     over time from correctly classified test epochs and plot the average.
     Expected: attention peaks in 500-1500 ms post-cue (ERD window).

  2. Spatial filter heatmap
     Project the first CNN block's filter weights onto a bar chart ordered
     by electrode position. Expected: C3/C4 weighted heavily for left/right
     hand; Cz for feet.

All plots saved to results/figures/.

Usage:
    python visualize.py --subject 1
    python visualize.py --all          # average across all subjects
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless rendering (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALL_SUBJECTS, DEVICE, CHECKPOINT_DIR, FIGURES_DIR,
    CLASS_NAMES, N_TIMES, SFREQ,
)
from dataset import load_subject_data, subject_dependent_split, make_loader
from models import BrainWave

os.makedirs(FIGURES_DIR, exist_ok=True)

# PhysioNet 64-channel names in order (international 10-20 / 10-10 extension)
CHANNEL_NAMES = [
    'Fc5','Fc3','Fc1','Fcz','Fc2','Fc4','Fc6',
    'C5','C3','C1','Cz','C2','C4','C6',
    'Cp5','Cp3','Cp1','Cpz','Cp2','Cp4','Cp6',
    'Fp1','Fpz','Fp2','Af7','Af3','Afz','Af4','Af8',
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'Ft7','Ft8','T7','T8','T9','T10',
    'Tp7','Tp8','P7','P5','P3','P1','Pz','P2','P4','P6','P8',
    'Po7','Po3','Poz','Po4','Po8','O1','Oz','O2','Iz',
]
# Highlight these electrodes (motor cortex)
HIGHLIGHT_CHANNELS = {"C3", "Cz", "C4", "Cp3", "Cpz", "Cp4"}


# Load model
def load_brainwave(subject_id: int) -> BrainWave:
    """Load saved BrainWave checkpoint for one subject."""
    ckpt = os.path.join(CHECKPOINT_DIR, f"brainwave_S{subject_id:03d}.pt")
    if not os.path.exists(ckpt):
        return None
    model = BrainWave()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval().to(DEVICE)
    return model


# 1. Temporal attention plot
def extract_attention_per_class(
    model: BrainWave,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_per_class: int = 50,
) -> dict:
    """
    For each class, collect the CLS token's averaged attention weights
    over correctly classified test epochs.

    Returns dict: class_label → attention_curve [N_TIMES]
    """
    model.eval()
    class_attn = {c: [] for c in range(4)}

    loader = make_loader(X_test, y_test, shuffle=False)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits, attn_list = model.get_attention_weights(xb)
            preds = logits.argmax(dim=1).cpu()

            for i in range(len(yb)):
                true_label = yb[i].item()
                pred_label = preds[i].item()
                if true_label != pred_label:
                    continue  # only correctly classified

                # Average attention weights across all layers
                # Each attn: [B, n_heads, L+1, L+1]
                # CLS row: [B, n_heads, 0, :] → attention from CLS to all tokens
                attn_curve = torch.stack(
                    [a[i].mean(dim=0)[0, 1:]  # [L] (skip CLS position itself)
                     for a in attn_list]
                ).mean(dim=0)  # average over layers → [N_TIMES]

                attn_curve = attn_curve.cpu().numpy()

                if len(class_attn[true_label]) < max_per_class:
                    class_attn[true_label].append(attn_curve)

    # Average within each class
    result = {}
    for c in range(4):
        if class_attn[c]:
            result[c] = np.mean(class_attn[c], axis=0)
    return result


def plot_temporal_attention(
    attn_by_class: dict,
    subject_id: object,
    save_path: str,
):
    """Plot CLS attention over time for each class."""
    time_axis = np.linspace(0, 3.0, N_TIMES)

    colors = ["#2196F3", "#4CAF50", "#F44336", "#9C27B0"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=False)
    axes = axes.flatten()

    for c in range(4):
        ax = axes[c]
        if c not in attn_by_class:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(CLASS_NAMES[c])
            continue

        curve = attn_by_class[c]
        # Smooth with a simple moving average for readability
        kernel = np.ones(20) / 20
        smoothed = np.convolve(curve, kernel, mode="same")

        ax.plot(time_axis, smoothed, color=colors[c], linewidth=2, label=CLASS_NAMES[c])
        ax.fill_between(time_axis, 0, smoothed, alpha=0.15, color=colors[c])

        # Shade expected ERD window
        ax.axvspan(0.5, 1.5, alpha=0.12, color="gray", label="Expected ERD (0.5–1.5s)")

        ax.set_title(CLASS_NAMES[c], fontsize=12, fontweight="bold")
        ax.set_ylabel("Attention weight")
        ax.set_xlabel("Time after cue (s)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 3)
        ax.grid(True, alpha=0.3)

    subj_str = f"Subject {subject_id}" if isinstance(subject_id, int) else subject_id
    fig.suptitle(
        f"BrainWave: CLS Token Temporal Attention by Class\n({subj_str})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention plot: {save_path}")


# 2. Spatial filter plot
def plot_spatial_filters(model: BrainWave, subject_id: object, save_path: str):
    """
    Visualize the CNN spatial filter weights as a bar chart over electrode index.
    Highlights motor cortex electrodes.
    """
    with torch.no_grad():
        # The first Conv2d in the spatial block has weight shape [d, 1, n_channels, 1]
        conv_weight = model.cnn.spatial[0].weight  # [d, 1, 64, 1]
        # Take the absolute mean across output filters (d)
        filter_importance = conv_weight.squeeze().abs().mean(dim=0).cpu().numpy()
        # filter_importance: [64]

    # Normalize
    filter_importance = filter_importance / (filter_importance.max() + 1e-8)

    ch_names = CHANNEL_NAMES[:len(filter_importance)]

    # Color by whether the electrode is in the motor cortex highlight set
    colors = [
        "#E53935" if ch.upper() in {c.upper() for c in HIGHLIGHT_CHANNELS}
        else "#90A4AE"
        for ch in ch_names
    ]

    fig, ax = plt.subplots(figsize=(18, 4))
    bars = ax.bar(range(len(ch_names)), filter_importance, color=colors, width=0.7)

    ax.set_xticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax.set_ylabel("Normalized filter importance\n(mean abs weight across filters)")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    # Legend
    motor_patch  = mpatches.Patch(color="#E53935", label="Motor cortex (C3/Cz/C4)")
    other_patch  = mpatches.Patch(color="#90A4AE", label="Other electrodes")
    ax.legend(handles=[motor_patch, other_patch], fontsize=10)

    subj_str = f"Subject {subject_id}" if isinstance(subject_id, int) else subject_id
    ax.set_title(
        f"BrainWave Spatial CNN Filter Importance by Electrode ({subj_str})",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spatial filter plot: {save_path}")


# Main
def visualize_subject(subject_id: int):
    """Generate both plots for one subject."""
    model = load_brainwave(subject_id)
    if model is None:
        print(f"No checkpoint for S{subject_id:03d} — skipping.")
        return

    X, y = load_subject_data(subject_id)
    if X is None:
        print(f"No data for S{subject_id:03d} — skipping.")
        return

    _, _, _, _, X_test, y_test = subject_dependent_split(X, y)
    if len(X_test) == 0:
        return

    # Temporal attention
    attn = extract_attention_per_class(model, X_test, y_test)
    plot_temporal_attention(
        attn, subject_id,
        os.path.join(FIGURES_DIR, f"attention_S{subject_id:03d}.png")
    )

    # Spatial filters
    plot_spatial_filters(
        model, subject_id,
        os.path.join(FIGURES_DIR, f"spatial_S{subject_id:03d}.png")
    )


def visualize_all():
    """Average attention weights across all subjects and plot once."""
    all_attn = {c: [] for c in range(4)}
    all_models = []

    subjects_with_ckpt = [
        s for s in ALL_SUBJECTS
        if os.path.exists(os.path.join(CHECKPOINT_DIR, f"brainwave_S{s:03d}.pt"))
    ]

    if not subjects_with_ckpt:
        print("No checkpoints found. Run training first.")
        return

    print(f"Averaging attention across {len(subjects_with_ckpt)} subjects ...")

    for subj in subjects_with_ckpt[:30]:  # cap for speed
        model = load_brainwave(subj)
        if model is None:
            continue
        X, y = load_subject_data(subj)
        if X is None:
            continue
        _, _, _, _, X_test, y_test = subject_dependent_split(X, y)
        if len(X_test) == 0:
            continue

        attn = extract_attention_per_class(model, X_test, y_test)
        for c, curve in attn.items():
            all_attn[c].append(curve)

        all_models.append(model)

    mean_attn = {c: np.mean(curves, axis=0) for c, curves in all_attn.items() if curves}

    plot_temporal_attention(
        mean_attn, "Average across all subjects",
        os.path.join(FIGURES_DIR, "attention_all_subjects.png")
    )

    # Spatial filter: average across models
    if all_models:
        all_weights = []
        for m in all_models:
            w = m.cnn.spatial[0].weight.squeeze().abs().mean(dim=0).cpu().numpy()
            all_weights.append(w / (w.max() + 1e-8))
        mean_weights = np.mean(all_weights, axis=0)

        # Temporarily patch one model's weights to reuse plot function
        dummy_model = all_models[0]
        dummy_model.cnn.spatial[0].weight.data = torch.from_numpy(
            mean_weights[None, None, :, None]
        ).repeat(64, 1, 1, 1).to(DEVICE)
        plot_spatial_filters(
            dummy_model, "Average across all subjects",
            os.path.join(FIGURES_DIR, "spatial_all_subjects.png")
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize BrainWave attention")
    parser.add_argument("--subject", type=int, default=None,
                        help="Single subject to visualize")
    parser.add_argument("--all", action="store_true",
                        help="Average across all subjects")
    args = parser.parse_args()

    if args.subject:
        visualize_subject(args.subject)
    elif args.all:
        visualize_all()
    else:
        # Default: visualize first subject with a checkpoint
        subjects_with_ckpt = [
            s for s in ALL_SUBJECTS
            if os.path.exists(os.path.join(CHECKPOINT_DIR, f"brainwave_S{s:03d}.pt"))
        ]
        if subjects_with_ckpt:
            visualize_subject(subjects_with_ckpt[0])
        else:
            print("No checkpoints found. Run: python train.py --mode subj_dep")


if __name__ == "__main__":
    main()
