"""
config.py
---------
All project hyperparameters and paths live here.
Change things here rather than hunting through multiple files.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
FIGURES_DIR     = os.path.join(RESULTS_DIR, "figures")

for d in [DATA_DIR, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
# PhysioNet subjects to use (exclude 6 with known recording anomalies)
BAD_SUBJECTS = {88, 92, 100, 104, 106, 108}
ALL_SUBJECTS = [s for s in range(1, 110) if s not in BAD_SUBJECTS]  # 103 subjects

# Runs that contain motor imagery
# Runs 4, 8, 12: T1 = left hand, T2 = right hand
# Runs 6, 10, 14: T1 = both fists, T2 = both feet
# We use left hand (run 4,8,12 T1) and right hand (T2) + rest (T0)
# AND both feet (run 6,10,14 T2) → 4 classes total
HAND_RUNS = [4, 8, 12]   # T1=left hand, T2=right hand
FEET_RUNS = [6, 10, 14]  # T1=both fists (skip), T2=both feet
ALL_MI_RUNS = HAND_RUNS + FEET_RUNS

# Label mapping
# 0 = rest, 1 = left hand, 2 = right hand, 3 = both feet
CLASS_NAMES = ["Rest", "Left Hand", "Right Hand", "Both Feet"]
N_CLASSES = 4

# ── Preprocessing ─────────────────────────────────────────────────────────────
SFREQ           = 160          # Hz — PhysioNet native sample rate
L_FREQ          = 4.0          # bandpass low cutoff (Hz)
H_FREQ          = 40.0         # bandpass high cutoff (Hz)
TMIN            = -0.5         # epoch start relative to cue (seconds)
TMAX            = 3.0          # epoch end relative to cue (seconds)
BASELINE        = (None, 0)    # baseline window for correction
ARTIFACT_THRESH = 100e-6       # peak-to-peak amplitude rejection threshold (V)
N_CHANNELS      = 64           # number of EEG channels
N_TIMES         = 480          # time samples per epoch (3s × 160Hz)

# ── Model ─────────────────────────────────────────────────────────────────────
EMBED_DIM       = 64           # CNN output / Transformer model dimension (d)
N_HEADS         = 8            # Transformer attention heads
N_LAYERS        = 4            # Transformer encoder layers (L)
FF_DIM          = 256          # Transformer feed-forward hidden dim (4*d)
DROPOUT         = 0.3          # dropout probability throughout
MAX_SEQ_LEN     = 600          # positional encoding table size (> N_TIMES+1)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
LR              = 3e-4         # initial learning rate
WEIGHT_DECAY    = 1e-4
N_EPOCHS        = 100
PATIENCE        = 15           # early stopping patience (epochs on val loss)
SEED            = 42

# ── Evaluation ───────────────────────────────────────────────────────────────
TRAIN_RATIO     = 0.8          # subject-dependent train/test split
VAL_RATIO       = 0.1          # of training data → validation
N_FOLDS         = 5            # subject-independent cross-validation folds

# ── CSP baseline ─────────────────────────────────────────────────────────────
CSP_COMPONENTS  = 8            # number of CSP spatial filters

# ── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
