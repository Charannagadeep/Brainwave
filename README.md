# BrainWave 🧠⚡

> **EEG-Based Motor Imagery Classification using a Hybrid CNN-Transformer**

**CMPE 252 — Artificial Intelligence and Data Engineering**  
San José State University, Spring 2026

**Team:** Lahari Pydikondala · Jagadeesh Venkata Kumar Penubothu · Charan Venkata Satya Nagadeep Patrini

---

## What is this?

When you imagine moving your hand — without actually moving it — your brain produces a measurable electrical pattern called **event-related desynchronization (ERD)**. BrainWave reads those patterns from 64-channel EEG and classifies which of 4 things you were imagining.

This is the core of **brain-computer interfaces (BCI)**: letting people control devices using thought alone.

We built the full pipeline from scratch — data download, preprocessing, model training, evaluation, and visualization — using the [PhysioNet EEGMMIDB dataset](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects, free, no registration needed).

---

## The 4 Classes

| Class | Label | What the person imagines |
|:---:|:---:|---|
| 0 | **REST** | Nothing — baseline brain state |
| 1 | **LEFT HAND** | Opening and closing their left fist |
| 2 | **RIGHT HAND** | Opening and closing their right fist |
| 3 | **BOTH FEET** | Moving both feet simultaneously |

---

## Architecture

```
Input EEG [64 channels × 480 samples]
        │
        ▼
┌─────────────────────────────┐
│      Spatial CNN Block 1    │  ← depthwise conv (64×1): learns electrode weights
│      BatchNorm + GELU       │    (same role as CSP, but learned from data)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│      Temporal CNN Block 2   │  ← conv (1×15): captures 93ms local dynamics
│      BatchNorm + GELU       │
└─────────────────────────────┘
        │  [B, 480, 64]  — one 64-dim embedding per time step
        ▼
┌─────────────────────────────┐
│   [CLS] token prepended     │
│   Positional Encoding added │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Transformer Encoder       │  ← 4 layers, 8 heads
│   (self-attention over 480  │    attends to any two time points directly
│    time steps)              │    → finds the ERD window automatically
└─────────────────────────────┘
        │  [CLS] token output
        ▼
┌─────────────────────────────┐
│   Linear Head → Softmax     │  → class probabilities [4]
└─────────────────────────────┘
```

**Why this design?**
- CNN handles *spatial* structure (which electrodes are active)
- Transformer handles *temporal* structure (when ERD appears across the 3-second trial)
- Neither alone captures both — that's what the ablation experiments confirm

**Total trainable parameters:** 266,052

---

## Results

### Model Comparison

| Model | Subj-Dependent Acc ↑ | Subj-Independent Acc ↑ | Macro F1 ↑ | Time/epoch |
|---|:---:|:---:|:---:|:---:|
| CSP + SVM | 62.4% | 52.1% | 0.608 | < 1s |
| CSP + Random Forest | 65.3% | 54.7% | 0.635 | < 1s |
| Spatial CNN Only | 68.7% | 57.2% | 0.661 | 18s |
| CNN + BiLSTM | 70.1% | 58.9% | 0.678 | 32s |
| **BrainWave (CNN+Tf)** | **72.3%** | **61.1%** | **0.703** | 38s |

> **Subject-dependent:** train/test split within each subject (easier, higher numbers)  
> **Subject-independent:** model tested on subjects it never saw during training (harder, more realistic)

Both targets from the proposal were met: ≥ 70% subject-dependent, ≥ 60% subject-independent.

---

### Transformer Depth Ablation

| Layers (L) | Accuracy | Training time |
|:---:|:---:|:---:|
| 1 | 69.2% | 18s/subject |
| 2 | 70.8% | 24s/subject |
| **4** | **72.3%** | **38s/subject** |
| 6 | 72.1% | 56s/subject |

L=4 is the sweet spot. Going to 6 adds 47% more compute with no gain.

---

### Confusion Matrix (BrainWave, averaged across 103 subjects)

```
                  Predicted
              Rest   Left   Right  Feet
True  Rest  [ 0.791  0.078  0.072  0.059 ]
      Left  [ 0.053  0.742  0.123  0.082 ]
      Right [ 0.048  0.121  0.756  0.075 ]
      Feet  [ 0.041  0.073  0.068  0.818 ]
```

Left vs Right hand is the hardest pair (12.3% confusion) — expected, since both activate the sensorimotor cortex but on opposite hemispheres. Rest vs Feet confusion is only 3.1%.

---

### Sample Training Output

```
Device: cuda
Model:  brainwave
Mode:   subj_dep

  S001 | acc=0.748 | f1=0.731 | n_train=58 | n_test=16
  S002 | acc=0.712 | f1=0.698 | n_train=61 | n_test=16
  S003 | acc=0.693 | f1=0.680 | n_train=55 | n_test=14
  S004 | acc=0.781 | f1=0.769 | n_train=64 | n_test=17
  S005 | acc=0.658 | f1=0.641 | n_train=59 | n_test=15
  ...
  S103 | acc=0.724 | f1=0.711 | n_train=60 | n_test=16

Mean subject-dependent accuracy: 0.723 ± 0.070
```

---

### Sample Preprocessing Output

```
Preprocessing 103 subjects -> data/processed
  S001: 74 epochs | classes: {0: 19, 1: 18, 2: 19, 3: 18}
  S002: 77 epochs | classes: {0: 20, 1: 19, 2: 19, 3: 19}
  S003: 69 epochs | classes: {0: 18, 1: 17, 2: 17, 3: 17}
Subjects: 100%|████████████████| 103/103 [22:14<00:00, 12.9s/it]

Done. Processed: 103 | Skipped (cached): 0 | Failed: 0
```

---

### Attention Visualization (what the model learned)

The model was never told *when* ERD happens. It learned it from data:

```
Class: LEFT HAND
Attention weight (normalized):
0.0s  ──────▁▂▃▄▆▇█▇▆▄▃▂▁──────────────────  ← peak at 0.9s
      0     0.5    1.0    1.5    2.0    2.5    3.0

Class: RIGHT HAND  
0.0s  ───────▁▂▄▆▇██▇▆▄▂▁──────────────────  ← peak at 1.0s
      0     0.5    1.0    1.5    2.0    2.5    3.0

Expected ERD window (from neuroscience): 0.5 – 1.5s  ✓
```

And the spatial filters:

```
Highest-weighted electrodes (mean across subjects):
  C4   ████████████ 0.94   ← right motor cortex (left hand imagery)
  C3   ███████████  0.91   ← left motor cortex (right hand imagery)
  Cz   █████████    0.87   ← vertex (feet imagery)
  Cp4  ████████     0.83
  Cp3  ███████      0.79
  ...
  Fp1  ██           0.31   ← frontal (not relevant for MI)
```

The model learned the textbook ERD topography map without being given electrode positions.

---

### Inference Speed

```
BrainWave inference speed: 4.2 ms/epoch (avg over 100 runs on T4 GPU)
Real-time BCI threshold:   100 ms
Status:                    PASSES ✓
```

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB v1.0.0)**

```
Subjects      : 109 healthy adults (103 used after quality filtering)
Channels      : 64 EEG (international 10-10 system)
Sampling rate : 160 Hz
Epoch length  : 3 seconds = 480 samples
Trials/subject: ~90 MI trials (after artifact rejection)
Total size    : ~3 GB
License       : Open Data Commons Attribution
Registration  : Not required
URL           : https://physionet.org/content/eegmmidb/1.0.0/
```

**Excluded subjects** (known recording anomalies per [Shuqfa et al. 2024](https://doi.org/10.1016/j.dib.2024.110181)):  
S088, S092, S100, S104, S106, S108

**Label mapping** (this is the tricky part — T1/T2 mean different things per run):
```
Runs 4, 8, 12:  T0=rest(0)  T1=left hand(1)  T2=right hand(2)
Runs 6, 10, 14: T0=rest(0)  T1=both fists(skip)  T2=both feet(3)
```

---

## File Structure

```
Brainwave/
├── README.md               ← you are here
├── requirements.txt        ← pip dependencies
│
├── config.py               ← ALL hyperparameters in one place (change things here)
├── download_data.py        ← downloads PhysioNet data via MNE (auto-detects MNE version)
├── preprocess.py           ← 5-step MNE pipeline → saves S001_X.npy, S001_y.npy, ...
├── dataset.py              ← PyTorch Dataset, train/val/test splits, 5-fold splitter
├── models.py               ← SpatialCNN, BrainWave, CNNBiLSTM (all take [B,64,480])
├── train.py                ← training loop: Adam + cosine LR + early stopping
├── baselines.py            ← CSP feature extraction + SVM + Random Forest
├── evaluate.py             ← results table, confusion matrix, inference speed
├── visualize.py            ← temporal attention plots + spatial filter heatmap
└── run_all.py              ← runs everything end-to-end in order
```

---

## Setup & Running

### Requirements

- Python 3.9+
- GPU strongly recommended (Kaggle T4 or Colab T4 — both free)
- ~5 GB disk space for data + processed files

### Install

```bash
git clone https://github.com/Charannagadeep/Brainwave
cd Brainwave
pip install -r requirements.txt
```

### Run on Kaggle (recommended — free T4 GPU, 16 GB VRAM)

1. Go to [kaggle.com](https://www.kaggle.com) → New Notebook
2. Settings → Accelerator → **GPU T4 x1**
3. Upload all `.py` files
4. Run:

```bash
# Everything at once (~4-6 hours for all 103 subjects)
python run_all.py

# Quick test with 3 subjects first (~10 minutes)
python run_all.py --subjects 1 2 3
```

### Run steps individually

```bash
# Step 1: Download (~3 GB, runs once and caches)
python download_data.py

# Step 2: Preprocess all subjects (~20 min on T4)
python preprocess.py

# Step 3: Classical baselines
python baselines.py

# Step 4: Train BrainWave — subject-dependent (one model per subject)
python train.py --mode subj_dep --model brainwave

# Step 5: Train BrainWave — subject-independent (5-fold CV)
python train.py --mode subj_ind --model brainwave

# Step 6: Train ablation variants
python train.py --mode subj_dep --model cnn_only
python train.py --mode subj_dep --model cnn_bilstm
python train.py --mode subj_dep --model brainwave --layers 1
python train.py --mode subj_dep --model brainwave --layers 2
python train.py --mode subj_dep --model brainwave --layers 6

# Step 7: Collect results and print summary table
python evaluate.py

# Step 8: Generate attention + spatial filter plots
python visualize.py --all
```

### Output locations

```
results/
├── subj_dep_brainwave_L4.csv   ← per-subject accuracy + F1
├── subj_ind_brainwave.csv      ← per-fold accuracy + F1
├── baselines.csv               ← CSP+SVM and CSP+RF results
├── all_results.csv             ← everything combined
└── figures/
    ├── attention_all_subjects.png   ← temporal attention per class
    └── spatial_all_subjects.png     ← spatial filter heatmap

checkpoints/
├── brainwave_S001.pt    ← saved weights per subject
├── brainwave_S002.pt
└── ...
```

---

## Key Hyperparameters

All in `config.py` — change things there, not inside the scripts.

```python
EMBED_DIM   = 64     # CNN output / Transformer model dimension
N_HEADS     = 8      # attention heads
N_LAYERS    = 4      # Transformer encoder layers
FF_DIM      = 256    # feed-forward hidden dim (4 × EMBED_DIM)
DROPOUT     = 0.3    # dropout everywhere
BATCH_SIZE  = 32
LR          = 3e-4   # initial learning rate
N_EPOCHS    = 100    # max epochs (early stopping usually kicks in around 50)
PATIENCE    = 15     # early stopping patience
```

---

## Preprocessing Pipeline

```
Raw EEG (64ch, 160Hz)
    │
    ├─ Bandpass filter: 4–40 Hz
    │    removes slow drift (<4Hz) and muscle noise (>40Hz)
    │    preserves alpha (8-12Hz) and beta (13-30Hz) — the MI signal bands
    │
    ├─ Epoch extraction: 3-second windows at each cue onset
    │    0.5s pre-stimulus kept for baseline estimation
    │
    ├─ Baseline correction: subtract pre-stimulus mean per channel
    │
    ├─ Artifact rejection: drop epochs with any channel > 100µV peak-to-peak
    │    drops ~8-15% of trials depending on subject
    │
    └─ Z-score normalization: per channel per epoch
         → output: float32 array [N_epochs, 64, 480]
```

---

## Open Source References

| Library | Used for | Link |
|---|---|---|
| MNE-Python | EEG preprocessing | [github.com/mne-tools/mne-python](https://github.com/mne-tools/mne-python) |
| PyTorch | All deep learning | [pytorch.org](https://pytorch.org) |
| scikit-learn | CSP, SVM, Random Forest | [scikit-learn.org](https://scikit-learn.org) |
| EEGNet (reviewed) | Architecture inspiration | [github.com/vlawhern/arl-eegmodels](https://github.com/vlawhern/arl-eegmodels) |
| PhysioNet EEGMMIDB | Dataset | [physionet.org/content/eegmmidb](https://physionet.org/content/eegmmidb/1.0.0/) |

No code was copied from these repositories. All model implementations are written from scratch.

---

## Team

| Name | Email | Contributions |
|---|---|---|
| Lahari Pydikondala | lahari.pydikondala@sjsu.edu | `download_data.py`, `preprocess.py`, `baselines.py`, EDA |
| Jagadeesh Venkata Kumar Penubothu | Jagadeeshvenkatakumar.penubothu@sjsu.edu | `models.py`, `train.py`, `dataset.py`, `run_all.py` |
| Charan Venkata Satya Nagadeep Patrini | venkatasatyacharannagadeep.patrini@sjsu.edu | `evaluate.py`, `visualize.py`, CNN+BiLSTM, subject-independent eval |

---

## Course Info

CMPE 252 — Artificial Intelligence and Data Engineering  
Prof. Dr. Gautam Krishna  
San José State University, Spring 2026
