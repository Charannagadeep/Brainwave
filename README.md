# BrainWave: EEG-Based Motor Imagery Classification

**CMPE 252 — Artificial Intelligence and Data Engineering**  
San José State University, Spring 2026  
Lahari Pydikondala · Jagadeesh Venkata Kumar Penubothu · Charan Venkata Satya Nagadeep Patrini

---

## What this project does

Classifies 4 motor imagery tasks (rest, left hand, right hand, both feet) from raw
64-channel EEG recordings using a hybrid CNN-Transformer architecture trained on the
PhysioNet EEGMMIDB dataset (109 subjects, free download, no registration).

---

## Project structure

```
brainwave/
├── README.md
├── requirements.txt
├── config.py              # all hyperparameters in one place
├── download_data.py       # downloads PhysioNet data via MNE
├── preprocess.py          # full MNE preprocessing pipeline
├── dataset.py             # PyTorch Dataset wrapper
├── models.py              # SpatialCNN, BrainWave, BiLSTM variants
├── train.py               # training loop with early stopping
├── baselines.py           # CSP + SVM / Random Forest pipelines
├── evaluate.py            # subject-dependent and subject-independent eval
├── visualize.py           # attention weights + spatial filter plots
├── run_all.py             # runs the full experiment pipeline end-to-end
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run on Kaggle (recommended — free T4 GPU)

1. Go to https://www.kaggle.com and create a new notebook
2. Enable GPU: Settings → Accelerator → GPU T4
3. Upload all `.py` files from this repo
4. Run in order:

```python
!python download_data.py          # downloads ~3 GB of EDF files
!python preprocess.py             # preprocesses all 103 subjects, saves .npy
!python baselines.py              # CSP+SVM and CSP+Random Forest
!python train.py --mode subj_dep  # trains BrainWave per subject (subject-dependent)
!python train.py --mode subj_ind  # 5-fold cross-validation (subject-independent)
!python evaluate.py               # collects all results into results/summary.csv
!python visualize.py              # generates attention + spatial filter plots
```

Or just run everything at once:
```python
!python run_all.py
```

### 3. Run on Google Colab (backup)

Same as Kaggle. Runtime → Change runtime type → GPU (T4).

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB v1.0.0)**  
- 109 subjects, 64 channels, 160 Hz  
- Free, no registration: https://physionet.org/content/eegmmidb/1.0.0/  
- License: Open Data Commons Attribution  
- 6 subjects excluded (recording anomalies): S088, S092, S100, S104, S106, S108  
- Total size: ~3 GB

The dataset downloads automatically when you run `download_data.py`.

---

## Results summary

| Model | Subj-Dependent Acc | Subj-Independent Acc | Macro F1 |
|---|---|---|---|
| CSP + SVM | 62.4% | 52.1% | 0.608 |
| CSP + Random Forest | 65.3% | 54.7% | 0.635 |
| Spatial CNN Only | 68.7% | 57.2% | 0.661 |
| CNN + BiLSTM | 70.1% | 58.9% | 0.678 |
| **BrainWave (CNN+Tf)** | **72.3%** | **61.1%** | **0.703** |

---

## Reproducing results

Everything needed to reproduce results from raw download to final numbers:

```bash
# Full pipeline (takes ~4-6 hours on a T4 for all 103 subjects)
python run_all.py

# Or run individual steps and check results interactively
python download_data.py
python preprocess.py
python baselines.py
python train.py --mode subj_dep --subjects all
python evaluate.py
```

Results land in `results/summary.csv`. Figures land in `results/figures/`.

---

## Open source references

- MNE-Python: https://github.com/mne-tools/mne-python (preprocessing)
- EEGNet (reviewed for architecture inspiration): https://github.com/vlawhern/arl-eegmodels
- PhysioNet dataset: https://physionet.org/content/eegmmidb/1.0.0/
- PyTorch: https://pytorch.org
- scikit-learn: https://scikit-learn.org
