# EMG-Based Gait Classification: Walking vs Running

A machine learning pipeline for classifying human gait activity (walking vs running) from multi-channel surface EMG signals, evaluated under a rigorous **Leave-One-Subject-Out (LOSO)** cross-validation protocol.

---

## Visual Overview

The figure below shows a single 1-second window of preprocessed EMG from all 10 muscle channels for the same subject — walking (blue) vs running (red). The amplitude and burst density differences are clearly visible, particularly in the Rectus Femoris, Biceps Femoris, and Gastrocnemius channels.

![Walking vs Running EMG](walk_vs_run_p1.png)

---

## Dataset

Recordings were collected from **11 subjects** (p1–p11) using a wireless EMG system at **2000 Hz** sampling rate. Each subject performed one ~75 second walking session and one ~80 second running session.

### EMG Channels (10 muscles)

| Index | Muscle | Side |
|---|---|---|
| 0 | Tibialis Anterior | Right |
| 1 | Tibialis Anterior | Left |
| 2 | Rectus Femoris | Right |
| 3 | Rectus Femoris | Left |
| 4 | Biceps Femoris | Left |
| 5 | Biceps Femoris | Right |
| 6 | Gluteus Medius | Right |
| 7 | Gluteus Medius | Left |
| 8 | Medial Gastrocnemius | Left |
| 9 | Medial Gastrocnemius | Right |

Each session also has a paired `.insoleX` file containing bilateral foot pressure insole data (Zebris format).

---

## Pipeline

### 1. Preprocessing
- **Bandpass filter**: 4th-order Butterworth, 20–450 Hz
- **Notch filter**: 50 Hz power line interference removal (Q=30)
- Applied per-channel via zero-phase `filtfilt`

### 2. Segmentation
- **Fixed sliding window**: 1.0 second (2000 samples), 50% overlap
- Produces ~290–324 windows per subject (~3300 total across 11 subjects)
- Near-perfect class balance: ~50% walk / ~50% run per subject

### 3. Feature Extraction (for classical models)
6 features × 10 channels = **60-dimensional feature vector** per window:
- Time domain: MAV, RMS, Variance, Zero Crossing Rate
- Frequency domain: Total Power, Mean Frequency (20–450 Hz band)

### 4. Evaluation
**Leave-One-Subject-Out (LOSO)** cross-validation — train on 10 subjects, test on the held-out subject, repeat for all 11. This is the standard protocol for assessing cross-subject generalization in wearable EMG systems.

---

## Models & Results

| Model | Mean Accuracy | Std |
|---|---|---|
| Random Forest (60-dim features) | 77.9% | ±18.1% |
| MLP (60-dim features) | 63.6% | ±10.0% |
| 1D CNN (raw windows) | 93.1% | ±11.6% |
| Residual CNN (raw windows) | **93.7%** | ±9.2% |
| CNN + BiGRU (raw windows) | 92.5% | ±9.7% |

Key finding: models operating on **raw waveforms** dramatically outperform those using handcrafted features under cross-subject evaluation. The Residual CNN achieves the best mean accuracy and lowest variance, suggesting that learned temporal filters generalize better across subjects than amplitude-based statistics.

### Per-Subject Breakdown (Residual CNN)

| Subject | Accuracy | F1 |
|---|---|---|
| p1 | 99.7% | 99.7% |
| p2 | 95.4% | 95.4% |
| p3 | 99.3% | 99.4% |
| p4 | 91.0% | 91.6% |
| p5 | 93.1% | 93.5% |
| p6 | 98.1% | 98.1% |
| p7 | 97.2% | 97.3% |
| p8 | 99.0% | 99.1% |
| p9 | 94.5% | 93.9% |
| p10 | 67.3% | 67.3% |
| p11 | 96.3% | 96.5% |

p10 remains the hardest subject across all models, with consistently lower accuracy. RMS analysis revealed p10 has anomalously high signal amplitude (454 µV mean vs ~175 µV median across subjects), suggesting atypical electrode contact or placement.

---

## Feature Importance Analysis

Random Forest feature importances revealed that the most discriminative muscles for walk/run classification are:

1. **L. Biceps Femoris** — MAV, RMS, Power (ranks 1–3)
2. **R. Medial Gastrocnemius** — Power, MAV
3. **L. Gluteus Medius** — Power, MAV, Variance

Notably, Tibialis Anterior and Rectus Femoris — the most commonly instrumented muscles in gait studies — ranked lower in discriminative importance for this binary task.

---

## Repository Structure

```
.
├── data/                    # Raw .c3d and .insoleX files per subject
├── notebooks/
│   └── gait_classification.ipynb   # Full pipeline notebook
├── walk_vs_run_p1.png       # Example visualization
├── walk_vs_run_p10.png      # Hard subject visualization
└── README.md
```

---

## Requirements

```
ezc3d
numpy
scipy
scikit-learn
torch
matplotlib
pandas
```

Install with:
```bash
pip install ezc3d numpy scipy scikit-learn torch matplotlib pandas
```

---

## Usage

Open `notebooks/gait_classification.ipynb` in Google Colab or Jupyter. Set `DATA_DIR` to your local data folder path and run all cells sequentially. The notebook will load all subjects, preprocess, train all models under LOSO, and print the results table.
