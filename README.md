# ECG Arrhythmia Classifier

A machine learning pipeline for classifying cardiac arrhythmias from ECG signals, comparing a Random Forest baseline against a 1D Convolutional Neural Network.

## Motivation

Early and accurate detection of arrhythmias can significantly support clinical triage. This project explores whether ML models can reliably distinguish normal heartbeats from abnormal ones — catching patterns that might be missed in manual review.

## Dataset

**MIT-BIH Arrhythmia Database** (PhysioNet)
- 6 recordings used: 100, 101, 106, 200, 208, 213
- 15,359 annotated heartbeats extracted
- Binary classification: Normal (N) vs Abnormal (V, A, F)
- Balanced dataset: 6,772 beats (3,386 per class)

## Pipeline

1. **Data Loading** — raw ECG signals loaded via `wfdb`
2. **Exploration** — signal visualization, PQRST analysis
3. **Preprocessing** — R-peak extraction, windowing (216 samples), label filtering, class balancing via undersampling
4. **Modeling** — Random Forest and 1D CNN trained and evaluated
5. **Evaluation** — accuracy, precision, recall, confusion matrix

## Results

| Model | Accuracy | Abnormal Recall | False Normals |
|---|---|---|---|
| Random Forest | 98.0% | 0.96 | 29 |
| 1D CNN | 97.5% | 0.95 | 34 |

The Random Forest achieved slightly higher overall accuracy. The 1D CNN produced zero false alarms but missed more arrhythmias. Both models show different clinical tradeoffs worth considering.

## Tech Stack

- Python 3.12
- PyTorch 2.2
- scikit-learn
- wfdb
- numpy, pandas, matplotlib

## How to Run
```bash
git clone https://github.com/ersandenizhanzorlutuna-cpu/ecg-arrhythmia-classifier
cd ecg-arrhythmia-classifier
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/ecg_classifier.ipynb` and run all cells.

## Author

Denizhan — Biomedical Engineering (BSc) | Software Engineering (MSc) | Berlin
