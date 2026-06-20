# 🛰️ AI-Based Satellite Anomaly Detection

> Unsupervised spacecraft health monitoring using Transformer-based time-series forecasting on NASA telemetry data.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?logo=pytorch)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**BMS College of Engineering · Department of Machine Learning · 2025-26**
Sahana BK · Sarayu KS · Shamratha G | Guide: Sowmya Lakshmi BS


---

## 📌 Overview

Modern satellites generate terabytes of multivariate telemetry data daily. Traditional threshold-based monitoring fails to catch subtle, multivariate, or early-stage faults. This project builds an **unsupervised anomaly detection system** using **PatchTST (Patch Time Series Transformer)** that learns normal spacecraft behavior and flags deviations in real time — without requiring labeled fault data for training.

The system is trained and evaluated on **NASA's SMAP and MSL telemetry datasets**, which contain real spacecraft sensor readings with expert-labeled anomaly intervals.

---

## ✨ Key Features

- 🧠 **Unsupervised learning** — trains only on normal operational data
- 🛰️ **Per-channel modeling** — one PatchTST model per telemetry channel for higher precision
- 📊 **Automatic thresholding** — optimal decision boundary via precision-recall curve optimization
- 🌐 **Full-stack web app** — Flask backend + interactive dashboard for live exploration
- 📤 **Upload & Predict** — test your own `.npy` telemetry files instantly
- 📈 **Rich visualizations** — anomaly timelines, ROC curves, confusion matrices, per-channel breakdowns

---

## 📊 Results Summary

| Metric    | Average (82 channels) | Best Channel (M-6) |
|-----------|------------------------|---------------------|
| AUC-ROC   | 0.882                  | 0.954                |
| F1-Score  | 0.689                  | 0.920                |
| Accuracy  | 0.957                  | 0.985                |
| AUC-PR    | 0.541                  | 0.849                |

- **14 / 82 channels** achieved AUC-ROC ≥ 0.90
- Model trained in **5–15 epochs** per channel with early stopping
- Inference is fast enough for near-real-time monitoring use cases

---

## 🏗️ System Architecture

```
Raw Telemetry (.npy)
        │
        ▼
 StandardScaler (per-channel normalization)
        │
        ▼
 Sliding Window Generation
   (context=96, horizon=24, patch=16)
        │
        ▼
 PatchTST Encoder
   ├─ Patch Embedding
   ├─ Learnable Positional Encoding
   ├─ Transformer Encoder (3 layers, 4 heads)
   └─ Linear Prediction Head
        │
        ▼
 Forecasting Error → Anomaly Score
        │
        ▼
 Precision-Recall Threshold → Binary Anomaly Label
        │
        ▼
 Flask Web Dashboard (visualization + live inference)
```

---

## 🗂️ Repository Structure

```
satellite-anomaly-detection/
├── app.py                  # Flask web application
├── model.py                # PatchTST architecture
├── pipeline.py              # Per-channel training + evaluation
├── visualize.py             # Report-quality plot generation
├── templates/
│   └── index.html           # Web dashboard frontend
├── models/                   # Trained model checkpoints (.pt)
├── results/                  # Evaluation CSVs
├── data/                     # NASA SMAP/MSL dataset (not tracked in git)
├── requirements.txt
├── Procfile                  # For Render/Heroku deployment
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU for faster training

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/satellite-anomaly-detection.git
cd satellite-anomaly-detection
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Dataset

Download the **NASA SMAP/MSL Anomaly Detection Dataset** from Kaggle:
🔗 https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl

Place files as:
```
data/train/*.npy
data/test/*.npy
data/labeled_anomalies.csv
```

### Train Models

```bash
python pipeline.py
```
Trains one PatchTST model per channel (~30 min on a T4 GPU for all 82 channels).

### Generate Visualizations

```bash
python visualize.py
```

### Run the Web App

```bash
python app.py
```
Visit **http://localhost:5000**

---

## 🖥️ Web Dashboard

| Tab | Functionality |
|-----|----------------|
| 📊 **Dashboard** | Sortable/searchable results table across all channels with summary statistics |
| 🔍 **Channel** | Deep-dive into any channel: anomaly score timeline, ROC curve, confusion matrix |
| 📤 **Predict** | Upload your own `.npy` telemetry file and get an instant anomaly report |
| ℹ️ **About** | Architecture, dataset, and team details |


<img width="1751" height="881" alt="Screenshot 2026-06-20 123327" src="https://github.com/user-attachments/assets/ebd2da5c-68e6-4381-aa20-7838be64ac72" />
<img width="1463" height="820" alt="Screenshot 2026-06-20 123340" src="https://github.com/user-attachments/assets/a91be441-d808-4586-a7fa-62e6a3decbe9" />
<img width="1538" height="901" alt="Screenshot 2026-06-20 123405" src="https://github.com/user-attachments/assets/440fedb2-a3cd-4279-a96c-53848a6c11b3" />
<img width="1450" height="757" alt="Screenshot 2026-06-20 123416" src="https://github.com/user-attachments/assets/41b09a69-9d34-49cc-8012-1c2975bbb00a" />
<img width="1521" height="471" alt="Screenshot 2026-06-20 123427" src="https://github.com/user-attachments/assets/1dad7042-5a35-43fa-ab7d-bdcfd335d3c7" />
<img width="1496" height="807" alt="Screenshot 2026-06-20 123446" src="https://github.com/user-attachments/assets/18c58563-b0aa-4950-b766-6a4a25793187" />
<img width="1500" height="874" alt="image" src="https://github.com/user-attachments/assets/07e4964b-73d7-48b6-949b-8ffb97935d7e" />



---

## 🔬 Methodology

1. **Preprocessing** — per-channel z-score normalization using `StandardScaler`, fit only on training (normal) data
2. **Windowing** — sliding windows of length 96 with 24-step forecast horizon, stride 1
3. **Training** — `SmoothL1Loss`, `AdamW` optimizer, cosine annealing LR schedule, early stopping (patience=5)
4. **Anomaly Scoring** — max absolute forecasting error across the prediction horizon
5. **Thresholding** — threshold chosen to maximize F1 via the precision-recall curve
6. **Evaluation** — AUC-ROC, AUC-PR, F1, Accuracy computed per channel

---

## 📚 Related Work

This project builds on and benchmarks against:
- Nie et al., *PatchTST: A Time Series is Worth 64 Words*, ICLR 2023
- Hundman et al., *Detecting Spacecraft Anomalies Using LSTMs*, KDD 2018
- ESA OPS-SAT Anomaly Detection Benchmark, 2024

A full literature review and gap analysis is included in the accompanying project report.

---

## 🛣️ Future Work

- Multi-modal anomaly detection (combining telemetry with system logs/images)
- Onboard real-time deployment via model compression/quantization
- Explainability via SHAP and attention visualization
- Cross-mission generalization through transfer learning
- Synthetic anomaly generation using GANs/VAEs for training augmentation

---

## 👩‍💻 Team

| Name | USN |
|------|-----|
| Sahana BK | 1BM23AI162 |
| Sarayu KS | 1BM23AI171 |
| Shamratha G | 1BM23AI173 |

**Guide:** Sowmya Lakshmi BS, Dept. of Machine Learning, BMSCE
**HOD:** Dr. M. Dakshayini

---

## 📄 Citation

```bibtex
@misc{satad2025,
  title   = {AI-Based Satellite Anomaly Detection for Spacecraft Health Monitoring},
  author  = {Sahana B K and Sarayu K S and Shamratha G},
  year    = {2025},
  school  = {BMS College of Engineering, Bengaluru},
  note    = {Department of Machine Learning, VTU}
}
```

---

## 📝 License

This project is licensed under the MIT License — free for academic and research use.
