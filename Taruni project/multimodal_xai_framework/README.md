# 🧠 Multi-Modal Explainable AI (XAI) Framework for Neurodegenerative Diagnosis

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)

A state-of-the-art **Dual-Stream Multi-Modal Fusion Deep Learning System** designed to bridge the clinical trust gap in neurodegenerative diagnostic decision support (e.g., Alzheimer's Disease and Mild Cognitive Impairment).

---

## 🏗️ System Architecture

```
                                    ┌────────────────────────┐
                                    │    Brain MRI Scans     │
                                    └───────────┬────────────┘
                                                │
                                                ▼
                                    [ ResNet-18 Image Stream ] ── (512d)
                                                │                    │
                                                ▼                    ▼
   ┌────────────────────────┐        ┌────────────────────┐    [ Multi-Modal ] ──► Diagnosis Output
   │ Clinical Metadata      │ ─────► │ MLP Clinical Stream│ ── (64d) Fusion Head  (CN / MCI / AD)
   │ (Age, MMSE, CDR, nWBV) │        └────────────────────┘
   └────────────────────────┘
                │
                ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │                     Explainable AI (XAI) Engine                        │
   ├───────────────────────────────────┬────────────────────────────────────┤
   │  Grad-CAM (Visual Brain Heatmaps) │ SHAP (Clinical Feature Impact)     │
   └───────────────────────────────────┴────────────────────────────────────┘
```

---

## 📂 Project Structure

```
multimodal_xai_framework/
├── config.py                 # Configuration settings, features & hyperparams
├── requirements.txt           # Python dependencies
├── train.py                   # Script to train the dual-stream model
├── data/
│   ├── synthetic_generator.py # Generates MRI scans & OASIS clinical data
│   ├── dataset.py             # PyTorch Dataset loader & scaler
│   └── clinical_data.csv      # CSV metadata
├── models/
│   ├── image_stream.py        # CNN / ResNet-18 feature extractor
│   ├── clinical_stream.py     # MLP tabular feature extractor
│   └── fusion_network.py      # Feature concatenation & classification head
├── explainability/
│   ├── gradcam.py             # Grad-CAM activation heatmap generator
│   └── shap_explainer.py      # SHAP feature importance attributions
├── app/
│   └── main.py                # FastAPI server REST endpoints
├── static/
│   ├── index.html             # Glassmorphic Web Dashboard UI
│   ├── style.css              # Dark mode styling & layout
│   └── app.js                 # Interactive JavaScript logic
└── saved_models/              # Trained weights (.pth) & scaler (.pkl)
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Dual-Stream Model
Run `train.py` to generate the synthetic OASIS-style multi-modal dataset and train the fusion network:
```bash
python train.py
```

### 3. Launch the Clinical Transparency Dashboard
Start the FastAPI server:
```bash
python app/main.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 🌟 Key Features

1. **Dual-Stream Feature Fusion:** Combines visual spatial representations from brain scans with numeric clinical scores.
2. **Grad-CAM Visual Explanations:** Overlays color-coded activation heatmaps on MRI scans highlighting anatomical region impact.
3. **SHAP Feature Attributions:** Breaks down the exact risk increase or decrease associated with clinical indicators (MMSE score, CDR, Age, brain volume ratio).
4. **Interactive Dashboard UI:** Modern dark-themed dashboard with sample patient preset buttons, image uploader, and diagnostic confidence progress bars.
