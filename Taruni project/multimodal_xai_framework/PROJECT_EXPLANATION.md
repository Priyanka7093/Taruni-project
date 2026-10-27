# 🧠 Multi-Modal Explainable AI (XAI) Framework for Neurodegenerative Diagnosis
## Comprehensive Team Explanation & Project Onboarding Guide

---

## 📌 Executive Summary

Welcome to the **Multi-Modal Explainable AI (XAI) Framework for Neurodegenerative Diagnosis** project! 

The central goal of this project is to address the **"black box" problem and trust gap in medical AI**. In clinical settings (such as diagnosing **Alzheimer's Disease** or **Mild Cognitive Impairment**), doctors cannot rely on AI predictions unless the model provides clear, verifiable biological and clinical evidence for its decision.

This project implements a **Dual-Stream Multi-Modal Fusion Deep Learning Network** combined with two complementary **Explainable AI (XAI)** engines:
1. **Grad-CAM** for visual brain MRI region highlighting.
2. **SHAP-inspired Sensitivity Attributions** for clinical tabular score explanations.

The system is delivered as a complete end-to-end web application with a FastAPI backend and a modern glassmorphic dashboard UI.

---

## 🎯 Clinical Problem & Diagnostic Targets

Neurodegenerative disorders evolve progressively. Early detection is critical for intervention:
* **Cognitively Normal (CN)**: Healthy control individuals with no significant cognitive decline or brain tissue loss.
* **Mild Cognitive Impairment (MCI)**: Intermediate stage between normal aging and dementia. Subtle memory complaints, minor ventricular expansion, and slight cognitive metric drops.
* **Alzheimer's Disease (AD)**: Advanced neurodegenerative condition characterized by severe memory loss, marked cortical atrophy, and significant expansion of fluid-filled ventricles.

### Why Multi-Modal Data?
Single-modality models (e.g., image-only or clinical-only) miss key diagnostic signals:
* **Brain MRI Scans (Unstructured Spatial Data)**: Reveal structural changes like brain tissue shrinkage (cortical atrophy) and enlarged lateral ventricles.
* **Clinical & Psychometric Indicators (Structured Tabular Data)**: Measure cognitive performance (MMSE score), functional impairment (CDR), and normalized brain volume metrics (nWBV, eTIV).

---

## 🏗️ System Architecture & Workflow

```
                                 ┌────────────────────────┐
                                 │    Brain MRI Scans     │
                                 │    (224 x 224 RGB)     │
                                 └───────────┬────────────┘
                                             │
                                             ▼
                                 [ ResNet-18 Image Stream ] ──► (512d Embedding)
                                             │                         │
                                             ▼                         ▼
┌────────────────────────┐       ┌────────────────────────┐   [ Multi-Modal ] ──► Final Diagnosis
│  Clinical Metadata     │ ────► │ MLP Clinical Stream    │ ─►[ Fusion Head ]    (CN / MCI / AD)
│ (Age, MMSE, CDR, nWBV) │       └────────────────────────┘   (576d -> 128d -> 3)
└────────────────────────┘                   │
             │                               │
             ▼                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Explainable AI (XAI) Engine                               │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│  Grad-CAM (Visual Brain Scan Heatmaps)    │ SHAP Attributions (Clinical Feature Impact)│
│  - Captures conv layer spatial activations│ - Perturbation-based risk impact analysis  │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

---

## 🔬 Model Architecture Deep-Dive

The model (`MultiModalFusionModel` in [`models/fusion_network.py`](file:///d:/Taruni%20project/multimodal_xai_framework/models/fusion_network.py)) consists of three primary neural subnetworks:

### 1. Visual Imaging Stream (`ImageStreamNetwork` in [`models/image_stream.py`](file:///d:/Taruni%20project/multimodal_xai_framework/models/image_stream.py))
* **Backbone**: Modified ResNet-18 pretrained architecture.
* **Input**: 2D Axial Brain MRI scan slice tensor `(Batch, 3, 224, 224)`.
* **Output**: Dense feature representation vector of size **512**.
* **Key Detail**: The classification fc head of ResNet-18 is replaced with `nn.Identity()`, preserving latent spatial representation.

### 2. Tabular Clinical Stream (`ClinicalStreamNetwork` in [`models/clinical_stream.py`](file:///d:/Taruni%20project/multimodal_xai_framework/models/clinical_stream.py))
* **Backbone**: Multi-Layer Perceptron (MLP) with Batch Normalization, ReLU activations, and Dropout.
* **Input**: Standardized clinical feature vector of size **8** (Age, Gender, MMSE, CDR, Educ, eTIV, nWBV, ASF).
* **Output**: Dense clinical representation vector of size **64**.

### 3. Multi-Modal Fusion Head
* **Fusion Strategy**: Late fusion via feature concatenation.
* **Fused Vector Size**: $512 + 64 = 576$ dimensions.
* **Classification Pipeline**:
  $$\text{Input}(576) \longrightarrow \text{Linear}(128) \longrightarrow \text{BatchNorm1d} \longrightarrow \text{ReLU} \longrightarrow \text{Dropout}(0.3) \longrightarrow \text{Linear}(3)$$
* **Output**: Logits over 3 target classes $\rightarrow$ Softmax probabilities.

---

## 💡 Explainable AI (XAI) Mechanism

### A. Grad-CAM (Gradient-weighted Class Activation Mapping)
* **Location**: [`explainability/gradcam.py`](file:///d:/Taruni%20project/multimodal_xai_framework/explainability/gradcam.py)
* **How it Works**:
  1. Hooks into the last convolutional block of ResNet-18 (`layer4[1].conv2`).
  2. Computes gradients of the target class score with respect to feature map activations.
  3. Pools gradients to derive neuron importance weights ($\alpha_k$).
  4. Generates a 2D spatial heatmap overlay using OpenCV `JET` colormap (Red/Yellow = high diagnostic relevance, Blue = neutral).
* **Clinical Significance**: Shows radiologists and neurologists which exact brain structures (e.g., enlarged lateral ventricles or temporal lobe atrophy) triggered the diagnosis.

### B. SHAP Clinical Feature Attributions
* **Location**: [`explainability/shap_explainer.py`](file:///d:/Taruni%20project/multimodal_xai_framework/explainability/shap_explainer.py)
* **How it Works**:
  1. Uses sensitivity perturbation analysis on clinical indicators.
  2. Perturbs individual features back towards mean baseline values while holding the brain image constant.
  3. Evaluates changes in target class confidence to quantify positive (risk increasing) or negative (risk reducing) feature impact.
  4. Generates natural language clinical interpretations (e.g., *"MMSE score of 17.0 is below normal threshold (<24), strongly indicating cognitive decline."*).

---

## 📊 Dataset & Features Overview

The data pipeline standardizes OASIS (Open Access Series of Imaging Studies) and ADNI metrics:

| Feature Name | Type | Description / Normal Range | Diagnostic Impact |
| :--- | :--- | :--- | :--- |
| **Age** | Numeric | Patient age in years (60 - 90+) | Higher age slightly increases overall dementia risk profile. |
| **Gender** | Categorical | 0: Female, 1: Male | Demographic factor. |
| **MMSE** | Numeric | Mini-Mental State Examination (0-30) | **Critical**: Score $<24$ indicates cognitive impairment. |
| **CDR** | Categorical | Clinical Dementia Rating (0, 0.5, 1.0, 2.0) | **Critical**: 0 = None, 0.5 = Very Mild, 1 = Mild, 2 = Moderate. |
| **Educ** | Numeric | Years of Formal Education | High education correlates with cognitive reserve. |
| **eTIV** | Numeric | Estimated Total Intracranial Volume ($mm^3$) | Normalization baseline for head size variations. |
| **nWBV** | Numeric | Normalized Whole Brain Volume ($0.60 - 0.85$) | **Critical**: Low values ($<0.73$) signal brain atrophy. |
| **ASF** | Numeric | Atlas Scaling Factor | Volume correction factor. |

### Data Generation & Loader
* [`data/synthetic_generator.py`](file:///d:/Taruni%20project/multimodal_xai_framework/data/synthetic_generator.py): Synthesizes realistic axial MRI slice images programmatically using Pillow (`ImageDraw`) with class-conditioned ventricular expansion and tissue atrophy.
* [`data/dataset.py`](file:///d:/Taruni%20project/multimodal_xai_framework/data/dataset.py): Implements PyTorch `Dataset` with `StandardScaler` for clinical data and standard ImageNet normalization for image slices.

---

## 📁 Repository Structure & Directory Map

```
multimodal_xai_framework/
├── config.py                 # Central hyperparameter & path configurations
├── requirements.txt           # Environment dependencies (PyTorch, FastAPI, OpenCV, etc.)
├── train.py                   # Script to train model & save weights (.pth & scaler.pkl)
│
├── data/
│   ├── synthetic_generator.py # Programmatic MRI & OASIS dataset generator
│   ├── dataset.py             # PyTorch Dataset loader & data preprocessing
│   └── clinical_data.csv      # Generated patient tabular records
│
├── models/
│   ├── image_stream.py        # ResNet-18 CNN image embedding extractor (512d)
│   ├── clinical_stream.py     # MLP tabular embedding extractor (64d)
│   └── fusion_network.py      # Concatenation fusion & 3-class classification head
│
├── explainability/
│   ├── gradcam.py             # Grad-CAM heatmap generator for MRI scans
│   └── shap_explainer.py      # Clinical feature sensitivity & attribution engine
│
├── app/
│   └── main.py                # FastAPI server hosting REST API endpoints
│
├── static/
│   ├── index.html             # Glassmorphic UI Dashboard layout
│   ├── style.css              # Dark mode styling & responsive CSS
│   └── app.js                 # Frontend interactive logic & API fetcher
│
└── saved_models/              # Directory storing trained model weights (.pth) & scaler (.pkl)
```

---

## 🛠️ Step-by-Step Onboarding & Quick Start Guide

### 1. Environment Setup
Create a virtual environment and install the project dependencies:
```bash
# Navigate to the framework folder
cd multimodal_xai_framework

# Install required Python packages
pip install -r requirements.txt
```

### 2. Training the Fusion Model
Run `train.py` to generate the synthetic dataset (if missing), train the dual-stream neural network for 15 epochs, and save the best model weights to `saved_models/`:
```bash
python train.py
```
*Expected Output*: Displays train loss, test accuracy, and weighted F1-score for each epoch.

### 3. Launching the Web Dashboard & API
Start the FastAPI server using Uvicorn:
```bash
python app/main.py
```
Or directly via uvicorn:
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4. Interactive Testing
1. Open your web browser to **`http://127.0.0.1:8000`**.
2. Click on the **Preset Sample Patients** (CN, MCI, AD) for one-click testing.
3. Observe:
   * **Diagnostic Decision**: Predicted stage (e.g., Alzheimer's Disease) with confidence %.
   * **Grad-CAM Brain Overlay**: Heatmap showing active activation zones on the brain scan.
   * **Clinical Feature Impact**: Ranked list of clinical factors driving the decision.

---

## 👥 Recommended Team Member Roles & Contribution Areas

To distribute work effectively across your team, here are suggested responsibility areas:

1. **Model & Deep Learning Specialist**:
   * *Tasks*: Experiment with alternative CNN backbones (e.g., EfficientNet, Vision Transformers / ViT), tune fusion layer dimensions, add cross-attention fusion mechanisms.
2. **Explainability & Bio-Informatics Lead**:
   * *Tasks*: Enhance Grad-CAM to Integrated Gradients or Layer-CAM, refine SHAP background baseline distributions, expand clinical interpretation descriptions.
3. **Data Engineering Lead**:
   * *Tasks*: Integrate real OASIS-1/OASIS-3 or ADNI NIfTI/DICOM dataset pipelines, add 3D MRI volume slice extraction and preprocessing.
4. **Full-Stack / UI-UX Engineer**:
   * *Tasks*: Add PDF diagnostic report download feature, enhance chart visualization for feature attributions, add interactive image slice slider.

---

## 🏁 Summary Checklist for Teammates

- [x] Dual-Stream Architecture implemented ([`models/fusion_network.py`](file:///d:/Taruni%20project/multimodal_xai_framework/models/fusion_network.py))
- [x] Grad-CAM visual heatmap engine functional ([`explainability/gradcam.py`](file:///d:/Taruni%20project/multimodal_xai_framework/explainability/gradcam.py))
- [x] Clinical SHAP attributions implemented ([`explainability/shap_explainer.py`](file:///d:/Taruni%20project/multimodal_xai_framework/explainability/shap_explainer.py))
- [x] FastAPI server & REST API operational ([`app/main.py`](file:///d:/Taruni%20project/multimodal_xai_framework/app/main.py))
- [x] Glassmorphic web user interface ready ([`static/index.html`](file:///d:/Taruni%20project/multimodal_xai_framework/static/index.html))
