import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Diagnostic Target Classes
# BINARY: Normal vs Impaired (collapsed from 3-class CN/MCI/AD due to
# small real-data sample size - 122 subjects total, only 28 in AD class)
CLASSES = ["Normal (CN)", "Impaired (MCI/AD)"]
NUM_CLASSES = len(CLASSES)

# Clinical Tabular Features (real ADNI data - eTIV/nWBV/ASF dropped since ADNI
# doesn't report them the same way OASIS did; MOCA also dropped since it's
# 100% missing in this ADNI1-era image-matched cohort)
CLINICAL_FEATURES = [
    "Age",      # Patient Age (years)
    "Gender",   # 0: Female, 1: Male
    "MMSE",     # Mini-Mental State Examination (0-30, lower = cognitive impairment)
    "CDR",      # Clinical Dementia Rating - Sum of Boxes (ADNI's CDRSB)
    "Educ",     # Years of Education
    "FAQ"       # Functional Activities Questionnaire (0-30, higher = more impairment)
]
NUM_CLINICAL_FEATURES = len(CLINICAL_FEATURES)

# Image Settings
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3

# Model Architecture Dimensions
CNN_EMBED_DIM = 512
MLP_EMBED_DIM = 64
FUSION_HIDDEN_DIM = 128

# Training Settings
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "multimodal_fusion_model.pth")
PREPROCESSOR_PATH = os.path.join(SAVED_MODELS_DIR, "preprocessor.pkl")