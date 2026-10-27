import os
import sys
import pickle
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import (
    MODEL_PATH, PREPROCESSOR_PATH, STATIC_DIR, CLASSES,
    CLINICAL_FEATURES, IMAGE_SIZE, DEVICE
)
from models.fusion_network import MultiModalFusionModel
from explainability.gradcam import GradCAMExplainer
from explainability.shap_explainer import ClinicalSHAPExplainer
from data.synthetic_generator import generate_dataset, generate_brain_mri_slice

app = FastAPI(
    title="Multi-Modal XAI Diagnostic Decision Support API",
    description="Backend service providing diagnostic predictions, Grad-CAM heatmaps, and SHAP feature attributions."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model, scaler, and explainers
model = None
scaler = None
gradcam_explainer = None
shap_explainer = None

image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.on_event("startup")
def load_resources():
    global model, scaler, gradcam_explainer, shap_explainer

    # Ensure synthetic dataset and model exist
    csv_path = os.path.join(BASE_DIR, "data", "clinical_data.csv")
    if not os.path.exists(csv_path) or not os.path.exists(MODEL_PATH):
        print("[!] Model weights or dataset missing. Training baseline model...")
        from train import train_model
        train_model()

    # Load fitted Scaler
    with open(PREPROCESSOR_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load Model
    model = MultiModalFusionModel(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Initialize Explainers
    gradcam_explainer = GradCAMExplainer(model)
    shap_explainer = ClinicalSHAPExplainer(model, scaler)
    print("[+] Model and XAI Engines successfully loaded!")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>API is running. UI loading...</h1>")

@app.get("/api/sample_presets")
def get_sample_presets():
    """
    Returns preset sample patients (Normal, Impaired) for instant testing in the UI.
    Binary classification: Normal (CN) vs Impaired (MCI/AD combined).
    """
    presets = [
        {
            "id": "preset_normal",
            "name": "Normal Cognition (Sample Patient 1)",
            "target": "Normal (CN)",
            "clinical": {
                "Age": 70.0, "Gender": 0, "MMSE": 29.0, "CDR": 0.0,
                "Educ": 16.0, "FAQ": 1.0
            },
            "class_idx": 0
        },
        {
            "id": "preset_mci",
            "name": "Mild Impairment (Sample Patient 2)",
            "target": "Impaired (MCI/AD)",
            "clinical": {
                "Age": 74.0, "Gender": 1, "MMSE": 24.0, "CDR": 1.0,
                "Educ": 14.0, "FAQ": 6.0
            },
            "class_idx": 1
        },
        {
            "id": "preset_severe",
            "name": "Severe Impairment (Sample Patient 3)",
            "target": "Impaired (MCI/AD)",
            "clinical": {
                "Age": 80.0, "Gender": 0, "MMSE": 17.0, "CDR": 5.0,
                "Educ": 12.0, "FAQ": 20.0
            },
            "class_idx": 1
        }
    ]
    return JSONResponse(content={"presets": presets})

@app.post("/api/predict_and_explain")
async def predict_and_explain(
    file: UploadFile = File(None),
    class_preset: int = Form(None),
    age: float = Form(75.0),
    gender: int = Form(0),
    mmse: float = Form(24.0),
    cdr: float = Form(1.0),
    educ: float = Form(14.0),
    faq: float = Form(5.0)
):
    """
    Main Endpoint: Accepts MRI image + clinical attributes, returns diagnosis + Grad-CAM base64 + SHAP attributions.
    """
    try:
        # 1. Process Image Input
        if file is not None:
            contents = await file.read()
            raw_pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        else:
            # Fallback sample slice if no file uploaded
            c_label = class_preset if class_preset is not None else 1
            raw_pil_image = generate_brain_mri_slice(class_label=c_label, seed=42)

        image_tensor = image_transform(raw_pil_image).to(DEVICE)

        # 2. Process Clinical Input
        clinical_dict = {
            "Age": float(age), "Gender": int(gender), "MMSE": float(mmse),
            "CDR": float(cdr), "Educ": float(educ), "FAQ": float(faq)
        }

        raw_clinical_vals = [clinical_dict[feat] for feat in CLINICAL_FEATURES]
        scaled_clinical_vals = scaler.transform([raw_clinical_vals])[0]
        clinical_tensor = torch.tensor(scaled_clinical_vals, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 3. Model Inference
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0), clinical_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class_idx = int(np.argmax(probs))
        predicted_label = CLASSES[pred_class_idx]
        confidence = float(probs[pred_class_idx])

        # 4. Generate Grad-CAM Heatmap
        base64_heatmap, _, _ = gradcam_explainer.get_base64_heatmap(
            raw_pil_image, image_tensor, clinical_tensor.squeeze(0)
        )

        # Convert original input image to base64 for side-by-side rendering
        buffered_orig = io.BytesIO()
        raw_pil_image.resize((224, 224)).save(buffered_orig, format="PNG")
        base64_original = base64.b64encode(buffered_orig.getvalue()).decode("utf-8")

        # 5. Generate SHAP Attributions
        attributions, _, _ = shap_explainer.explain_instance(
            clinical_dict, image_tensor, target_class=pred_class_idx
        )

        # Format probabilities dictionary
        class_probabilities = {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}

        return JSONResponse(content={
            "success": True,
            "diagnosis": predicted_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": class_probabilities,
            "original_image": f"data:image/png;base64,{base64_original}",
            "heatmap_image": f"data:image/png;base64,{base64_heatmap}",
            "attributions": attributions,
            "clinical_inputs": clinical_dict
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
