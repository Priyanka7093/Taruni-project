import numpy as np
import torch
import torch.nn.functional as F
from config import CLINICAL_FEATURES, CLASSES

class ClinicalSHAPExplainer:
    """
    Explainability Engine for Clinical Tabular Features.
    Calculates feature importance attribution scores for clinical patient metrics.
    """
    def __init__(self, model, scaler):
        self.model = model
        self.model.eval()
        self.scaler = scaler

    def explain_instance(self, clinical_unscaled: dict, raw_image_tensor: torch.Tensor, target_class: int = None):
        """
        Calculates feature attributions via sensitivity perturbation analysis.
        """
        feature_names = CLINICAL_FEATURES
        raw_vals = [clinical_unscaled[feat] for feat in feature_names]
        
        # Scale tabular data
        scaled_vals = self.scaler.transform([raw_vals])[0]
        base_clinical_tensor = torch.tensor(scaled_vals, dtype=torch.float32).unsqueeze(0)

        if raw_image_tensor.dim() == 3:
            raw_image_tensor = raw_image_tensor.unsqueeze(0)

        with torch.no_grad():
            base_logits = self.model(raw_image_tensor, base_clinical_tensor)
            base_probs = F.softmax(base_logits, dim=1).numpy()[0]

        if target_class is None:
            target_class = int(np.argmax(base_probs))

        base_score = base_probs[target_class]
        attributions = []

        # Perturbation-based sensitivity scoring
        for i, feat_name in enumerate(feature_names):
            perturbed_scaled = scaled_vals.copy()
            # Perturb feature by 1 std dev in baseline space (set to zero mean in scaled space)
            perturbed_scaled[i] = 0.0 
            perturbed_tensor = torch.tensor(perturbed_scaled, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                p_logits = self.model(raw_image_tensor, perturbed_tensor)
                p_probs = F.softmax(p_logits, dim=1).numpy()[0]

            p_score = p_probs[target_class]
            impact = float(base_score - p_score) # positive impact means feature pushes towards target class

            # Clinical interpretation string
            desc = self._get_clinical_interpretation(feat_name, raw_vals[i], impact)

            attributions.append({
                "feature": feat_name,
                "value": raw_vals[i],
                "impact": round(impact, 4),
                "importance": round(abs(impact), 4),
                "description": desc
            })

        # Sort features by highest impact magnitude
        attributions = sorted(attributions, key=lambda x: x["importance"], reverse=True)
        return attributions, target_class, CLASSES[target_class]

    def _get_clinical_interpretation(self, feature_name: str, value: float, impact: float) -> str:
        direction = "increases risk score" if impact > 0 else "reduces risk score"
        
        if feature_name == "MMSE":
            if value < 24:
                return f"MMSE score of {value} is below normal threshold (<24), strongly indicating cognitive decline."
            return f"MMSE score of {value} is in normal range (24-30)."
        elif feature_name == "CDR":
            if value > 0:
                return f"Clinical Dementia Rating of {value} indicates active dementia symptoms."
            return f"CDR of {value} indicates no clinical dementia impairment."
        elif feature_name == "nWBV":
            if value < 0.73:
                return f"Normalized Whole Brain Volume ({value}) shows signs of cerebral tissue atrophy."
            return f"Brain volume ({value}) is within healthy limits."
        elif feature_name == "Age":
            return f"Patient age of {int(value)} years {direction}."
        else:
            return f"{feature_name} parameter value ({value}) {direction}."
