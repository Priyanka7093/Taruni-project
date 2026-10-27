import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import base64
import io

class GradCAMExplainer:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) Explainer for visual imaging data.
    Highlights anatomical brain regions that influenced the model's diagnostic prediction.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            self.target_layer = self.model.image_stream.get_last_conv_layer()
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, image_tensor: torch.Tensor, clinical_tensor: torch.Tensor, target_class: int = None):
        """
        Generates Grad-CAM activation heatmap overlay.
        """
        self.model.zero_grad()
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if clinical_tensor.dim() == 1:
            clinical_tensor = clinical_tensor.unsqueeze(0)

        image_tensor.requires_grad = True

        # Forward pass
        logits = self.model(image_tensor, clinical_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()

        target_score = logits[0, target_class]
        target_score.backward()

        # Compute neuron importance weights alpha
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU to keep positive influence
        
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap) # Normalize [0, 1]
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap, target_class, probs[0, target_class].item()

    def overlay_heatmap(self, raw_pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
        """
        Overlays Grad-CAM heatmap on original brain MRI image.
        """
        img_np = np.array(raw_pil_image.resize((224, 224)))
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply JET colormap (red/yellow = high activation, blue = low activation)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

        # Blend image and heatmap
        blended = cv2.addWeighted(img_np, 1.0 - alpha, color_heatmap, alpha, 0)
        return Image.fromarray(blended)

    def get_base64_heatmap(self, raw_pil_image: Image.Image, image_tensor: torch.Tensor, clinical_tensor: torch.Tensor):
        heatmap, target_class, confidence = self.generate_heatmap(image_tensor, clinical_tensor)
        overlay_img = self.overlay_heatmap(raw_pil_image, heatmap)
        
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str, target_class, confidence
