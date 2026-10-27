import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from config import DATA_DIR, CLINICAL_FEATURES, CLASSES


def generate_brain_mri_slice(class_label: int, seed: int = 42) -> Image.Image:
    """
    Generates a realistic synthetic 2D brain MRI axial slice.
    Varying ventricular expansion and tissue atrophy based on diagnosis class (0: CN, 1: MCI, 2: AD).
    """
    np.random.seed(seed)
    img_size = (224, 224)
    img = Image.new("L", img_size, color=10) # Dark skull background
    draw = ImageDraw.Draw(img)

    center = (112, 112)
    outer_radius_x = 85 + np.random.randint(-3, 3)
    outer_radius_y = 95 + np.random.randint(-3, 3)

    # 1. Skull & Scalp contour
    draw.ellipse(
        [center[0] - outer_radius_x, center[1] - outer_radius_y, 
         center[0] + outer_radius_x, center[1] + outer_radius_y],
        fill=140, outline=200, width=4
    )

    # 2. Brain Parenchyma (Gray/White matter tissue)
    brain_rx = outer_radius_x - 12
    brain_ry = outer_radius_y - 12
    draw.ellipse(
        [center[0] - brain_rx, center[1] - brain_ry, 
         center[0] + brain_rx, center[1] + brain_ry],
        fill=70
    )

    # 3. Cortex tissue wrinkles (sulci patterns)
    for i in range(16):
        angle = i * (2 * np.pi / 16)
        x1 = center[0] + int((brain_rx - 15) * np.cos(angle))
        y1 = center[1] + int((brain_ry - 15) * np.sin(angle))
        x2 = center[0] + int(brain_rx * np.cos(angle))
        y2 = center[1] + int(brain_ry * np.sin(angle))
        draw.line([x1, y1, x2, y2], fill=25, width=2)

    # 4. Ventricles (CSF - Fluid filled cavities)
    # Severe Alzheimer's (AD) has enlarged ventricles (hydrocephalus ex vacuo)
    vent_size_factor = 1.0 if class_label == 0 else (1.8 if class_label == 1 else 2.8)
    v_w = int(12 * vent_size_factor)
    v_h = int(28 * vent_size_factor)

    # Left & Right lateral ventricles
    draw.ellipse([center[0] - v_w - 6, center[1] - v_h//2, center[0] - 4, center[1] + v_h//2], fill=15)
    draw.ellipse([center[0] + 4, center[1] - v_h//2, center[0] + v_w + 6, center[1] + v_h//2], fill=15)

    # 5. Temporal Lobe / Hippocampus focal region (Atrophy marker)
    if class_label > 0:
        atrophy_intensity = 30 if class_label == 1 else 15
        draw.ellipse([center[0] - 45, center[1] + 15, center[0] - 25, center[1] + 35], fill=atrophy_intensity)
        draw.ellipse([center[0] + 25, center[1] + 15, center[0] + 45, center[1] + 35], fill=atrophy_intensity)

    # Apply slight Gaussian blur for realistic MRI noise texture
    img_blurred = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    
    # Convert to RGB 3-channel
    rgb_img = Image.merge("RGB", (img_blurred, img_blurred, img_blurred))
    return rgb_img


def generate_dataset(num_samples: int = 180):
    """
    Generates synthetic multi-modal dataset (images + tabular clinical metrics).
    Clinical features match the real ADNI schema: Age, Gender, MMSE, CDR, Educ, MOCA, FAQ
    (eTIV/nWBV/ASF replaced with MOCA/FAQ since ADNI doesn't report the OASIS-style volumetrics).
    """
    images_dir = os.path.join(DATA_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    data_list = []
    samples_per_class = num_samples // len(CLASSES)

    for c_idx in range(len(CLASSES)):
        for i in range(samples_per_class):
            sample_id = f"PATIENT_{c_idx}_{i:03d}"
            img_filename = f"{sample_id}.png"
            img_path = os.path.join(images_dir, img_filename)

            # Generate and save image
            img = generate_brain_mri_slice(class_label=c_idx, seed=c_idx*1000 + i)
            img.save(img_path)

            # Clinical metrics generation conditioned on class label
            if c_idx == 0:  # CN (Cognitively Normal)
                age = float(np.random.normal(70, 5))
                gender = int(np.random.choice([0, 1]))
                mmse = float(np.clip(np.random.normal(29.0, 1.0), 27, 30))
                cdr = 0.0
                educ = float(np.clip(np.random.normal(15, 2), 10, 20))
                moca = float(np.clip(np.random.normal(27.0, 1.5), 24, 30))
                faq = float(np.clip(np.random.normal(1.0, 0.8), 0, 3))
            elif c_idx == 1:  # MCI (Mild Cognitive Impairment)
                age = float(np.random.normal(74, 5))
                gender = int(np.random.choice([0, 1]))
                mmse = float(np.clip(np.random.normal(25.0, 1.8), 22, 27))
                cdr = 0.5
                educ = float(np.clip(np.random.normal(14, 2.5), 8, 20))
                moca = float(np.clip(np.random.normal(21.0, 2.5), 17, 25))
                faq = float(np.clip(np.random.normal(6.0, 2.5), 2, 12))
            else:  # AD (Alzheimer's Disease)
                age = float(np.random.normal(78, 6))
                gender = int(np.random.choice([0, 1]))
                mmse = float(np.clip(np.random.normal(18.0, 3.5), 10, 22))
                cdr = float(np.random.choice([1.0, 2.0], p=[0.7, 0.3]))
                educ = float(np.clip(np.random.normal(13, 3), 6, 18))
                moca = float(np.clip(np.random.normal(13.0, 3.5), 5, 18))
                faq = float(np.clip(np.random.normal(20.0, 5.0), 10, 30))

            data_list.append({
                "Subject_ID": sample_id,
                "Image_Path": img_filename,
                "Age": round(age, 1),
                "Gender": gender,
                "MMSE": round(mmse, 1),
                "CDR": cdr,
                "Educ": round(educ, 1),
                "MOCA": round(moca, 1),
                "FAQ": round(faq, 1),
                "Group": c_idx  # Target Label
            })

    df = pd.DataFrame(data_list)
    csv_path = os.path.join(DATA_DIR, "clinical_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] Dataset successfully created: {len(df)} samples saved to {csv_path}")
    return csv_path


if __name__ == "__main__":
    generate_dataset()