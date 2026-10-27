import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config import DATA_DIR, CLINICAL_FEATURES, IMAGE_SIZE, PREPROCESSOR_PATH, BATCH_SIZE

class MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, scaler: StandardScaler = None, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.is_train = is_train

        # Extract Tabular Clinical Features
        self.clinical_data = self.df[CLINICAL_FEATURES].values.astype(np.float32)

        if scaler is None:
            self.scaler = StandardScaler()
            self.clinical_data = self.scaler.fit_transform(self.clinical_data)
        else:
            self.scaler = scaler
            self.clinical_data = self.scaler.transform(self.clinical_data)

        # Image Augmentations / Normalization
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image_Path"]
        img_path = os.path.join(self.images_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        clinical_tensor = torch.tensor(self.clinical_data[idx], dtype=torch.float32)
        label_tensor = torch.tensor(int(row["Group"]), dtype=torch.long)

        return image_tensor, clinical_tensor, label_tensor

def get_dataloaders(csv_file: str = None, images_dir: str = None, test_size: float = 0.2, random_state: int = 42):
    if csv_file is None:
        csv_file = os.path.join(DATA_DIR, "clinical_data.csv")
    if images_dir is None:
        images_dir = os.path.join(DATA_DIR, "images")

    df = pd.read_csv(csv_file)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["Group"])

    train_dataset = MultiModalDataset(train_df, images_dir=images_dir, is_train=True)
    scaler = train_dataset.scaler

    # Save fitted scaler for inference
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(scaler, f)

    test_dataset = MultiModalDataset(test_df, images_dir=images_dir, scaler=scaler, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, scaler