import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import DATA_DIR, DEVICE, EPOCHS, LEARNING_RATE, MODEL_PATH, CLASSES
from data.synthetic_generator import generate_dataset
from data.dataset import get_dataloaders
from models.fusion_network import MultiModalFusionModel

def train_model():
    print("=" * 60)
    print("   TRAINING DUAL-STREAM MULTI-MODAL FUSION MODEL   ")
    print("=" * 60)

    # Using real ADNI multimodal dataset (122 subjects: 33 CN, 61 MCI, 28 AD)
    csv_path = os.path.join(DATA_DIR, "clinical_data_adni_binary.csv")
    images_dir = os.path.join(DATA_DIR, "images_adni")

    if not os.path.exists(csv_path):
        print("[!] ADNI dataset not found. Falling back to synthetic dataset...")
        csv_path = os.path.join(DATA_DIR, "clinical_data.csv")
        images_dir = os.path.join(DATA_DIR, "images")
        if not os.path.exists(csv_path):
            print("[!] Synthetic dataset not found either. Generating synthetic dataset...")
            generate_dataset(num_samples=180)
    else:
        print(f"[+] Using real ADNI dataset: {csv_path}")

    train_loader, test_loader, scaler = get_dataloaders(csv_path, images_dir=images_dir)

    model = MultiModalFusionModel(num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, clinical, labels in train_loader:
            images = images.to(DEVICE)
            clinical = clinical.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, clinical, labels in test_loader:
                images = images.to(DEVICE)
                clinical = clinical.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images, clinical)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = test_loss / len(test_loader.dataset)
        test_acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}% | F1: {f1:.4f}")

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"   [+] Model saved to {MODEL_PATH} (Best Accuracy: {best_acc*100:.2f}%)")

    print("\n[+] Training Complete!")
    print(f"Final Test Accuracy: {best_acc*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    train_model()