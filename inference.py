import os
import csv
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

DATA_ROOT = "./cv_hw1_data/data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

NUM_CLASSES = 100
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
CSV_PATH = "prediction_inf.csv"


class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.files = sorted(
            [
                p
                for p in self.root.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, path.stem


def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


@torch.no_grad()
def predict_test(model, loader, device, csv_path, idx_to_class):
    model.eval()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])

        for images, names in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()

            for name, pred in zip(names, preds):
                true_label = int(idx_to_class[pred])
                writer.writerow([name, true_label])


def main():
    test_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(TRAIN_DIR)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    test_dataset = TestDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    predict_test(model, test_loader, DEVICE, CSV_PATH, idx_to_class)
    print(f"Saved prediction to {CSV_PATH}")


if __name__ == "__main__":
    main()
