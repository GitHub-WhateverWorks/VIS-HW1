import os
import csv
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import logging
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)

logger = logging.getLogger()

DATA_ROOT = "./cv_hw1_data/data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

NUM_CLASSES = 100
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 20
LR = 3e-4
VAL_SPLIT = 0.1
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"
CSV_PATH = "prediction.csv"


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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        ce_loss = -(true_dist * log_probs).sum(dim=1)
        pt = (probs * true_dist).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        return loss.mean()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


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
    set_seed(SEED)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    )

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

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=test_transform)

    test_dataset = TestDataset(TEST_DIR, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_model(NUM_CLASSES).to(DEVICE)

    criterion = FocalLoss(num_classes=NUM_CLASSES, gamma=2.0, smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        logger.info(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)
        scheduler.step()
    logger.info(f"Best Val Acc: {best_val_acc:.4f}")
    logger.info(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    predict_test(model, test_loader, DEVICE, CSV_PATH, idx_to_class)
    logger.info(f"Saved prediction to {CSV_PATH}")


if __name__ == "__main__":
    main()
