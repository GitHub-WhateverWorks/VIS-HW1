# Image Classification with ResNet

## Introduction

This project tackles a 100-class image classification task using a ResNet backbone.
The model is trained on RGB images and predicts the object category ID for each test image.
To satisfy the assignment constraint, the backbone is based only on ResNet, with lightweight modifications applied to the classification head and training strategy to improve generalization.

The final model uses a pretrained ResNet-50 backbone with a dropout-regularized classification head.
During training, data augmentation, weight decay, label smoothing / focal loss experiments, and a cosine annealing scheduler were used to reduce overfitting and improve validation accuracy.

---

## Environment Setup

### 1. Install Dependencies

- pip install -r requirements.txt

## Usage

### Training

- How to train my model

- python main.py

### Inference

- How to check my results

- Train once to get best_model.pth that inference.py needs

- python inference.py

### Performance Snapshot

![alt text](image.png)