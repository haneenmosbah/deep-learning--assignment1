# Chicken Disease Classification using Deep Learning

## Project Overview

This project implements a deep learning model to classify chicken diseases using convolutional neural networks. The system is designed to assist in early disease detection from poultry images.

The model is built using TensorFlow and Keras, and trained on a labeled image dataset containing multiple disease categories.

---

## Problem Statement

Poultry diseases can significantly affect production and farm profitability. Manual diagnosis requires expert knowledge and may not always be accessible.

This project aims to:
- Automatically classify chicken diseases from images
- Provide a scalable AI-based diagnostic assistant
- Improve early detection accuracy

---

## Dataset

- Image-based dataset of chicken diseases
- 5,000 – 20,000 images
- Multiple disease classes
- Stratified split into:
  - 70% Training
  - 15% Validation
  - 15% Testing

---

## Model Architecture

Base Model:
- ResNet50 (without top layer)

Framework:
- TensorFlow
- Keras API

Architecture Flow:

Input Image (224x224x3)  
→ ResNet50 (Feature Extraction)  
→ GlobalAveragePooling2D  
→ BatchNormalization  
→ Dense (256, ReLU)  
→ Dropout (0.4)  
→ Dense (128, ReLU)  
→ Dropout (0.3)  
→ Softmax Output Layer  

---

## Training Strategy

Phase 1: Feature Extraction
- Base model frozen
- Optimizer: AdamW
- Learning Rate: 1e-4
- EarlyStopping applied

Phase 2: Fine Tuning
- Unfreeze last layers
- Freeze BatchNormalization layers
- Learning Rate reduced to 1e-5

Callbacks:
- EarlyStopping
- ModelCheckpoint

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Final Validation Accuracy:
~66% (training without ImageNet pretrained weights)

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn

---

## How to Run

1. Clone the repository
2. Install dependencies
3. Update dataset path
4. Run the training script
