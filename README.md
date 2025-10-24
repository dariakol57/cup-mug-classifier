# ☕ Cup vs Mug Image Classifier

A deep learning project using **TensorFlow/Keras** and **EfficientNetB0** to classify images as either **cups** or **mugs**.  
The model is fine-tuned on a Kaggle dataset and achieves high accuracy through transfer learning.

---

## 🚀 Project Overview

This project demonstrates **binary image classification** using a pretrained convolutional neural network (CNN).  
We leverage **EfficientNetB0 (ImageNet weights)** as the base model and fine-tune it on a custom dataset of cup and mug images.

### Key Features:
- Transfer learning with EfficientNetB0  
- Two-stage training (warm-up + fine-tuning)  
- Visualization of training progress (accuracy and loss)  
- Support for GPU acceleration (Kaggle or Colab)  

---

## 📂 Dataset

**Dataset used:** [Cup_Mug_Dataset](https://www.kaggle.com/datasets)  

Structure:
Cup_mug_data/
├── train/
│ ├── Cup/
│ └── Mugs/
└── val/
├── Cup/
└── Mugs/



Each folder contains images of cups and mugs in different styles and lighting conditions.

---

## 🧠 Model Architecture

- **Base model:** EfficientNetB0 (`include_top=False`, pretrained on ImageNet)
- **Custom head:**
  - GlobalAveragePooling2D  
  - Dense(128, activation='relu')  
  - Dropout(0.4)  
  - Dense(2, activation='softmax')  

### Training strategy:
1. **Stage 1 (Warm-up):**  
   - Base frozen (`layer.trainable = False`)  
   - Only top layers trained for several epochs  

2. **Stage 2 (Fine-tuning):**  
   - Unfreeze last 50 layers of the base  
   - Lower learning rate for stable fine-tuning  

---

## ⚙️ Installation & Setup

### Requirements
- Python 3.10+
- TensorFlow 2.12+
- Matplotlib
- NumPy



