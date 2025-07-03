# 🔥 Fire Detection using CNN (ResNet50)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This deep learning project detects fire in road surveillance imagery using a Convolutional Neural Network with the ResNet50V2 architecture. It follows a clear pipeline — from preprocessing image datasets to training a transfer learning model that classifies fire vs. no fire scenes with high accuracy.

---

## 📌 Overview

- **🎯 Goal**: Detect fire in real-world images for early hazard detection and alert systems.
- **📂 Dataset**: Image dataset with labeled fire and non-fire scenes (e.g. surveillance footage).
- **🧠 Model**: Pretrained ResNet50V2 with custom top layers.
- **🔍 Techniques**: Transfer Learning, Image Augmentation, EarlyStopping, ReduceLROnPlateau.
- **📊 Evaluation**: Accuracy, Training vs Validation Loss, Model Checkpointing.

---

## 📁 Project Structure

| File                         | Description                                                         |
|------------------------------|---------------------------------------------------------------------|
| `Fire_Detection.ipynb`       | 🧠 Main notebook: preprocessing, model training, evaluation         |
| `train/` and `val/` folders  | 📁 Training and validation images (Not included due to size)        |
| `model.h5`                   | 💾 Saved trained model (if exported)                                |
| `README.md`                  | 📘 Project overview and usage instructions                          |

---

## 🔄 Data Flow Summary

1. 📁 **Image Dataset**  
   └─ `Train/`, `Vali/`, `Test/` directories, each with `Fire/` and `Non‑Fire/` sub‑folders.

   ⬇️

2. 🧹 **ImageDataGenerator**  
   - Rescales pixels (`1./255`).  
   - Streams batches for **train**, **validation**, and **test** sets.

   ⬇️

3. 🤖 **Model Experiments**  
   a. **ResNet50V2** (transfer learning, frozen base)  
   b. **VGG19**  (transfer learning, frozen base)  
   c. **Custom CNN**  (3 conv blocks → GAP/FC layers)

   ⬇️

4. 🏃 **Training Loop** (max 30 epochs each)  
   - `EarlyStopping` (patience = 5, monitor =`accuracy`)  
   - `ReduceLROnPlateau` (auto‑tunes LR)  
   - `ModelCheckpoint` → saves **best‑val‑accuracy** weights

   ⬇️

5. 📈 **Evaluation & Visuals**  
   - Accuracy/Loss curves (`plt`)  
   - Confusion Matrix + Classification Report (scikit‑learn)

   ⬇️

6. 💾 **Model Export**  
   - `model/resnet50v2_best.keras`  
   - `model/vgg19_best.keras`  
   - `model/custom_cnn_best.keras` (if enabled)
   
---


## 🧠 Tech Stack

| Category       | Tools & Libraries                              |
|----------------|------------------------------------------------|
| **Language**   | Python                                          |
| **Frameworks** | TensorFlow, Keras                              |
| **Model**      | ResNet50V2 (Transfer Learning)                 |
| **Tools**      | Google Colab, GitHub                           |
| **Callbacks**  | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| **Visualization** | Matplotlib                                |

---

## 🌐 Run Options

You can run the notebook on:

- ✅ **Google Colab** (recommended for GPU)
- ✅ **Jupyter Notebook**
- ✅ **VS Code** (with Python + TensorFlow installed)

---

## 🚀 Project Highlights

- 🔥 Real-world fire image detection
- ⚡ Transfer learning with ResNet50V2
- 📉 Callback usage for smarter training
- 📸 Trained on road surveillance-style imagery
- 🧪 Easily deployable to edge devices like Jetson Nano or Raspberry Pi
- 🧹 Clean structure and readable code for quick reuse

---

## 🤝 How to Use or Contribute

- 🍴 Fork this repo and add your own dataset
- 🔁 Try other models (like MobileNetV2 or EfficientNet)
- ⚙️ Add object detection or localization (YOLO, SSD)
- 📲 Deploy using Flask or Streamlit for live detection
- 🤖 Integrate with IoT systems or emergency alert APIs

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, or share it with credit.

---

## 🙋 Author

**Saurav Singh Negi**  
🎓 B.Tech Student | 💻 Deep Learning Enthusiast  
🔗 [GitHub: SAURAV0703](https://github.com/SAURAV0703)

