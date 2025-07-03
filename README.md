# 🔥 Fire Detection using CNN (ResNet50 + VGG19)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This deep learning project detects fire in road surveillance imagery using Convolutional Neural Networks with **ResNet50V2**, **VGG19**, and a custom CNN architecture. It implements transfer learning and a clean training pipeline with callbacks and evaluation metrics.

---

## 📌 Overview

- **🎯 Goal**: Detect fire in real-world images for early hazard detection and alert systems.
- **🧠 Models**: ResNet50V2, VGG19 (with frozen base), and a custom CNN.
- **📊 Evaluation**: Accuracy, training/validation loss plots, classification report.
- **🧪 Features**: Transfer learning, callbacks (EarlyStopping, ReduceLROnPlateau), and confusion matrix analysis.

---

## 📁 Project Structure

| File                      | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| `Fire_Detection.ipynb`    | 🧠 Main notebook with full training & evaluation pipeline          |
| `README.md`               | 📘 Project documentation (this file)                              |

> ⚠️ Dataset and trained models are **not included** due to size. Instructions are provided below to replicate the project.

---

## 🔄 Data Flow Summary

1. 📁 **Image Dataset Structure** (you must upload manually): 
   └─ `Train/`, `Vali/`, `Test/` directories, each with `Fire/` and `Non‑Fire/` sub‑folders.
   
    ⬇️

3. 🧹 **ImageDataGenerator**  
   - Rescales images (`1./255`)  
   - Streams batches from `Train/`, `Vali/`, and `Test/`
     
    ⬇️

3. 🤖 **Model Training**  
   - `ResNet50V2` and `VGG19` (pretrained, frozen)  
   - A 3-block **Custom CNN**  
   - Trained for up to 30 epochs with:
     - `EarlyStopping`
     - `ReduceLROnPlateau`
     - `ModelCheckpoint`
       
    ⬇️

4. 📊 **Evaluation & Visualization**  
   - Accuracy & loss plots  
   - Confusion matrix and classification report
     
    ⬇️

5. 💾 **(Optional)** Save best model weights as `.keras`  
   *(Paths/code included but files not uploaded in this repo)*

---

## 🧠 Tech Stack

| Category       | Tools & Libraries                              |
|----------------|------------------------------------------------|
| **Language**   | Python                                          |
| **Frameworks** | TensorFlow, Keras                              |
| **Models**     | ResNet50V2, VGG19, Custom CNN                  |
| **Tools**      | Google Colab, GitHub                           |
| **Visualization** | Matplotlib, Seaborn                        |
| **Evaluation** | Scikit-learn (confusion matrix, reports)       |

---

## 🌐 Run Options

You can run this project using:

- ✅ **Google Colab** *(Recommended: GPU-backed runtime)*
- ✅ **Jupyter Notebook**
- ✅ **VS Code** (with appropriate dependencies installed)

---

## 🚀 Project Highlights

- 🔥 Real-world fire image classification
- 🔁 Multiple model experiments in one notebook
- 🧠 Uses transfer learning for better generalization
- ⚙️ Callbacks make training efficient and dynamic
- 🧪 Evaluation using confusion matrix and reports

---

## 📦 How to Reproduce

1. Prepare the dataset in the required folder structure (`Train/`, `Vali/`, `Test/`).
2. Upload the dataset to your Google Colab runtime.
3. Open and run `Fire_Detection.ipynb`.
4. Adjust epochs, batch size, or model selection as needed.
5. (Optional) Save trained models for deployment or testing.

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, or share it. Just credit the original author.

---

## 🙋 Author

**Saurav Singh Negi**  
🎓 B.Tech Student | 💻 Deep Learning Enthusiast  
🔗 [GitHub: SAURAV0703](https://github.com/SAURAV0703)
