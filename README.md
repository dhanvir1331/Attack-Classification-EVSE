# ⚡ EVSE Cyberattack Detection using ML & DL 🚗🔒

A full-scale machine learning and deep learning pipeline for detecting cyberattacks on **Electric Vehicle Supply Equipment (EVSE)** systems. This project showcases feature selection, preprocessing, classical ML models, PCA, DL architectures (CNN, ResNet, etc.), transfer learning, and comparative evaluation — all on a real-world dataset.

---

## 📂 Table of Contents

- [📖 Project Description](#-project-description)
- [📊 Dataset Overview](#-dataset-overview)
- [🔧 Project Architecture](#-project-architecture)
- [📈 Models Implemented](#-models-implemented)
- [🔍 Feature Engineering](#-feature-engineering)
- [🏋️ Model Training & Evaluation](#-model-training--evaluation)
- [📊 Visualization Dashboard](#-visualization-dashboard)
- [🚀 Results & Insights](#-results--insights)
- [💡 Future Work](#-future-work)
- [📚 References](#-references)

---

## 📖 Project Description

With the rise of **smart mobility infrastructure**, EV charging stations are becoming critical components of modern power grids. However, they are increasingly vulnerable to **cyberattacks**, including DDoS, spoofing, and interface manipulation.

This project builds a **robust classification system** using both classical and deep learning approaches to:
- Detect cyberattacks vs. benign activity
- Interpret features using Point-Biserial correlation
- Optimize model performance with PCA and transfer learning
- Visualize performance and decision boundaries

---

## 📊 Dataset Overview

- 📁 **File**: `EVSE-B-HPC-Kernel-Events-Combined.csv`
- 🔢 **Samples**: ~100,000+
- 🏷️ **Classes**: `Label = 0 (Benign), 1 (Attack)`
- 📐 **Features**: 800+ including CPU counters, memory metrics, I/O loads, system calls, etc.
- 🧹 **Cleaned**: Removed low-variance features and handled nulls with intelligent strategies

---

## 🔧 Project Architecture

```bash
📦 EVSE_Cybersecurity_Detector
├── 📁 data/
│   └── Cleaned_EVSE_Data.csv
├── 📁 notebooks/
│   └── Final_EVSE_attack_classification_nb.ipynb
├── 📊 plots/
│   └── model_comparison.png
├── 📄 README.md
```

---

## 📈 Models Implemented

### 🧠 Classical ML Models
- ✅ Logistic Regression
- 🌳 Random Forest
- 🧮 Support Vector Machine (SVM)
- 👥 K-Nearest Neighbors
- ⚡ XGBoost

### 🤖 Deep Learning Models
- 🧱 Basic ANN
- 🌀 CNN (Shallow & Deep)
- 🔁 LSTM (Optional)
- 🏗️ ResNet-style Network
- 🧠 Transfer Learning with MobileNetV2

---

## 🔍 Feature Engineering

- ✅ Removed low-information numeric features (low variance or <3 unique values)
- 🔢 Applied `StandardScaler` to normalize inputs
- 📉 Applied PCA (`n_components=0.78`) to retain ~78% variance and reduce noise
- 📊 Used **Point-Biserial Correlation** to select statistically impactful features

---

## 🏋️ Model Training & Evaluation

Each model was:
- Trained on an 80-20 stratified split
- Validated using classification metrics:
  - Accuracy
  - Precision, Recall, F1-Score (Per Class)
  - Confusion Matrix
- For DL models:
  - Early Stopping
  - Dropout Regularization
  - L2 penalty
  - Epoch-wise tracking for training vs validation

---

## 📊 Visualization Dashboard

- ✅ Confusion matrices for every model
- 📈 Accuracy vs Epochs (for DL)
- 📉 Loss curves
- 📊 Side-by-side bar plots for:
  - Accuracy
  - Precision / Recall / F1 per class
- 🔁 ML vs DL comparative charts

![ML vs DL](plots/model_comparison.png)

---

## 🚀 Results & Insights

| Model               | Accuracy | Precision (1) | Recall (1) | F1 (1) |
|---------------------|----------|----------------|------------|--------|
| Logistic Regression | 87.93%   | 93%            | 88%        | 90%    |
| Random Forest       | 98.14%   | 98%            | 99%        | 99%    |
| SVM                 | 95.71%   | 98%            | 95%        | 97%    |
| KNN                 | 98.46%   | 99%            | 99%        | 99%    |
| XGBoost             | 98.38%   | 99%            | 98%        | 99%    |
| **CNN (Deep)**      | 97.50%   | 99%            | 98%        | 99%    |
| **Transfer (TF)**   | **98.80%** | **99%**        | **99%**    | **99%** |

✅ **Transfer Learning with MobileNetV2** provided the best generalization on unseen data.

---

## 💡 Future Work

- ⚡ Deploy model in edge devices or Raspberry Pi
- 🔐 Build an online intrusion detection dashboard
- 📦 Automate real-time streaming data classification
- 🎯 Use attention-based transformers to detect complex temporal patterns
- 🔍 Interpretability using SHAP or LIME

---

## 📚 References

- [Point-Biserial Correlation — SciPy Docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pointbiserialr.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Keras Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
- EVSE Cybersecurity Research Papers and Datasets (Anonymized source)

---

> _"Cybersecurity in EVs is not optional—it's foundational."_  
> — This project is a step toward building trustworthy transportation infrastructure.
