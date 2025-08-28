# 🩺 Breast Cancer Detection using Neural Networks (PyTorch)

This project implements a **Neural Network classifier** to detect **breast cancer** (benign vs malignant tumors) using the **Breast Cancer Wisconsin Dataset**.  
It demonstrates the full pipeline of **data preprocessing, model building, training, and evaluation** in PyTorch.

---

## 📊 Dataset
- Source: [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  
- Features: 30 numerical attributes (e.g., radius, texture, perimeter).  
- Target:  
  - `0` → Malignant  
  - `1` → Benign  

---

## ⚙️ Project Workflow
1. **Data Preprocessing**
   - Train-test split (80/20).  
   - Standardization with `StandardScaler` for normalized features.  
   - Converted to **PyTorch tensors** and pushed to GPU (if available).  

2. **Model Architecture**
   - Input layer → 30 features.  
   - Hidden layer → 64 neurons, **ReLU** activation.  
   - Output layer → 1 neuron, **Sigmoid** activation (binary classification).  

3. **Training**
   - Loss: **Binary Cross-Entropy (BCELoss)**.  
   - Optimizer: **Adam** with learning rate = 0.001.  
   - Epochs: 100.  

4. **Evaluation**
   - Accuracy measured on test set.  
   - Typically achieves **90%+ accuracy**.  

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/your-username/breast-cancer-detector.git
cd breast-cancer-detector

# Install dependencies
pip install torch scikit-learn numpy

# Run the notebook
jupyter notebook breast_cancer.ipynb
