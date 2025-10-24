# 🚗 Vehicle Type Classification (UCI Dataset)

This project uses the **UCI Statlog (Vehicle Silhouettes)** dataset to classify vehicles into four types: `bus`, `opel`, `saab`, and `van`, using geometric features extracted from their silhouettes.

---

## ⚙️ Overview

- **Dataset:** 846 samples, 18 numeric features  
- **Goal:** Multi-class classification  
- **Models tested:** KNN, SVM, Random Forest, MLP  
- **Preprocessing:** Label encoding, scaling (`StandardScaler`, `MinMaxScaler`, `RobustScaler`)  
- **Optimization:** `GridSearchCV` / `RandomizedSearchCV` with pipelines  
- **Evaluation:** Accuracy, confusion matrix, classification report  

---

## 🧠 Best Result

| Model | Scaler | Accuracy |
|--------|---------|-----------|
| SVC (RBF) | StandardScaler | **≈ 83–85%** |

The dataset is small and noisy; top realistic accuracy is around **90%**.

---

