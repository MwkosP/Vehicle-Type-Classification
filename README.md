# 🚗 Vehicle Type Classification (UCI Vehicle Silhouettes)

A Machine Learning project that classifies vehicles into four categories (`bus`, `opel`, `saab`, `van`) using geometric silhouette features from the **UCI Statlog (Vehicle Silhouettes)** dataset.  
This project demonstrates a complete ML workflow — from data preprocessing to model evaluation and hyperparameter tuning using pipelines.

---

## 📁 Dataset

**Source:** [UCI Machine Learning Repository – Vehicle Silhouettes](https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes))

- **Samples:** 846  
- **Features:** 18 continuous numeric attributes (shape, ratios, symmetry)  
- **Classes:** `bus`, `opel`, `saab`, `van` (balanced)  

These features describe geometric properties of vehicle silhouettes extracted from images.  
The dataset is small and noisy, making it a good challenge for classical ML algorithms.

---

## ⚙️ Project Overview

This project follows a **modular, scalable ML pipeline**:

1. **Data Loading** – Load dataset via `sklearn.datasets.load_wine` or from CSV.
2. **Exploration** – Check feature distributions, missing values, and class balance.
3. **Preprocessing**
   - Encoding categorical labels (`LabelEncoder` or `OneHotEncoder`)
   - Feature scaling (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, etc.)
4. **Model Training**
   - Main model: `KNeighborsClassifier`
   - Optional: `SVC`, `RandomForestClassifier`, `MLPClassifier`
5. **Model Evaluation**
   - Accuracy, confusion matrix, and classification report
6. **Hyperparameter Tuning**
   - `GridSearchCV` / `RandomizedSearchCV` with full pipeline
   - Tested multiple scalers and distance metrics
7. **Visualization**
   - 3D scatter plots using PCA + Plotly
   - Confusion matrix heatmap

---

## 🧠 Key Techniques Used

- **Scikit-Learn Pipelines** for streamlined preprocessing + model steps  
- **Cross-Validation (cv=5)** for robust performance estimates  
- **Grid Search** to tune:
  - `n_neighbors`, `weights`, `metric`, `p`, `leaf_size`, `algorithm`
  - Multiple **scalers** in the same search  
- **StandardScaler**, **MinMaxScaler**, **RobustScaler**, **Normalizer**  
- **PCA (3D Visualization)** for dimensionality reduction  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 🧩 Best Results

| Model | Scaler | Accuracy (Test) | Notes |
|--------|---------|-----------------|-------|
| `SVC (RBF)` | `StandardScaler()` | **≈ 0.835 (83.5%)** | Strong overall performance |
| `KNN` | `MinMaxScaler()` | ≈ 0.80–0.82 | Stable, interpretable |
| `RandomForest` | `None` | ≈ 0.86–0.88 | Top classical result |
| `MLPClassifier` | `StandardScaler()` | ≈ 0.87–0.89 | Slightly better, more complex |

⚠️ Typical ceiling accuracy on this dataset (based on literature) is **≈ 90–91%** due to data noise and overlapping classes.

---

## 🧾 Example Output

