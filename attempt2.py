
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits as Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler ,Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score,GridSearchCV,ParameterGrid,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm.notebook import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
import time
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', None)  # Show all columns


def estimate_gridsearch_time(pipe, param_grid, X, y, cv=5, sample_size=50):
    """
    est_minutes : float
        Estimated total runtime in minutes.
    """

    # Create parameter grid
    grid_list = list(ParameterGrid(param_grid))
    n_combos = len(grid_list)

    # Take a small timing sample
    sample_params = grid_list[0]
    pipe.set_params(**sample_params)

    start = time.time()
    pipe.fit(X[:sample_size], y[:sample_size])  # quick sample fit
    elapsed = time.time() - start

    # Estimate total runtime (rough)
    est_total = elapsed * n_combos * cv
    est_minutes = est_total / 60

    print(f"🔹 Estimated parameter combinations: {n_combos}")
    print(f"🔹 Cross-validation folds: {cv}")
    print(f"⏳ Estimated total runtime: ~{est_minutes:.1f} minutes "
          f"(based on {sample_size} samples)")

    return est_minutes







#-------------------------#-------------------#-------------------
#-------------------------#-------------------#-------------------

#LOAD DATASET
dataset =fetch_openml("vehicle", version=1, as_frame=True).frame
#print(dataset)




#X,Y 
X = dataset.drop( columns = dataset.columns[-1])
y = dataset[dataset.columns[-1]]


#One hot encoding
le = LabelEncoder()
y_encoded = le.fit_transform(dataset['Class'])
print(pd.Series(y_encoded).value_counts())



# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y) #80-20% split



pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier())
])

param_grid = [
    {
        'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), Normalizer()],
        'model': [KNeighborsClassifier()],
        'model__n_neighbors': list(range(1, 21, 2)),
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    },
    {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'model': [SVC()],
        'model__C': np.logspace(-3, 2, 6),
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    },
    {
        'scaler': [StandardScaler()],
        'model': [RandomForestClassifier()],
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, None],
        'model__min_samples_split': [2, 5]
    },
    {
        'scaler': [StandardScaler()],
        'model': [LogisticRegression(max_iter=1000)],
        'model__C': np.logspace(-3, 2, 6),
        'model__penalty': ['l2']
    },
    {
        'scaler': [StandardScaler()],
        'model': [GradientBoostingClassifier()],
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2]
    }
]



grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    error_score=0
)

# GRID SEARCH
est_minutes = estimate_gridsearch_time(pipe, param_grid, X_train, y_train, cv=5)
grid.fit(X_train, y_train)

print("Best scaler:", grid.best_params_['scaler'])
print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", round(grid.best_score_, 3))
print("Test accuracy:", round(grid.score(X_test, y_test), 3))





#BEST MODEL
best_model = grid.best_estimator_

# Predict on test data
y_pred = best_model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))







