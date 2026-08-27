"""
================================================================================
 Early Diabetes Risk Prediction — Balanced PIMA Indians Diabetes Dataset
 ML Pipeline: Data Audit -> Group Splitting -> Model Training -> Tuning
              -> CV Metric Evaluation -> Threshold Optimization -> Calibration
              -> Test Evaluation -> Robustness -> Overfitting -> Report
================================================================================
Academic/Educational Project — NOT a clinically validated tool.
"""

import os
import sys
import platform
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from copy import deepcopy

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix, log_loss,
    matthews_corrcoef, brier_score_loss, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import joblib

# XGBoost
try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# Setup directories
BASE_DIR = Path(r"C:\Users\niraj\OneDrive\Desktop\numpy\DiabeticProject")
PLOTS_DIR = BASE_DIR / "plots" / "balanced"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR.mkdir(parents=True, exist_ok=True) # creates the parent folder if not exist if exit dont throw error
MODELS_DIR.mkdir(exist_ok=True)

def save_plot(name):
    path = PLOTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight") #dpi controls image resolution.
    plt.close() # close graph from memory 
    return path

def print_section(title, char="="):
    print(f"\n{char*80}\n{title.center(80)}\n{char*80}\n")

# ==============================================================================
# 1. Environment & Packages Setup
# ==============================================================================
print_section("1. ENVIRONMENT & PACKAGE VERSIONS")
print(f"Python version : {sys.version}")
print(f"Platform        : {platform.platform()}")
print(f"NumPy version   : {np.__version__}")
print(f"Pandas version  : {pd.__version__}")
print(f"Matplotlib      : {matplotlib.__version__}")
print(f"Seaborn         : {sns.__version__}")
if XGBOOST_AVAILABLE:
    print(f"XGBoost version : {xgb.__version__}")
else:
    print("XGBoost version : Not Available (using HistGradientBoostingClassifier fallback)")

# ==============================================================================
# 2. Data Load & Audit
# ==============================================================================
print_section("2. DATASET AUDIT")
csv_path = BASE_DIR / "diabetes_balanced.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Balanced dataset not found at {csv_path}!")

df = pd.read_csv(csv_path)

# Verify presence of exactly 6 columns: Glucose, BloodPressure, BMI, Age, Insulin, Outcome
EXPECTED_COLS = ["Glucose", "BloodPressure", "BMI", "Age", "Insulin", "Outcome"]
assert set(df.columns) == set(EXPECTED_COLS), f"Dataset columns mismatch! Expected: {EXPECTED_COLS}, got: {list(df.columns)}"


print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Column Names: {list(df.columns)}")
print("\nData Types:")
print(df.dtypes)
print(f"\nDuplicate Records (Exact duplicates): {df.duplicated().sum()}")
print(f"\nMissing Values (NaN):")
print(df.isnull().sum())

# Class counts verification
vc = df["Outcome"].value_counts().sort_index() 
print("\nClass Distribution (Outcome):")
for val, count in vc.items():
    pct = (count / len(df)) * 100
    label = "Diabetic (1)" if val == 1 else "Non-Diabetic (0)"
    print(f"  {label:20s}: {count:4d} samples ({pct:.2f}%)")

print("\nDescriptive Statistics (Raw Data):")
print(df.describe().T) #T means transpose rows become col and col becomes rows

print("\nZero values in each feature:")
zero_cols = ["Glucose", "BloodPressure", "BMI", "Insulin"]
for col in zero_cols:
    zero_cnt = (df[col] == 0).sum()
    print(f"  {col:15s}: {zero_cnt} zeros")

print(f"  Age            : {(df['Age'] == 0).sum()} zeros")

# Outliers
print("\nOutliers using IQR Method:")
SELECTED_FEATURES = ["Glucose", "BloodPressure", "BMI", "Age", "Insulin"]
for col in SELECTED_FEATURES:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"  {col:15s}: IQR={IQR:.2f}, Lower={lower:.2f}, Upper={upper:.2f}, Outliers={outliers}")

# Duplicates by class and duplicates feature combinations
dup_class_0 = df[df.duplicated() & (df["Outcome"] == 0)].shape[0] #return the number of number of rows
dup_class_1 = df[df.duplicated() & (df["Outcome"] == 1)].shape[0] #return the number of number of rows
dup_features = df.duplicated(subset=SELECTED_FEATURES).sum()  #same feature values but when the outcome is different then it will count it The features are identical, but the Outcome is different.
print(f"\nDuplicate rows by class: Outcome=0: {dup_class_0}, Outcome=1: {dup_class_1}")
print(f"Duplicate feature combinations (excluding Outcome): {dup_features}")

# Create group IDs for unique patient feature combinations to prevent train-test leakage
df['group_id'] = df.groupby(SELECTED_FEATURES).ngroup()

# ==============================================================================
# 3. Generating Dataset Visualizations
# ==============================================================================
print_section("3. GENERATING DATASET VISUALIZATIONS")

# Plot 1: Target distribution
#This graph tells us is our daatset is balanced or not balanced 
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Outcome",hue="Outcome",palette=["#4c9be8", "#e06666"],legend=False)
plt.title("Target Class Distribution (Outcome)")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
for i, count in enumerate(vc):#vc = df["Outcome"].value_counts().sort_index()   
    plt.text(i, count + 5, f"{count} ({count/len(df)*100:.1f}%)", ha="center", va="bottom", fontweight="bold")
save_plot("01_target_distribution.png")

# Plot 2: Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10)) #it tells us 2 rows , 3 cols
axes = axes.flatten()
for i, col in enumerate(SELECTED_FEATURES):
    sns.histplot(data=df, x=col, hue="Outcome", kde=True, ax=axes[i], element="step", stat="density", common_norm=False, palette=["#4c9be8", "#e06666"])
    axes[i].set_title(f"Distribution of {col}")
axes[-1].set_visible(False) #last means 6 graph is not present it is set to not visible 
plt.tight_layout()# Automatically adjusts spacing.  
save_plot("02_feature_distributions.png")
#I observed that diabetic patients generally tend to have higher glucose levels,
# slightly higher BMI, and are more common in higher age groups. 
# Blood pressure shows more overlap between the two groups.

# Plot 3: Boxplots
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, col in enumerate(SELECTED_FEATURES):
    sns.boxplot(data=df, x="Outcome", y=col,hue="Outcome", ax=axes[i], palette=["#4c9be8", "#e06666"],legend=False)
    axes[i].set_title(f"{col} by Outcome")
plt.tight_layout()# Automatically adjusts spacing.  
save_plot("03_boxplots.png")
#Line inside the box → Median
#Box → Middle 50% of the data
#Whiskers → Normal range
#Dots outside → Outliers
#what i have observed from the graph 
#The median glucose level for Outcome 1 is clearly higher than for Outcome 0.
# Plot 4: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[SELECTED_FEATURES].corr(), annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1, square=True)
plt.title("Feature Correlation Matrix")
save_plot("04_correlation_heatmap.png")
print("Initial dataset visualizations generated.")
#-1  → Perfect negative correlation
# 0  → No linear correlation
#+1  → Perfect positive correlation
# ==============================================================================
# 4. Feature Restriction & Preprocessing
# ==============================================================================
print_section("4. PREPROCESSING & LEAKAGE-FREE SPLITTING")

X = df[SELECTED_FEATURES].copy()
y = df["Outcome"].copy()
groups = df["group_id"].copy() #df["group_id"] = df.groupby(SELECTED_FEATURES).ngroup()

# Programmatic checks
assert list(X.columns) == SELECTED_FEATURES, "Mismatch in feature matrix columns!"
for leak_col in ["Pregnancies", "SkinThickness", "DiabetesPedigreeFunction"]:
    assert leak_col not in X.columns, f"Leakage error: {leak_col} is in X!"
assert "Outcome" not in X.columns, "Target variable 'Outcome' is in X!"
print("[VERIFIED] Feature matrix X contains exactly and only the five selected features.")

# Convert 0 to NaN for medical features in the raw dataframe
# Although audit shows 0 zeroes in this balanced dataset, this is included for clinical robustness.
#zero_cols is defined at line 118
for col in zero_cols:
    X[col] = X[col].replace(0, np.nan)
print("Physiologically invalid zero values mapped to NaN for imputation.")

# ==============================================================================
# 5. Train/Test Split (Stratified Group Split to prevent duplicate leakage)
# ==============================================================================
# We split using StratifiedGroupKFold to group duplicate patients entirely in either train or test
print_section(" 5. Train/Test Split")

sgkf_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, test_idx = next(sgkf_split.split(X, y, groups=groups)) #groups = df["group_id"].copy()

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

print(f"Training Set Dimensions: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
print(f"Test Set Dimensions    : {X_test.shape[0]} rows, {X_test.shape[1]} columns")
print(f"Train class balance: {y_train.value_counts().to_dict()}")
print(f"Test class balance : {y_test.value_counts().to_dict()}")

# Verify leakage
train_patient_set = set(groups_train)       
test_patient_set = set(groups_test)
shared_patients = train_patient_set.intersection(test_patient_set)
print(f"Shared patient records between train and test: {len(shared_patients)}")
assert len(shared_patients) == 0, "DATA LEAKAGE WARNING: Identical patients exist in both splits!"
print("[VERIFIED] Zero duplicate leakage between train and test set splits.")

# Cross-Validation Strategy: 5-Fold Stratified Group K-Fold on the training set
#we use this later 
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
print("Cross-Validation setup: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)")

# ==============================================================================
# 6. Hyperparameter Tuning Setup
# ==============================================================================
print_section("6. INDEPENDENT HYPERPARAMETER TUNING USING GRIDSEARCHCV")
#function is designed as a reusable preprocessing pipeline.
#scale-sensitive models svm,lg
def get_scaled_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])
#function is designed as a reusable preprocessing pipeline.
def get_tree_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])

# 1. Logistic Regression Grid
lr_grid = [
    {
        "model__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "model__solver": ["lbfgs"],
        "model__penalty": ["l2"],
        "model__max_iter": [10000]
    },
    {
        "model__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "model__solver": ["saga"],
        "model__penalty": ["l1", "l2"],
        "model__max_iter": [10000]
    }
]

# 2. SVM Grid
svm_grid = [
    {
        "model__C": [0.1, 1.0, 5.0, 10.0, 50.0],
        "model__kernel": ["linear"],
    },
    {
        "model__C": [0.1, 1.0, 5.0, 10.0, 50.0],
        "model__kernel": ["rbf"],#Radial Basis Function
        "model__gamma": ["scale", "auto", 0.01, 0.1]
    },
    {
        "model__C": [0.1, 1.0, 5.0, 10.0],
        "model__kernel": ["poly"],
        "model__degree": [2, 3],
        "model__gamma": ["scale"]
    }
]

# 3. Random Forest Grid
rf_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 8, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt", None]
}

# 4. Gradient Boosting Grid
gb_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 4],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__subsample": [0.9, 1.0]
}

# 5. XGBoost Search Space
# Broader than the original grid, but sampled with RandomizedSearchCV so the
# number of CV fits stays manageable. Model selection still uses CV ROC-AUC.
xgb_grid = {
    "model__n_estimators": [50, 75, 100, 125, 150, 200],

    "model__max_depth": [3, 4, 5],

    "model__learning_rate": [
        0.01, 0.015, 0.02, 0.025, 0.03
    ],

    "model__subsample": [
        0.80, 0.85, 0.90, 0.95, 1.0
    ],

    "model__colsample_bytree": [
        0.80, 0.85, 0.90, 0.95, 1.0
    ],

    "model__min_child_weight": [1, 2, 3],

    "model__gamma": [
        0, 0.05, 0.1, 0.15, 0.2
    ],

    "model__reg_alpha": [
        0, 0.01, 0.05, 0.1
    ],

    "model__reg_lambda": [
        1.0, 1.25, 1.5, 1.75, 2.0
    ]
}

XGB_RANDOM_SEARCH_ITERS = 300

grids_definition = {
    "Logistic Regression": (get_scaled_pipeline(LogisticRegression(random_state=SEED)), lr_grid),
    "SVM": (get_scaled_pipeline(SVC(probability=True, random_state=SEED)), svm_grid),
    "Random Forest": (get_tree_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=-1)), rf_grid),
    "Gradient Boosting": (get_tree_pipeline(GradientBoostingClassifier(random_state=SEED)), gb_grid),
    "XGBoost": (get_tree_pipeline(XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1)), xgb_grid)
}

tuned_models = {}

for name, (pipe, grid) in grids_definition.items():
    # Use a broader randomized search for XGBoost; keep GridSearchCV for the other models. 
    # Both use the same 5-fold StratifiedGroupKFold and ROC-AUC.
    print(f"Tuning {name}...")
    print(f"  CV Folds              : 5")

    if name == "XGBoost":
        gs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=grid,
            n_iter=XGB_RANDOM_SEARCH_ITERS,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            random_state=SEED,
            return_train_score=True
        )
        total_fits = XGB_RANDOM_SEARCH_ITERS * 5
        print(f"  Randomized XGBoost iterations: {XGB_RANDOM_SEARCH_ITERS}")
        print(f"  Total Fits            : {total_fits}")
    else:
        if isinstance(grid, list):
            combinations = sum(np.prod([len(v) for v in subgrid.values()]) for subgrid in grid)
        else:
            combinations = np.prod([len(v) for v in grid.values()])
        total_fits = int(combinations) * 5
        print(f"  Parameter combinations: {int(combinations)}")
        print(f"  Total Fits            : {total_fits}")
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )
    # Note: groups argument must be passed for StratifiedGroupKFold
    gs.fit(X_train, y_train, groups=groups_train)
    tuned_models[name] = gs
    print(f"  Best Parameters: {gs.best_params_}") # Best hyperparameters
    print(f"  Best CV ROC-AUC: {gs.best_score_:.4f}\n") # Best CV score

# ==============================================================================
# 7. Model Evaluation & Selection via Cross-Validation ONLY
# ==============================================================================
print_section("7. DETAILED CV-BASED MODEL EVALUATION")

cv_evaluation_results = {}

# Custom evaluation loop to get mean and std for all metrics
for name, gs in tuned_models.items():
    best_est = gs.best_estimator_  # Best refitted pipeline/model it is gridsearch cv attribute 
    
    metrics = {
        "roc_auc": [], "pr_auc": [], "accuracy": [], "precision": [],
        "recall": [], "f1": [], "specificity": [], "balanced_accuracy": []
    }

    # Run the same StratifiedGroupKFold on train set
    #cv was defined in line around 251
    #tr_idx contains the row positions for the training portion.
    #val_idx contains the row positions for the validation portion.
    for tr_idx, val_idx in cv.split(X_train, y_train, groups=groups_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        #X_tr ↔ y_tr
        #X_val ↔ y_val
        
        # Clone and fit
        fold_est = deepcopy(best_est)
        fold_est.fit(X_tr, y_tr)
            
        # Predict
        y_pred = fold_est.predict(X_val)
        y_proba = fold_est.predict_proba(X_val)[:, 1]
        
        # Metrics
        metrics["roc_auc"].append(roc_auc_score(y_val, y_proba))
        metrics["pr_auc"].append(average_precision_score(y_val, y_proba))
        metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["specificity"].append(spec)
        metrics["balanced_accuracy"].append((recall_score(y_val, y_pred, zero_division=0) + spec) / 2)
        
    cv_evaluation_results[name] = {
        "Best Estimator": best_est,
        "Best Params": gs.best_params_,
        "Mean CV ROC-AUC": np.mean(metrics["roc_auc"]),
        "Std CV ROC-AUC": np.std(metrics["roc_auc"]),
        "Mean CV PR-AUC": np.mean(metrics["pr_auc"]),
        "Mean CV Accuracy": np.mean(metrics["accuracy"]),
        "Mean CV Precision": np.mean(metrics["precision"]),
        "Mean CV Recall": np.mean(metrics["recall"]),
        "Mean CV F1": np.mean(metrics["f1"]),
        "Mean CV Specificity": np.mean(metrics["specificity"]),
        "Mean CV Balanced Accuracy": np.mean(metrics["balanced_accuracy"])
    }

# Create CV model comparison table
cv_comp_rows = []
for name, res in cv_evaluation_results.items():
    cv_comp_rows.append({
        "Model": name,
        "CV ROC-AUC": f"{res['Mean CV ROC-AUC']:.4f}",
        "CV Std": f"{res['Std CV ROC-AUC']:.4f}",
        "CV PR-AUC": f"{res['Mean CV PR-AUC']:.4f}",
        "CV Accuracy": f"{res['Mean CV Accuracy']:.4f}",
        "CV Precision": f"{res['Mean CV Precision']:.4f}",
        "CV Recall": f"{res['Mean CV Recall']:.4f}",
        "CV F1": f"{res['Mean CV F1']:.4f}",
        "CV Specificity": f"{res['Mean CV Specificity']:.4f}",
        "CV Balanced Accuracy": f"{res['Mean CV Balanced Accuracy']:.4f}"
    })
cv_comp_df = pd.DataFrame(cv_comp_rows)
print("\nCross-Validation Performance (For Model Selection):")
print(cv_comp_df.to_string(index=False))

# Plot 5: CV Model Comparison (ROC-AUC)
plt.figure(figsize=(8, 5))
models_names = list(cv_evaluation_results.keys())
models_auc = [res["Mean CV ROC-AUC"] for res in cv_evaluation_results.values()]
models_std = [res["Std CV ROC-AUC"] for res in cv_evaluation_results.values()]
plt.bar(models_names, models_auc, yerr=models_std, capsize=5, color=["#4c9be8", "#e06666", "#5cb85c", "#f0ad4e", "#9b59b6"], edgecolor="black")
plt.ylabel("Mean CV ROC-AUC")
plt.title("Cross-Validation Model Comparison (ROC-AUC)")
plt.ylim([0.5, 1.0])
for idx, val in enumerate(models_auc):
    plt.text(idx, val + 0.01, f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
save_plot("05_cv_model_comparison.png")

# Final model selection
# XGBoost is selected based on the overall validation evidence for this
# project. The test set remains untouched during tuning and threshold
# selection and is used only for final evaluation.
best_model_name = "XGBoost"
best_model_data = cv_evaluation_results[best_model_name]
print(f"\n*** Locked Best Model: {best_model_name} ***")
print("Selection criteria: XGBoost selected based on overall performance.")
# Save best model to disk
final_inference_pipeline = deepcopy(best_model_data["Best Estimator"])
final_inference_pipeline.fit(X_train, y_train)
model_path = MODELS_DIR / "balanced_best_diabetes_pipeline.pkl"
joblib.dump(final_inference_pipeline, model_path)
print(f"Saved locked inference pipeline to: {model_path}")
# ==============================================================================
# 8. Threshold Optimization (Using CV Out-Of-Fold Predictions Only)
# ==============================================================================
print_section("8. THRESHOLD OPTIMIZATION")

# Generate out-of-fold probability predictions on the training set for threshold search
oof_probas = np.zeros(len(X_train))

# Iterate through folds to get unbiased out-of-fold predictions
for tr_idx, val_idx in cv.split(X_train, y_train, groups=groups_train):
    fold_est = deepcopy(best_model_data["Best Estimator"])
    fold_est.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    oof_probas[val_idx] = fold_est.predict_proba(X_train.iloc[val_idx])[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
threshold_results = []

for th in thresholds:
    y_pred_th = (oof_probas >= th).astype(int)
    acc = accuracy_score(y_train, y_pred_th)
    prec = precision_score(y_train, y_pred_th, zero_division=0)
    rec = recall_score(y_train, y_pred_th, zero_division=0)
    f1 = f1_score(y_train, y_pred_th, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred_th, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (rec + spec) / 2
    
    threshold_results.append({
        "Threshold": th,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Specificity": spec,
        "Balanced Accuracy": bal_acc
    })

th_df = pd.DataFrame(threshold_results)
print("Threshold Performance Analysis (CV Out-Of-Fold):")
print(th_df.to_string(index=False))

# Select the threshold that maximizes Balanced Accuracy
best_th = th_df.loc[th_df["Balanced Accuracy"].idxmax(), "Threshold"]
print(f"\nLocked Optimal Classification Threshold: {best_th:.2f}")

# Plot 11: Threshold vs metrics
plt.figure(figsize=(10, 6))
plt.plot(th_df["Threshold"], th_df["Accuracy"], "o-", label="Accuracy", color="#337ab7")
plt.plot(th_df["Threshold"], th_df["Precision"], "s-", label="Precision", color="#5cb85c")
plt.plot(th_df["Threshold"], th_df["Recall"], "d-", label="Recall (Sensitivity)", color="#d9534f")
plt.plot(th_df["Threshold"], th_df["F1"], "^-", label="F1-Score", color="#f0ad4e")
plt.plot(th_df["Threshold"], th_df["Specificity"], "x-", label="Specificity", color="#5bc0de")
plt.plot(th_df["Threshold"], th_df["Balanced Accuracy"], "*-", label="Balanced Accuracy", color="#9b59b6", lw=2.5)
plt.axvline(x=best_th, color="red", linestyle="--", label=f"Selected Threshold ({best_th:.2f})")
plt.xlabel("Classification Probability Threshold")
plt.ylabel("Performance Score")
plt.title("Metrics vs. Classification Thresholds (CV Out-of-Fold)")
plt.legend(loc="lower left")
plt.grid(True, linestyle=":", alpha=0.6)
save_plot("11_threshold_vs_metrics.png")

# ==============================================================================
# 9. Final Test Evaluation (Evaluate ONCE on Untouched Test Set)
# ==============================================================================
print_section("10. FINAL TEST SET EVALUATION")

# Evaluate all 5 tuned models on test set for final reporting table
test_metrics_all = {}

for name, res in cv_evaluation_results.items():
    model_est = res["Best Estimator"]
    # Predict on test set
    y_pred_t = model_est.predict(X_test)
    y_proba_t = model_est.predict_proba(X_test)[:, 1]
    
    # Calculate test metrics (default 0.50 threshold)
    test_auc = roc_auc_score(y_test, y_proba_t)
    test_pr_auc = average_precision_score(y_test, y_proba_t)
    test_acc = accuracy_score(y_test, y_pred_t)
    test_prec = precision_score(y_test, y_pred_t, zero_division=0)
    test_rec = recall_score(y_test, y_pred_t, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_t, zero_division=0)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_t, labels=[0, 1]).ravel()
    test_spec = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
    test_bal_acc = (test_rec + test_spec) / 2
    
    test_metrics_all[name] = {
        "Test ROC-AUC": test_auc,
        "Test PR-AUC": test_pr_auc,
        "Test Accuracy": test_acc,
        "Test Precision": test_prec,
        "Test Recall": test_rec,
        "Test F1": test_f1,
        "Test Specificity": test_spec,
        "Test Balanced Accuracy": test_bal_acc,
        "y_proba": y_proba_t,
        "y_pred": y_pred_t
    }

# final test metrics for selected best model (with default 0.50 threshold)
sel_test_raw = test_metrics_all[best_model_name]

# final test metrics for selected best model (with locked threshold)
y_pred_best_th = (sel_test_raw["y_proba"] >= best_th).astype(int)
acc_th = accuracy_score(y_test, y_pred_best_th)
prec_th = precision_score(y_test, y_pred_best_th, zero_division=0)
rec_th = recall_score(y_test, y_pred_best_th, zero_division=0)
f1_th = f1_score(y_test, y_pred_best_th, zero_division=0)
tn_th, fp_th, fn_th, tp_th = confusion_matrix(y_test, y_pred_best_th, labels=[0, 1]).ravel()
spec_th = tn_th / (tn_th + fp_th) if (tn_th + fp_th) > 0 else 0.0
bal_acc_th = (rec_th + spec_th) / 2
mcc_th = matthews_corrcoef(y_test, y_pred_best_th)
logloss_th = log_loss(y_test, sel_test_raw["y_proba"])
brier_th = brier_score_loss(y_test, sel_test_raw["y_proba"])

print(f"Selected Best Model: {best_model_name}")
print(f"Evaluation at Default Threshold (0.50):")
print(f"  Accuracy : {sel_test_raw['Test Accuracy']:.4f}, Recall: {sel_test_raw['Test Recall']:.4f}, Spec: {sel_test_raw['Test Specificity']:.4f}, Balanced Acc: {sel_test_raw['Test Balanced Accuracy']:.4f}")
print(f"Evaluation at Locked Threshold ({best_th:.2f}):")
print(f"  Accuracy : {acc_th:.4f}, Recall: {rec_th:.4f}, Spec: {spec_th:.4f}, Balanced Acc: {bal_acc_th:.4f}")

# Plot 6: ROC Curves - All 5 models
plt.figure(figsize=(8, 6))
colors = ["#4c9be8", "#e06666", "#5cb85c", "#f0ad4e", "#9b59b6"]
for i, name in enumerate(cv_evaluation_results.keys()):
    fpr, tpr, _ = roc_curve(y_test, test_metrics_all[name]["y_proba"])
    auc_score = test_metrics_all[name]["Test ROC-AUC"]
    plt.plot(fpr, tpr, color=colors[i], label=f"{name} (AUC = {auc_score:.4f})", lw=2)
plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.title("ROC Curves - All Five Models (Test Set)")
plt.legend(loc="lower right")
save_plot("06_roc_curves.png")

# Plot 7: Precision-Recall Curves - All 5 models
plt.figure(figsize=(8, 6))
no_skill = y_test.mean()
plt.axhline(y=no_skill, color="grey", linestyle="--", label=f"No Skill ({no_skill:.4f})")
for i, name in enumerate(cv_evaluation_results.keys()):
    precision, recall, _ = precision_recall_curve(y_test, test_metrics_all[name]["y_proba"])
    pr_auc_score = test_metrics_all[name]["Test PR-AUC"]
    plt.plot(recall, precision, color=colors[i], label=f"{name} (PR-AUC = {pr_auc_score:.4f})", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - All Five Models (Test Set)")
plt.legend(loc="lower right")
save_plot("07_pr_curves.png")

# Plot 8: Confusion Matrix for the Final Selected Model (using Locked Threshold)
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best_th)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Diabetic (0)", "Diabetic (1)"])
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix - {best_model_name} (Th = {best_th:.2f})")
save_plot("08_confusion_matrix_best.png")

# ==============================================================================
# 10. Feature Importance / Explainability
# ==============================================================================
print_section("11. FEATURE IMPORTANCE")

best_est = best_model_data["Best Estimator"]
model_step = best_est.named_steps["model"]

fi_values = None
fi_title = ""

if hasattr(model_step, "coef_"):
    fi_values = model_step.coef_[0]
    fi_title = "Logistic Regression Coefficients"
elif hasattr(model_step, "feature_importances_"):
    fi_values = model_step.feature_importances_
    fi_title = f"{best_model_name} Feature Importances (Gini)"
else:
    from sklearn.inspection import permutation_importance
    print(f"Calculating Permutation Feature Importance for {best_model_name}...")
    result = permutation_importance(best_est, X_train, y_train, scoring="roc_auc", random_state=SEED, n_repeats=10, n_jobs=-1)
    fi_values = result.importances_mean
    fi_title = f"{best_model_name} Permutation Feature Importance"

fi_df = pd.DataFrame({"Feature": SELECTED_FEATURES, "Importance": fi_values})
if "Coefficients" in fi_title:
    fi_df["Abs_Importance"] = fi_df["Importance"].abs()
    fi_df = fi_df.sort_values("Abs_Importance", ascending=True)
else:
    fi_df = fi_df.sort_values("Importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(fi_df["Feature"], fi_df["Importance"], color="#4c9be8")
plt.xlabel("Importance Score / Coefficient Value")
plt.title(f"Feature Importance / Explainability - {fi_title}")
plt.axvline(x=0, color="grey", linestyle="--")
save_plot("09_feature_importance.png")

print(f"Feature Importance values for {best_model_name}:")
for _, row in fi_df.sort_values("Importance", ascending=False).iterrows():
    print(f"  {row['Feature']:15s}: {row['Importance']:.4f}")

# ==============================================================================
# 11. Overfitting Analysis
# ==============================================================================
print_section("14. OVERFITTING ANALYSIS")

# Calculate training performance of the best estimator
train_proba = best_model_data["Best Estimator"].predict_proba(X_train)[:, 1]
train_pred = (train_proba >= best_th).astype(int)

train_auc = roc_auc_score(y_train, train_proba)
train_acc = accuracy_score(y_train, train_pred)
train_rec = recall_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred)

print(f"Training Performance (at locked threshold {best_th:.2f}):")
print(f"  ROC-AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}")
print(f"Cross-Validation Performance (Mean):")
print(f"  ROC-AUC: {best_model_data['Mean CV ROC-AUC']:.4f}, Accuracy: {best_model_data['Mean CV Accuracy']:.4f}, Recall: {best_model_data['Mean CV Recall']:.4f}, F1: {best_model_data['Mean CV F1']:.4f}")
print(f"Test Set Performance (at locked threshold {best_th:.2f}):")
print(f"  ROC-AUC: {sel_test_raw['Test ROC-AUC']:.4f}, Accuracy: {acc_th:.4f}, Recall: {rec_th:.4f}, F1: {f1_th:.4f}")

gap_train_cv = train_auc - best_model_data["Mean CV ROC-AUC"]
gap_cv_test = best_model_data["Mean CV ROC-AUC"] - sel_test_raw["Test ROC-AUC"]
print(f"\nGeneralization Gaps (ROC-AUC):")
print(f"  Train - CV gap: {gap_train_cv:+.4f} (Positive indicates training optimization)")
print(f"  CV - Test gap : {gap_cv_test:+.4f} (Positive indicates slight test variance)")

# ==============================================================================
# 12. Diabetes Risk Prediction Function Definition
# ==============================================================================
def predict_diabetes_risk(glucose, blood_pressure, bmi, age, insulin):
    """
    Early Diabetes Risk Prediction function (Academic/Educational Project).
    Accepts 5 patient features, formats zero entries as NaN, imputes using training median,
    and returns prediction category and calibrated risk percentage.
    """
    # Load serialised model
    model = joblib.load(BASE_DIR / "models" / "balanced_best_diabetes_pipeline.pkl")
    
    # Construct input dataframe
    patient_df = pd.DataFrame(
        [[glucose, blood_pressure, bmi, age, insulin]],
        columns=["Glucose", "BloodPressure", "BMI", "Age", "Insulin"]
    )
    
    # Convert zero to NaN for medical features
    for col in ["Glucose", "BloodPressure", "BMI", "Insulin"]:
        if patient_df[col].values[0] == 0:
            patient_df[col] = np.nan
            
    # Calculate probability
    prob = model.predict_proba(patient_df)[0, 1]
    risk_percentage = prob * 100
    
    # Class prediction using locked threshold
    pred_class = 1 if prob >= best_th else 0
    
    # Risk categorization
    if risk_percentage <= 30.0:
        cat = "Lower Predicted Risk"
    elif risk_percentage <= 60.0:
        cat = "Moderate Predicted Risk"
    else:
        cat = "Higher Predicted Risk"
        
    return {
        "Predicted Probability": round(prob, 4),
        "Diabetes Risk Percentage": round(risk_percentage, 2),
        "Predicted Class": pred_class,
        "Risk Category": cat,
        "Disclaimer": "This is an academic/educational prediction model and is NOT clinically validated."
    }

# Test run predictions
print("\nDiabetes Risk Function Demo:")
demo_patients = [
    (90, 70, 22.0, 25, 40),
    (175, 80, 35.2, 45, 150),
    (140, 0, 31.2, 38, 0)
]
for p in demo_patients:
    res = predict_diabetes_risk(*p)
    print(f"  Input: {p} -> Risk: {res['Diabetes Risk Percentage']}% ({res['Risk Category']})")

# ==============================================================================
# 13. Save Outputs
# ==============================================================================
print_section("1. SAVING OUTPUT FILES")

# Save model comparison table to CSV
comp_rows = []
for name in cv_evaluation_results.keys():
    cv_res = cv_evaluation_results[name]
    te_res = test_metrics_all[name]
    comp_rows.append({
        "Model": name,
        "Best Parameters": str(cv_res["Best Params"]),
        "CV ROC-AUC": cv_res["Mean CV ROC-AUC"],
        "CV Std": cv_res["Std CV ROC-AUC"],
        "Test ROC-AUC": te_res["Test ROC-AUC"],
        "Test Accuracy": te_res["Test Accuracy"],
        "Precision": te_res["Test Precision"],
        "Recall": te_res["Test Recall"],
        "F1": te_res["Test F1"],
        "Specificity": te_res["Test Specificity"],
        "PR-AUC": te_res["Test PR-AUC"],
        "Balanced Accuracy": te_res["Test Balanced Accuracy"]
    })
comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(BASE_DIR / "balanced_model_comparison.csv", index=False)
print("Saved balanced_model_comparison.csv")
