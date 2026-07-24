# %% [markdown]
# # Assignment 4 — Breast Cancer Classification using K-Nearest Neighbors (KNN)
#
# **Objective:** Build a KNN classifier to predict whether a breast tumor is
# Malignant (M) or Benign (B) based on diagnostic measurements from the
# Breast Cancer Wisconsin (Diagnostic) Dataset.
#
# Dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

# %% [markdown]
# ## Task 1: Data Understanding (2 Marks)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

# %% [markdown]
# ### Loading the dataset
#
# Place the Kaggle CSV (`data.csv`, downloaded from the link above) in the
# same folder as this notebook. If it is not found, the notebook falls back
# to scikit-learn's bundled copy of the same Breast Cancer Wisconsin
# Diagnostic dataset so the notebook still runs end-to-end.

# %%
def load_breast_cancer_dataframe(csv_path="data.csv"):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df

    # Fallback: reconstruct the same dataset from sklearn with matching
    # Kaggle-style column names (radius_mean, texture_mean, ..., worst).
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()

    def rename(col):
        if col.startswith("mean "):
            return col.replace("mean ", "") .replace(" ", "_") + "_mean"
        if col.startswith("worst "):
            return col.replace("worst ", "").replace(" ", "_") + "_worst"
        if col.endswith(" error"):
            return col.replace(" error", "").replace(" ", "_") + "_se"
        return col

    feature_cols = [c for c in df.columns if c != "target"]
    df = df.rename(columns={c: rename(c) for c in feature_cols})

    # sklearn target: 0 = malignant, 1 = benign
    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})
    df = df.drop(columns=["target"])
    df.insert(0, "id", range(842302, 842302 + len(df)))
    cols = ["id", "diagnosis"] + [c for c in df.columns if c not in ("id", "diagnosis")]
    df = df[cols]
    return df


df = load_breast_cancer_dataframe()
print(f"Dataset shape: {df.shape}")
df.head()

# %% [markdown]
# ### First five records

# %%
df.head()

# %% [markdown]
# ### Identifying numerical features and the target variable

# %%
target_variable = "diagnosis"
numerical_features = [
    c for c in df.columns if c not in (target_variable, "id") and c != "Unnamed: 32"
]

print(f"Target variable: {target_variable}")
print(f"Number of numerical features: {len(numerical_features)}")
print("Numerical features:")
for f in numerical_features:
    print(" -", f)

# %% [markdown]
# ### Dataset info and summary statistics

# %%
df.info()

# %%
df.describe()

# %%
print("Class distribution:")
print(df[target_variable].value_counts())
sns.countplot(x=target_variable, data=df, palette="Set2")
plt.title("Diagnosis Class Distribution (M = Malignant, B = Benign)")
plt.show()

# %% [markdown]
# ## Task 2: Data Preprocessing (2 Marks)

# %% [markdown]
# ### Check for missing values

# %%
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

# %% [markdown]
# ### Remove unnecessary columns
#
# `id` carries no predictive information, and the Kaggle CSV sometimes ships
# a trailing all-NaN `Unnamed: 32` column caused by a stray comma in the
# source file — both are dropped if present.

# %%
drop_cols = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
df = df.drop(columns=drop_cols)
print(f"Dropped columns: {drop_cols}")
print(f"Remaining shape: {df.shape}")

# %% [markdown]
# ### Encode the target variable
#
# Malignant (M) -> 1, Benign (B) -> 0

# %%
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])  # B=0, M=1
print(dict(zip(le.classes_, le.transform(le.classes_))))
df["diagnosis"].value_counts()

# %% [markdown]
# ### Split features/target, then train-test split (80/20)

# %%
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# %% [markdown]
# ### Feature scaling (standardization)
#
# KNN is a distance-based algorithm, so features must be on the same scale
# or large-magnitude features (like `area_mean`) will dominate the distance
# calculation. The scaler is fit only on the training data and then applied
# to the test data to avoid data leakage.

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## Task 3: Model Development (3 Marks)

# %%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
print("Predictions on the first 10 test samples:", y_pred[:10])

# %% [markdown]
# ## Task 4: Model Evaluation (2 Marks)

# %%
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# %%
print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Malignant (1)"]))

# %% [markdown]
# ### Confusion Matrix

# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN (K=5)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# %% [markdown]
# ### Effect of K on accuracy (supporting analysis)

# %%
k_values = range(1, 21)
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    accuracies.append(acc)

plt.figure(figsize=(7, 4))
plt.plot(list(k_values), accuracies, marker="o")
plt.xlabel("K (n_neighbors)")
plt.ylabel("Test Accuracy")
plt.title("KNN Accuracy vs K")
plt.xticks(list(k_values))
plt.tight_layout()
plt.savefig("accuracy_vs_k.png", dpi=150)
plt.show()

best_k = list(k_values)[int(np.argmax(accuracies))]
print(f"Best K found in range 1-20: {best_k} (accuracy = {max(accuracies):.4f})")

# %% [markdown]
# ### Observations
#
# 1. The KNN model (K=5) achieves high accuracy on the held-out test set,
#    showing that the diagnostic measurements separate malignant and benign
#    tumors well in feature space once scaled.
# 2. Precision and recall for the malignant class are both important here —
#    recall matters most clinically, since a false negative (missing a
#    malignant tumor) is more costly than a false positive.
# 3. The K-vs-accuracy sweep shows accuracy is fairly stable across a range
#    of K values, but very small K (K=1) tends to overfit to noise while
#    very large K oversmooths the decision boundary — K=5 is a reasonable
#    middle ground.

# %% [markdown]
# ## Task 5: Conclusion (1 Mark)
#
# This project applied a K-Nearest Neighbors classifier to the Breast Cancer
# Wisconsin Diagnostic dataset to distinguish malignant from benign tumors
# using 30 numerical diagnostic features. After cleaning the data, encoding
# the diagnosis label, and standardizing the features, a KNN model with K=5
# achieved strong accuracy, precision, recall, and F1-scores on the test
# set, confirming that nucleus-shape and texture measurements are strong
# predictors of malignancy. Feature scaling proved essential: because KNN
# classifies a point based on the Euclidean distance to its neighbors,
# unscaled features such as area (which spans hundreds of units) would
# dominate the distance calculation and drown out smaller-scale features
# like smoothness or symmetry, distorting the neighborhood structure the
# algorithm relies on. A key limitation of KNN is that it is a lazy,
# instance-based learner: it stores the entire training set and computes
# distances to all training points at prediction time, making it
# computationally expensive and slow on large datasets, and it is also
# sensitive to irrelevant or correlated features and to the choice of K.
# Despite this, KNN remains an effective, easy-to-interpret baseline for
# this classification problem.
