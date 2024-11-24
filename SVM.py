import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Paths to the preprocessed data
print(f"[{datetime.now()}] Loading data...")
images_path = "images.npy"
labels_path = "labels.npy"

# Load preprocessed images and labels
images = np.load(images_path)
labels = np.load(labels_path)

print(f"[{datetime.now()}] Data loaded successfully. Splitting data into training and testing sets...")
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    images.reshape(images.shape[0], -1), labels, test_size=0.3, random_state=42, stratify=labels
)

# Scale features for SVM
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Apply PCA for dimensionality reduction
print(f"[{datetime.now()}] Applying PCA for dimensionality reduction...")
pca = PCA(n_components=50)  # Keep top 50 components
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

print(f"[{datetime.now()}] Data prepared with PCA. Training samples: {len(x_train_pca)}")

# Train classifier with GridSearchCV for optimal hyperparameters
print(f"[{datetime.now()}] Initializing SVM classifier and hyperparameter tuning...")
classifier = SVC(probability=True, class_weight='balanced')

parameters = [
    {'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100], 'kernel': ['rbf']},
    {'gamma': ['scale'], 'C': [0.1, 1, 10], 'kernel': ['linear']}
]

# Verbose set to 3 for detailed GridSearchCV logging
grid_search = GridSearchCV(classifier, parameters, cv=3, verbose=3, n_jobs=-1)

print(f"[{datetime.now()}] Training SVM classifier with GridSearchCV. This may take a while...")
grid_search.fit(x_train_pca, y_train)

print(f"[{datetime.now()}] Finished GridSearchCV fit!")

# Best classifier
best_estimator = grid_search.best_estimator_

print(f"[{datetime.now()}] Best parameters: {grid_search.best_params_}")
print(f"[{datetime.now()}] Evaluating the model on the test set...")

# Predict probabilities and find optimal threshold
y_proba = best_estimator.predict_proba(x_test_pca)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
optimal_idx = np.argmax(precision + recall)
optimal_threshold = thresholds[optimal_idx]
print(f"[{datetime.now()}] Optimal threshold based on Precision-Recall Curve: {optimal_threshold}")

# Apply optimal threshold
y_prediction = (y_proba >= optimal_threshold).astype(int)

# Metrics calculation
accuracy = accuracy_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction)
recall = recall_score(y_test, y_prediction)
f1 = f1_score(y_test, y_prediction)
conf_matrix = confusion_matrix(y_test, y_prediction)
report = classification_report(y_test, y_prediction, target_names=["Non-Pokémon", "Pokémon"])

# Print evaluation metrics
print(f"[{datetime.now()}] Classification Report:")
print(report)

# Identify false positives and false negatives
false_positives = np.where((y_test == 0) & (y_prediction == 1))[0]
false_negatives = np.where((y_test == 1) & (y_prediction == 0))[0]

print(f"[{datetime.now()}] Found {len(false_positives)} false positives and {len(false_negatives)} false negatives.")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Pokémon", "Pokémon"],
            yticklabels=["Non-Pokémon", "Pokémon"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# Visualize Misclassified Samples
def visualize_misclassified_samples(title, indices, x_test, y_test, y_pred):
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices[:10]):  # Show up to 10 samples
        plt.subplot(1, min(10, len(indices)), i + 1)
        plt.imshow(x_test[idx].reshape(128, 128, 3))  # Assuming original shape is 128x128x3
        plt.title(f"Pred: {'Pokémon' if y_pred[idx] == 1 else 'Non-Pokémon'}\n"
                  f"True: {'Pokémon' if y_test[idx] == 1 else 'Non-Pokémon'}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()


visualize_misclassified_samples(
    "False Positives (Non-Pokémon as Pokémon)",
    false_positives,
    images,
    y_test,
    y_prediction
)

visualize_misclassified_samples(
    "False Negatives (Pokémon as Non-Pokémon)",
    false_negatives,
    images,
    y_test,
    y_prediction
)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(best_estimator, x_train_pca, y_train, cv=5, n_jobs=-1)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Score", marker="o")
plt.plot(train_sizes, test_mean, label="Cross-Validation Score", marker="o")
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.show()

print(f"[{datetime.now()}] SVM classification completed successfully!")
