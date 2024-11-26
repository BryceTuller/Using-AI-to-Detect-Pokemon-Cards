import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split
from datetime import datetime
import seaborn as sns

# Paths to the preprocessed data
print(f"[{datetime.now()}] Loading data...")
images_path = "images.npy"
labels_path = "labels.npy"

# Load preprocessed images and labels
images = np.load(images_path)
labels = np.load(labels_path)

# Preprocess labels (categorical encoding)
labels = to_categorical(labels, num_classes=2)

print(f"[{datetime.now()}] Data loaded successfully. Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.3, random_state=42, stratify=labels
)

print(f"[{datetime.now()}] Data split into {len(x_train)} training samples and {len(x_test)} testing samples.")

# Define CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss')

# Train the model
print(f"[{datetime.now()}] Starting model training...")
history = cnn_model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Evaluate the model on training and testing sets
print(f"[{datetime.now()}] Evaluating the model...")
train_loss, train_accuracy = cnn_model.evaluate(x_train, y_train, verbose=0)
test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test, verbose=0)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%, Testing Accuracy: {test_accuracy * 100:.2f}%")

# Generate predictions for both sets
y_train_pred_proba = cnn_model.predict(x_train)
y_test_pred_proba = cnn_model.predict(x_test)
y_train_pred = np.argmax(y_train_pred_proba, axis=1)
y_test_pred = np.argmax(y_test_pred_proba, axis=1)
y_train_true = np.argmax(y_train, axis=1)
y_test_true = np.argmax(y_test, axis=1)

# Function to evaluate model performance
def evaluate_model_performance(y_true, y_pred, set_name):
    print(f"\nEvaluation on {set_name} Set:")
    print(classification_report(y_true, y_pred, target_names=["Non-Pokémon", "Pokémon"]))
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    print(f"False Positives: {false_positives}, False Negatives: {false_negatives}")
    return conf_matrix

# Evaluate on training set
conf_matrix_train = evaluate_model_performance(y_train_true, y_train_pred, "Training")

# Evaluate on testing set
conf_matrix_test = evaluate_model_performance(y_test_true, y_test_pred, "Testing")

# Plot confusion matrix for testing set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Pokémon", "Pokémon"],
            yticklabels=["Non-Pokémon", "Pokémon"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix (Testing Set)")
plt.show()

# Plot ROC Curve for testing set
fpr, tpr, _ = roc_curve(y_test_true, y_test_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve (Testing Set)")
plt.legend()
plt.grid()
plt.show()

# Plot Precision-Recall Curve for testing set
precision, recall, _ = precision_recall_curve(y_test_true, y_test_pred_proba[:, 1])
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"Precision-Recall Curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Testing Set)")
plt.legend()
plt.grid()
plt.show()

print(f"[{datetime.now()}] Model evaluation completed.")
