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
    auc,
    roc_curve
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

# Evaluate the model
print(f"[{datetime.now()}] Evaluating the model...")
test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save predictions and true labels
y_pred_proba = cnn_model.predict(x_test)  # Probabilities
y_pred = np.argmax(y_pred_proba, axis=1)  # Predicted labels
y_true = np.argmax(y_test, axis=1)  # True labels

# Fine-tune confidence threshold
thresholds = np.arange(0.5, 1.0, 0.05)
for threshold in thresholds:
    y_pred_custom = (y_pred_proba[:, 1] >= threshold).astype(int)
    print(f"Threshold: {threshold:.2f}")
    print(classification_report(y_true, y_pred_custom, target_names=["Non-Pokémon", "Pokémon"]))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Pokémon", "Pokémon"],
            yticklabels=["Non-Pokémon", "Pokémon"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"Precision-Recall Curve (AUC = {pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

# Custom Learning Curve Implementation
print(f"[{datetime.now()}] Plotting Learning Curve...")
train_sizes = np.linspace(0.1, 1.0, 5)
train_accuracies = []
val_accuracies = []

for train_size in train_sizes:
    print(f"Training on {int(train_size * len(x_train))} samples...")
    x_train_subset = x_train[:int(train_size * len(x_train))]
    y_train_subset = y_train[:int(train_size * len(y_train))]

    # Reinitialize model to prevent cumulative training effects
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

    history = cnn_model.fit(
        x_train_subset, y_train_subset,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=0
    )

    train_accuracies.append(history.history['accuracy'][-1])
    val_accuracies.append(history.history['val_accuracy'][-1])

# Plotting the Learning Curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(train_sizes, val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Training Set Size Fraction')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid()
plt.show()

print(f"[{datetime.now()}] Model evaluation completed.")
