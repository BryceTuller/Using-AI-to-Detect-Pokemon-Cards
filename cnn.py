import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from datetime import datetime
import cv2

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

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.title('Training and Validation Accuracy')
plt.show()

# Real-time camera functionality
print(f"[{datetime.now()}] Loading trained model for real-time detection...")
model = load_model("best_model.keras")  # Ensure the model is saved earlier during training

def preprocess_frame(frame):
    """
    Preprocess the frame for prediction.
    Resizes the frame and normalizes pixel values.
    """
    frame_resized = cv2.resize(frame, (128, 128))
    frame_array = frame_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(frame_array, axis=0)  # Add batch dimension

# Open the camera
print(f"[{datetime.now()}] Starting camera feed...")
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("[Error] Camera not accessible!")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("[Error] Failed to grab frame!")
        break

    # Preprocess the frame for prediction
    preprocessed_frame = preprocess_frame(frame)

    # Predict with the model
    prediction = model.predict(preprocessed_frame)[0]
    label = "Pokémon" if prediction[1] > prediction[0] else "Non-Pokémon"
    confidence = max(prediction) * 100

    # Add label to the frame
    cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the label
    cv2.imshow("Real-Time Pokémon Card Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
