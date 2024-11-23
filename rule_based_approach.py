# Import necessary libraries
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import local_binary_pattern

# Paths to the preprocessed data
images_path = "images.npy"
labels_path = "labels.npy"

# Load preprocessed images and labels
images = np.load(images_path)
labels = np.load(labels_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Function to compute edge count using Canny edge detection
def compute_edge_count(image):
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return np.sum(edges > 0)

# Function to compute the mean color intensity of an image
def compute_color_intensity(image):
    return np.mean(image)

# Function to estimate border thickness using contours
def compute_border_thickness(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(largest_contour)
        return min(w, h) * 0.03  # Approximation based on measurement
    return 0

# Function to compute the percentage of yellow in the border region
def compute_yellow_border_percentage(image):
    hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
    upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    h, w = mask.shape
    border_width = int(0.1 * w)  # 10% of width
    border_height = int(0.1 * h)  # 10% of height

    border_mask = np.zeros_like(mask)
    border_mask[:border_height, :] = 1  # Top border
    border_mask[-border_height:, :] = 1  # Bottom border
    border_mask[:, :border_width] = 1  # Left border
    border_mask[:, -border_width:] = 1  # Right border

    border_yellow_pixels = np.sum(mask * border_mask > 0)
    total_border_pixels = np.sum(border_mask > 0)

    return (border_yellow_pixels / total_border_pixels) * 100

# New Function: Compute texture features using Local Binary Patterns (LBP)
def compute_texture_pattern(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return lbp_hist.mean()  # Return mean of the histogram

# Rule-based classification function
def rule_based_classification(hist, edge_count, border_thickness, yellow_percentage, texture_pattern,
                              color_threshold, edge_threshold, border_threshold, yellow_threshold, texture_threshold,
                              weights):
    score = (
        weights['color_intensity'] * (1 if hist > color_threshold else 0) +
        weights['edge_count'] * (1 if edge_count > edge_threshold else 0) +
        weights['border_thickness'] * (1 if border_thickness < border_threshold else 0) +
        weights['yellow_border'] * (1 if yellow_percentage > yellow_threshold else 0) +
        weights['texture_pattern'] * (1 if texture_pattern > texture_threshold else 0)
    )
    return 1 if score >= weights['decision_threshold'] else 0

# Updated weights and thresholds
weights = {
    'color_intensity': 0.2,
    'edge_count': 0.1,
    'border_thickness': 0.2,
    'yellow_border': 0.3,
    'texture_pattern': 0.2,
    'decision_threshold': 0.4  # Hardcoded optimal threshold
}
color_threshold = 0.6
edge_threshold = 500
border_threshold = 3.0
yellow_threshold = 20.0
texture_threshold = 0.5

# Evaluate the model on the test set
predicted_labels = []
for image in X_test:
    color_intensity = compute_color_intensity(image)
    edge_count = compute_edge_count(image)
    border_thickness = compute_border_thickness(image)
    yellow_percentage = compute_yellow_border_percentage(image)
    texture_pattern = compute_texture_pattern(image)
    prediction = rule_based_classification(
        color_intensity,
        edge_count,
        border_thickness,
        yellow_percentage,
        texture_pattern,
        color_threshold,
        edge_threshold,
        border_threshold,
        yellow_threshold,
        texture_threshold,
        weights
    )
    predicted_labels.append(prediction)

# Metrics calculation
accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)
conf_matrix = confusion_matrix(y_test, predicted_labels)
report = classification_report(y_test, predicted_labels, target_names=["Non-Pokémon", "Pokémon"])

# Print evaluation metrics
print("Classification Report:")
print(report)

# Plot final confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Pokémon", "Pokémon"],
            yticklabels=["Non-Pokémon", "Pokémon"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Final Confusion Matrix")
plt.show()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, predicted_labels)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"Final ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Visualize False Positives and False Negatives
def visualize_misclassified_samples(title, indices, X_test, y_test, y_pred):
    """
    Displays the misclassified images (false positives or false negatives)
    with their predicted and true labels.
    """
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices[:10]):  # Display up to 10 samples for clarity
        plt.subplot(1, min(10, len(indices)), i + 1)
        plt.imshow(X_test[idx])
        plt.title(f"Pred: {'Pokémon' if y_pred[idx] == 1 else 'Non-Pokémon'}\n"
                  f"True: {'Pokémon' if y_test[idx] == 1 else 'Non-Pokémon'}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# Identify false positives and false negatives
false_positives = np.where((y_test == 0) & (np.array(predicted_labels) == 1))[0]
false_negatives = np.where((y_test == 1) & (np.array(predicted_labels) == 0))[0]

# Visualize False Positives
visualize_misclassified_samples(
    "False Positives (Non-Pokémon as Pokémon)",
    false_positives,
    X_test,
    y_test,
    predicted_labels
)

# Visualize False Negatives
visualize_misclassified_samples(
    "False Negatives (Pokémon as Non-Pokémon)",
    false_negatives,
    X_test,
    y_test,
    predicted_labels
)
