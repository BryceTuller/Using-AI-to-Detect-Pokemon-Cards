# Import necessary libraries
import numpy as np
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, auc
)
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

# Rule 1: Compute edge count using Canny edge detection
def compute_edge_count(image):
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return np.sum(edges > 0)

# Rule 2: Compute mean color intensity of an image
def compute_color_intensity(image):
    return np.mean(image)

# Rule 3: Estimate border thickness using contours
def compute_border_thickness(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(largest_contour)
        return min(w, h) * 0.03
    return 0

# Rule 4: Compute the percentage of yellow in the border region
def compute_yellow_border_percentage(image):
    hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    h, w = mask.shape
    border_width = int(0.1 * w)
    border_height = int(0.1 * h)

    border_mask = np.zeros_like(mask)
    border_mask[:border_height, :] = 1
    border_mask[-border_height:, :] = 1
    border_mask[:, :border_width] = 1
    border_mask[:, -border_width:] = 1

    border_yellow_pixels = np.sum(mask * border_mask > 0)
    total_border_pixels = np.sum(border_mask > 0)

    return (border_yellow_pixels / total_border_pixels) * 100

# Rule 5: Compute texture features using Local Binary Patterns (LBP)
def compute_texture_pattern(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return lbp_hist.mean()

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
    'decision_threshold': 0.4
}
color_threshold = 0.6
edge_threshold = 500
border_threshold = 3.0
yellow_threshold = 20.0
texture_threshold = 0.5

# Function to evaluate the model on a dataset
def evaluate_model(X, y, dataset_name):
    predicted_labels = []
    probabilities = []
    for image in X:
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
        probabilities.append(color_intensity * weights['color_intensity'])

    # Metrics calculation
    accuracy = accuracy_score(y, predicted_labels)
    precision = precision_score(y, predicted_labels)
    recall = recall_score(y, predicted_labels)
    f1 = f1_score(y, predicted_labels)
    conf_matrix = confusion_matrix(y, predicted_labels)
    report = classification_report(y, predicted_labels, target_names=["Non-Pokémon", "Pokémon"], zero_division=0)

    print(f"\nEvaluation on {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Pokémon", "Pokémon"],
                yticklabels=["Non-Pokémon", "Pokémon"])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix ({dataset_name})")
    plt.show()

    # ROC and AUC
    fpr, tpr, _ = roc_curve(y, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend()
    plt.grid()
    plt.show()

    return predicted_labels, conf_matrix

# Evaluate on training and testing sets
train_predicted, train_conf_matrix = evaluate_model(X_train, y_train, "Training Set")
test_predicted, test_conf_matrix = evaluate_model(X_test, y_test, "Testing Set")

# Visualize false positives and false negatives
def visualize_misclassified_samples(title, indices, X_data, y_data, y_pred):
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices[:10]):
        plt.subplot(1, min(10, len(indices)), i + 1)
        plt.imshow(X_data[idx])
        plt.title(f"Pred: {'Pokémon' if y_pred[idx] == 1 else 'Non-Pokémon'}\n"
                  f"True: {'Pokémon' if y_data[idx] == 1 else 'Non-Pokémon'}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# Identify and visualize false positives and negatives
train_false_positives = np.where((y_train == 0) & (np.array(train_predicted) == 1))[0]
train_false_negatives = np.where((y_train == 1) & (np.array(train_predicted) == 0))[0]

test_false_positives = np.where((y_test == 0) & (np.array(test_predicted) == 1))[0]
test_false_negatives = np.where((y_test == 1) & (np.array(test_predicted) == 0))[0]

visualize_misclassified_samples(
    "False Positives (Training Set)", train_false_positives, X_train, y_train, train_predicted)
visualize_misclassified_samples(
    "False Negatives (Training Set)", train_false_negatives, X_train, y_train, train_predicted)
visualize_misclassified_samples(
    "False Positives (Testing Set)", test_false_positives, X_test, y_test, test_predicted)
visualize_misclassified_samples(
    "False Negatives (Testing Set)", test_false_negatives, X_test, y_test, test_predicted)
