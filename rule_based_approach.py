import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Paths to the preprocessed data
images_path = "images.npy"
labels_path = "labels.npy"

# Load preprocessed images and labels
images = np.load(images_path)
labels = np.load(labels_path)

# Functions for computing features
def compute_edge_count(image):
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_count = np.sum(edges > 0)
    return edge_count

def compute_color_intensity(image):
    return np.mean(image)

def rule_based_classification(hist, edge_count, color_threshold, edge_threshold):
    is_pokemon_card = (hist > color_threshold) and (edge_count > edge_threshold)
    return 1 if is_pokemon_card else 0

# Best thresholds so far
best_color_threshold = 0.65
best_edge_threshold = 600
best_accuracy = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_conf_matrix = None

# Testing a range of values around the current best thresholds
color_thresholds = np.arange(0.62, 0.5, -0.01)
edge_thresholds = range(575, 1000, 50)

for color_threshold in color_thresholds:
    for edge_threshold in edge_thresholds:
        predicted_labels = []
        for image in images:
            color_intensity = compute_color_intensity(image)
            edge_count = compute_edge_count(image)
            prediction = rule_based_classification(color_intensity, edge_count, color_threshold, edge_threshold)
            predicted_labels.append(prediction)

        # Evaluate metrics
        accuracy = accuracy_score(labels, predicted_labels)
        precision = precision_score(labels, predicted_labels)
        recall = recall_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels)
        conf_matrix = confusion_matrix(labels, predicted_labels)

        print(f"Color Threshold: {color_threshold}, Edge Threshold: {edge_threshold}")
        print(f"Accuracy: {accuracy * 100:.2f}% | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f}")

        # Update best scores if precision is high and F1 score is better
        if precision >= best_precision and f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_accuracy = accuracy
            best_color_threshold = color_threshold
            best_edge_threshold = edge_threshold
            best_conf_matrix = conf_matrix

# Display best results
print("\nBest Rule-Based Approach - Evaluation Metrics:")
print(f"Best Color Threshold: {best_color_threshold}, Best Edge Threshold: {best_edge_threshold}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
print(f"Best Precision: {best_precision:.2f}")
print(f"Best Recall: {best_recall:.2f}")
print(f"Best F1 Score: {best_f1:.2f}")

# Plot the best confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(best_conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Pokémon", "Pokémon"],
            yticklabels=["Non-Pokémon", "Pokémon"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Best Rule-Based Approach")
plt.show()
