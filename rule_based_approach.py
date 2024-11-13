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


def compute_edge_count(image):
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    edge_count = np.sum(edges > 0)
    return edge_count


def compute_color_intensity(image):
    return np.mean(image)


def rule_based_classification(hist, edge_count, color_threshold=0.5, edge_threshold=1000):
    is_pokemon_card = (hist > color_threshold) and (edge_count > edge_threshold)
    return 1 if is_pokemon_card else 0


# Define threshold ranges for tuning
color_thresholds = [0.4, 0.5, 0.6]
edge_thresholds = [800, 1000, 1200]

# Store best results
best_accuracy = 0
best_params = (0, 0)
best_predicted_labels = []

# Tune thresholds by trying different combinations
for color_thresh in color_thresholds:
    for edge_thresh in edge_thresholds:
        predicted_labels = []
        for image in images:
            color_intensity = compute_color_intensity(image)
            edge_count = compute_edge_count(image)
            prediction = rule_based_classification(color_intensity, edge_count, color_threshold=color_thresh,
                                                   edge_threshold=edge_thresh)
            predicted_labels.append(prediction)

        # Convert predicted_labels to numpy array
        predicted_labels = np.array(predicted_labels)

        # Calculate evaluation metrics
        accuracy = accuracy_score(labels, predicted_labels)
        precision = precision_score(labels, predicted_labels)
        recall = recall_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels)

        print(f"Color Threshold: {color_thresh}, Edge Threshold: {edge_thresh}")
        print(
            f"Accuracy: {accuracy * 100:.2f}% | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f}\n")

        # Track the best result based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (color_thresh, edge_thresh)
            best_predicted_labels = predicted_labels

# Display the best results
print("Best Rule-Based Approach - Evaluation Metrics:")
print(f"Best Color Threshold: {best_params[0]}, Best Edge Threshold: {best_params[1]}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")

# Plot the confusion matrix for the best thresholds
conf_matrix = confusion_matrix(labels, best_predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Pokémon", "Pokémon"],
            yticklabels=["Non-Pokémon", "Pokémon"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Best Rule-Based Approach")
plt.show()
