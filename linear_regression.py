from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from skimage import io, transform, color
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------PRE PROCESS IMAGES-------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def linear_pre_process(folder_path, label, image_size=(128, 128)):
    images = []
    labels = []
    for i, filename in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = io.imread(img_path)

        img_resized = transform.resize(img, image_size, anti_aliasing=True, mode='reflect')

        if img_resized.shape[-1] == 4:
            img_resized = img_resized[..., :3]

        if img_resized.ndim == 2:
            img_resized = np.stack([img_resized] * 3, axis=-1)

        img_flattened = img_resized.flatten()
        images.append(img_flattened)
        labels.append(label)

        print(f"Image Processed = {i + 1}")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


x_is_pokemon, y_is_pokemon = linear_pre_process("data/PokemonCards", label=1)
x_isnt_pokemon, y_isnt_pokemon = linear_pre_process("data/NOTPokemonCards", label=0)

x = np.vstack((x_is_pokemon, x_isnt_pokemon))
y = np.concatenate((y_is_pokemon, y_isnt_pokemon))

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------TRAIN AND TEST MODEL-----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
# Training set predictions
y_prediction_train_continuous = model.predict(x_train)
y_prediction_train_continuous = (y_prediction_train_continuous >= 0.8).astype(int)

# Test set predictions
y_prediction_continuous = model.predict(x_test)
y_prediction_continuous = (y_prediction_continuous >= 0.8).astype(int)

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------PRINT REPORT AND PLOTS-----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
print("Classification Report (training set) : ")
print(classification_report(y_train, y_prediction_train_continuous))

print("Classification Report (test set) : ")
print(classification_report(y_test, y_prediction_continuous))

# Confusion Matrix
conf_matrix_test = confusion_matrix(y_test, y_prediction_continuous)
plt.figure(figsize=(6, 6))
sns.heatmap(
    conf_matrix_test,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Non-Pokemon", "Pokemon"],
    yticklabels=["Non-Pokemon", "Pokemon"],
)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

# ROC Curves
fpr_test, tpr_test, _ = roc_curve(y_test, model.predict(x_test))
roc_auc_test = auc(fpr_test, tpr_test)

fpr_train, tpr_train, _ = roc_curve(y_train, model.predict(x_train))
roc_auc_train = auc(fpr_train, tpr_train)

plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, label=f"Test ROC Curve (AUC = {roc_auc_test:.2f})", color="blue")
plt.plot(fpr_train, tpr_train, label=f"Train ROC Curve (AUC = {roc_auc_train:.2f})", color="green")
plt.plot([0, 1], [0, 1], "r--", label="Random Classifier")
plt.title("Receiver Operating Characteristic (ROC)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Distribution Histogram
plt.figure(figsize=(8, 6))
plt.hist(y_prediction_continuous[y_test == 0], bins=20, alpha=0.5, label="Actual Non-Pokemon")
plt.hist(y_prediction_continuous[y_test == 1], bins=20, alpha=0.5, label="Actual Pokemon")
plt.axvline(0.8, color='r', linestyle='--', label="Decision Boundary (0.8)")
plt.xlabel("Predicted Value")
plt.ylabel("Frequency")
plt.title("Distribution of Predicted Values for Test Set")
plt.legend()
plt.show()
