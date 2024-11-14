import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Load the images and labels (update paths if needed)
images = np.load("images.npy")
labels = np.load("labels.npy")

# Display a sample image from each class (Pokémon and Non-Pokémon)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Example Pokémon and Non-Pokémon images
pokemon_index = np.where(labels == 1)[0][0]
non_pokemon_index = np.where(labels == 0)[0][0]

# Sample Images
axs[0, 0].imshow(images[pokemon_index])
axs[0, 0].set_title("Sample Pokémon Image")
axs[0, 1].imshow(images[non_pokemon_index])
axs[0, 1].set_title("Sample Non-Pokémon Image")

# Mean Pixel Intensity Distribution
mean_intensities = [np.mean(image) for image in images]
axs[0, 2].hist(mean_intensities, bins=30, color='blue')
axs[0, 2].set_title("Mean Pixel Intensity Distribution")
axs[0, 2].set_xlabel("Mean Intensity")
axs[0, 2].set_ylabel("Frequency")

# Color Channel Intensity Distribution
colors = ('r', 'g', 'b')
for i, color in enumerate(colors):
    channel_values = images[:, :, :, i].ravel()  # Flatten each color channel
    axs[1, 0].hist(channel_values, bins=50, color=color, alpha=0.5, label=color)
axs[1, 0].set_title("Color Channel Intensity Distribution")
axs[1, 0].legend()

# Edge Detection Sample
sample_image = images[pokemon_index]
gray_image = cv2.cvtColor((sample_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray_image, 100, 200)
axs[1, 1].imshow(edges, cmap='gray')
axs[1, 1].set_title("Edge Detection Sample")

# Label Distribution
unique, counts = np.unique(labels, return_counts=True)
axs[1, 2].bar(["Non-Pokémon", "Pokémon"], counts, color=['blue', 'orange'])
axs[1, 2].set_title("Label Distribution")
axs[1, 2].set_ylabel("Count")

plt.tight_layout()
plt.show()
