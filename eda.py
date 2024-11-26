import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Load the images and labels
images = np.load("images.npy")
labels = np.load("labels.npy")

# Verify dataset properties
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")
print(f"Image shapes (unique): {np.unique([img.shape for img in images])}")

# Display a sample image from each class
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Example Pokémon and Non-Pokémon images
pokemon_indices = np.where(labels == 1)[0][:3]
non_pokemon_indices = np.where(labels == 0)[0][:3]

for i, idx in enumerate(pokemon_indices):
    axs[0, i].imshow(images[idx])
    axs[0, i].set_title(f"Pokémon Sample {i+1}")
    axs[0, i].axis('off')

for i, idx in enumerate(non_pokemon_indices):
    axs[1, i].imshow(images[idx])
    axs[1, i].set_title(f"Non-Pokémon Sample {i+1}")
    axs[1, i].axis('off')

plt.suptitle("Sample Images from Each Class", fontsize=16)
plt.tight_layout()
plt.show()

# Mean Pixel Intensity Distribution
pokemon_mean = [np.mean(images[i]) for i in np.where(labels == 1)[0]]
non_pokemon_mean = [np.mean(images[i]) for i in np.where(labels == 0)[0]]

plt.figure(figsize=(10, 6))
plt.hist(pokemon_mean, bins=30, alpha=0.5, label="Pokémon", color="orange")
plt.hist(non_pokemon_mean, bins=30, alpha=0.5, label="Non-Pokémon", color="blue")
plt.title("Mean Pixel Intensity Distribution by Class")
plt.xlabel("Mean Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()

# Color Channel Intensity Distribution
colors = ('r', 'g', 'b')
plt.figure(figsize=(10, 6))
for i, color in enumerate(colors):
    pokemon_channel_values = images[labels == 1, :, :, i].ravel()
    non_pokemon_channel_values = images[labels == 0, :, :, i].ravel()
    plt.hist(pokemon_channel_values, bins=50, color=color, alpha=0.5, label=f"Pokémon {color.upper()}")
    plt.hist(non_pokemon_channel_values, bins=50, color=color, linestyle='dashed', alpha=0.5, label=f"Non-Pokémon {color.upper()}")

plt.title("Color Channel Intensity Distribution by Class")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()

# Edge Detection Histogram
def compute_edge_count(image):
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    return np.sum(edges > 0)

pokemon_edge_counts = [compute_edge_count(images[i]) for i in np.where(labels == 1)[0]]
non_pokemon_edge_counts = [compute_edge_count(images[i]) for i in np.where(labels == 0)[0]]

plt.figure(figsize=(10, 6))
plt.hist(pokemon_edge_counts, bins=30, alpha=0.5, label="Pokémon", color="orange")
plt.hist(non_pokemon_edge_counts, bins=30, alpha=0.5, label="Non-Pokémon", color="blue")
plt.title("Edge Counts by Class")
plt.xlabel("Edge Count")
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()

# Label Distribution
unique, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(8, 6))
sns.barplot(x=["Non-Pokémon", "Pokémon"], y=counts, palette=['blue', 'orange'])
plt.title("Label Distribution")
plt.ylabel("Count")
plt.grid(axis='y')
plt.show()


print("EDA Completed.")
