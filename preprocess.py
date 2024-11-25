import os
import numpy as np
from skimage import io, transform, color
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def preprocess_images(folder_path, label, image_size=(128, 128), max_size=5 * 1024 * 1024,
                      allowed_extensions={'.jpg', '.jpeg', '.png'}):
    images = []
    labels = []
    skipped_files = 0  # Track skipped files for debugging

    for i, filename in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)

        if not os.path.isfile(img_path) or os.path.getsize(img_path) > max_size:
            skipped_files += 1
            print(f"Skipped file {filename} due to size.")
            continue
        if os.path.splitext(filename)[1].lower() not in allowed_extensions:
            skipped_files += 1
            print(f"Skipped file {filename} due to unsupported extension.")
            continue

        try:
            img = io.imread(img_path)

            if len(img.shape) == 2:
                img = color.gray2rgb(img)
            elif img.shape[2] == 4:
                img = color.rgba2rgb(img)

            img_resized = transform.resize(img, image_size, anti_aliasing=True)
            images.append(img_resized)
            labels.append(label)

        except Exception as e:
            print(f"Error processing image {filename}: {e}")

    print(f"Skipped {skipped_files} files in folder {folder_path}.")
    return np.array(images), np.array(labels)


# Paths for Pokémon and non-Pokémon images
pokemon_folder = 'data/PokemonCards'
non_pokemon_folder = 'data/NOTPokemonCards'

print("Processing Pokémon images...")
pokemon_images, pokemon_labels = preprocess_images(pokemon_folder, label=1)

print("Processing non-Pokémon images...")
non_pokemon_images, non_pokemon_labels = preprocess_images(non_pokemon_folder, label=0)

print("Combining and saving data...")
images = np.concatenate((pokemon_images, non_pokemon_images))
labels = np.concatenate((pokemon_labels, non_pokemon_labels))

# Shuffle the dataset
images, labels = shuffle(images, labels, random_state=42)

np.save('images.npy', images)
np.save('labels.npy', labels)
print("Data saved successfully.")

# Inspect a few samples
for i in range(5):
    plt.imshow(images[i])
    plt.title(f"Label: {'Pokémon' if labels[i] == 1 else 'Non-Pokémon'}")
    plt.axis('off')
    plt.show()
