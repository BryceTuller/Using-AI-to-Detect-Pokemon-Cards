import os
import numpy as np
from skimage import io, transform, color
import time


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------TO BE USED FOR LINEAR REGRESSION PRE PROCESSING-------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------TO BE USED FOR RULE BASED PRE PROCESSING--------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def preprocess_images(folder_path, label, image_size=(128, 128), max_size=5 * 1024 * 1024,
                      allowed_extensions={'.jpg', '.jpeg', '.png'}):
    images = []
    labels = []

    # Process each image in the folder
    for i, filename in enumerate(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)

        # Skip if file is too large or has unsupported extension
        if not os.path.isfile(img_path) or os.path.getsize(img_path) > max_size:
            continue
        if os.path.splitext(filename)[1].lower() not in allowed_extensions:
            continue

        # Log processing time for each image
        start_time = time.time()
        try:
            img = io.imread(img_path)

            # Convert grayscale or RGBA to RGB
            if len(img.shape) == 2:  # Grayscale image
                img = color.gray2rgb(img)
            elif img.shape[2] == 4:  # RGBA image
                img = color.rgba2rgb(img)

            # Resize image
            img_resized = transform.resize(img, image_size, anti_aliasing=True)
            images.append(img_resized)
            labels.append(label)

            # Log success and time taken
            print(f"Processed image {i + 1} in {time.time() - start_time:.2f} seconds")

        except Exception as e:
            print(f"Error processing image {filename}: {e}")

        # Log progress every 100 images
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} images so far...")

    # Return the numpy arrays of images and labels
    return np.array(images), np.array(labels)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------GENERATE ONE TIME PRE PROCESSING LISTS----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == 'main':
    # Paths for Pokémon and non-Pokémon images
    pokemon_folder = 'data/PokemonCards'
    non_pokemon_folder = 'data/NOTPokemonCards'

    # Preprocess Pokémon images
    print("Processing Pokémon images...")
    pokemon_images, pokemon_labels = preprocess_images(pokemon_folder, label=1)  # Label 1 for Pokémon cards

    # Preprocess non-Pokémon images
    print("Processing non-Pokémon images...")
    # Label 0 for non-Pokémon cards
    non_pokemon_images, non_pokemon_labels = preprocess_images(non_pokemon_folder, label=0)

    # Combine and save processed data
    print("Combining and saving data...")
    images = np.concatenate((pokemon_images, non_pokemon_images))
    labels = np.concatenate((pokemon_labels, non_pokemon_labels))
    np.save('images.npy', images)
    np.save('labels.npy', labels)
    print("Data saved successfully.")
