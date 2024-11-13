import os
import numpy
from skimage import io, transform


# Simple function to preprocess a class of images to the same size
def preprocess_images(folder_path, label, image_size=(128, 128)):
    images = []
    labels = []

    # For every file within the target directory, resize to set identical size
    for filename in os.listdir(folder_path):
        img = io.imread(os.path.join(folder_path, filename))
        img_resized = transform.resize(img, image_size)
        images.append(img_resized)
        labels.append(label)

    # Return an array of resized images and an array of their classification labels
    return numpy.array(images), numpy.array(labels)
