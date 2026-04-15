import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_image(path):
    # Read image
    img = cv2.imread(path)

    # Check if image loaded properly
    if img is None:
        print(f"Error loading image: {path}")
        return None

    # Resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize (0–255 → 0–1)
    img = img / 255.0

    return img