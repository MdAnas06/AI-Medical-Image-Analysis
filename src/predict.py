from tensorflow.keras.models import load_model
from src.preprocess import preprocess_image
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = load_model("models/model.h5")

def predict_image(path):
    # Preprocess image
    img = preprocess_image(path)

    if img is None:
        return "Error loading image"

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0][0]
    print("Raw Prediction Value :", pred)

    # Convert to label
    if pred > 0.7:
        label = "PNEUMONIA"
    else:
        label = "NORMAL"

    # Show image with prediction
    plt.imshow(img[0])
    plt.title(f"Prediction: {label}")
    plt.axis('off')
    plt.show()

    return label