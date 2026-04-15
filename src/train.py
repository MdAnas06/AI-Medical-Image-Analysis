import os
import numpy as np
from preprocess import preprocess_image
from model import build_model
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def load_data(folder):
    X, y = [], []

    limit = 1500  # balance both classes

    for label in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(folder, label)
        class_num = 0 if label == 'NORMAL' else 1

        count = 0

        for img_name in os.listdir(path):
            if count >= limit:
                break

            img_path = os.path.join(path, img_name)

            img = preprocess_image(img_path)

            if img is not None:
                X.append(img)
                y.append(class_num)
                count += 1

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # 🔥 VERY IMPORTANT: Shuffle data
    X, y = shuffle(X, y, random_state=42)

    return X, y


# ===========================
# MAIN TRAINING PIPELINE
# ===========================

print("Loading balanced data...")
X, y = load_data("data/chest_xray/train")

print("Data loaded!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Build model
model = build_model()

# Train model
print("Starting training...")

history = model.fit(
    X, y,
    epochs=10,              # increased epochs
    batch_size=32,
    validation_split=0.2
)

# Save model
model.save("models/model.h5")
print("Model saved!")

# ===========================
# PLOT ACCURACY GRAPH
# ===========================

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()

plt.savefig("outputs/accuracy.png")
plt.show()
