import os
import numpy as np
from preprocess import preprocess_image
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_data(folder):
    X, y = [], []

    for label in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(folder, label)
        class_num = 0 if label == 'NORMAL' else 1

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            img = preprocess_image(img_path)
            if img is not None:
                X.append(img)
                y.append(class_num)

    return np.array(X), np.array(y)

# Load test data
print("Loading test data...")
X_test, y_test = load_test_data("data/chest_xray/test")

# Load trained model
model = load_model("models/model.h5")

# Predictions
preds = model.predict(X_test)
y_pred = (preds > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()
