from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import os
from src.preprocess import preprocess_image

app = Flask(__name__)

# Load model
model = load_model("models/model.h5")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]

        if pred > 0.5:
            result = "PNEUMONIA"
        else:
            result = "NORMAL"

        return render_template("index.html", prediction=result)

    return "Error"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)