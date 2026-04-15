from src.predict import predict_image

# Change this path to test image
image_path = "data/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

result = predict_image(image_path)

print("Final Prediction:", result)