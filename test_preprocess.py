from src.preprocess import preprocess_image
import matplotlib.pyplot as plt

# CHANGE this path to any image from your dataset
image_path = "data/chest_xray/train/NORMAL/IM-0115-0001.jpeg"

img = preprocess_image(image_path)

if img is not None:
    print("Image shape:", img.shape)

    plt.imshow(img)
    plt.title("Processed Image")
    plt.axis('off')
    plt.show()
    