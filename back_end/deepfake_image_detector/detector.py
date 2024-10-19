from transformers import pipeline
from PIL import Image

# Load the image classification pipeline with the model "Organika/sdxl-detector"
classifier = pipeline("image-classification", model="dima806/ai_vs_real_image_detection")

def predict_fakeness(path):
    # Open an image (replace 'your_image.jpg' with the path to your image)
    image = Image.open(path)

    # Classify the image
    result = classifier(image)

    # Output the classification result
    return(result["FAKE"])
