from transformers import pipeline
from PIL import Image

# Load the image classification pipeline
classifier = pipeline("image-classification", model="dima806/ai_vs_real_image_detection")

def predict_fakeness(path):
    # Open an image
    image = Image.open(path)

    # Classify the image
    result = classifier(image)

    # Output the classification result
    return(result[0]['score'])
