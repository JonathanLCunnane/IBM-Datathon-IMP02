from transformers import pipeline
import imageio

# Load the video classification pipeline
classifier = pipeline("video-classification", model="muneeb1812/videomae-base-fake-video-classification")

def predict_fakeness(path):
    # Open the video using imageio
    video = imageio.get_reader(path, 'ffmpeg')

    # Read frames from the video
    frames = []
    for frame in video:
        frames.append(frame)

    # Classify the video
    result = classifier(frames)

    # Output the classification result
    return result[0]['score']
