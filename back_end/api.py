from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fake_news_text_detector.detector as text_scanner
import deepfake_image_detector.detector as image_scanner
import requests
from PIL import Image
from io import BytesIO

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Load the model once when the API starts (text model)
model_dir = './fake_news_text_detector/fakenews_model'  # Ensure the model is in this directory
classifier = text_scanner.load_model()

def convert_to_jpg(file_path):
    """
    Convert the input image to JPEG format.
    Returns the new file path.
    """
    print(file_path)
    if len(file_path.rsplit(".")) < 2:
        file_path = file_path + ".webp"
    img = Image.open(file_path)
    
    # Ensure the conversion only happens for non-JPEG files
    if img.format != 'JPEG':
        rgb_img = img.convert('RGB')  # Convert to RGB if necessary (to avoid issues with transparency)
        jpg_path = file_path.rsplit('.', 1)[0] + '.jpg'  # Change the extension to .jpg
        rgb_img.save(jpg_path, 'JPEG')
        return jpg_path
    
    # If already a JPEG, return the original path
    return file_path

@app.route('/scan_text', methods=['POST'])
def scan_text():
    """
    API endpoint to predict fake news based on input text.
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input, 'text' field is required."}), 400

    texts = data['text']
    if isinstance(texts, str):
        texts = [texts]  # Convert a single string into a list

    # Run the fake news detection
    scores = text_scanner.predict_fakeness(classifier, texts)
    
    # Prepare the response
    response = []
    for score in scores:
        response.append(float(score))
    
    return jsonify(response), 200

@app.route('/scan_image', methods=['POST'])
def scan_image():
    """
    API endpoint to detect deepfake images.
    Accepts either an uploaded image file or an image URL in the 'url' field.
    """
    file = request.files.get('file')
    url = request.json.get('url') if request.is_json else None
    
    # Ensure either file or URL is provided
    if not file and not url:
        return jsonify({"error": "No file or URL provided."}), 400
    
    try:
        if file :#and allowed_file(file.filename):
            # Save uploaded file to upload folder
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
        
        elif url:
            # Download image from URL
            response = requests.get(url)
            if response.status_code != 200:
                return jsonify({"error": "Failed to download image from URL."}), 400

            # Save image to the upload folder
            image_name = url.split("/")[-1]
            file_path = os.path.join(UPLOAD_FOLDER, image_name)

            # Convert the downloaded image into a file format and save
            img = Image.open(BytesIO(response.content))
            img.save(file_path)

        else:
            return jsonify({"error": "Invalid file format or unsupported URL."}), 400
        
        file_path = convert_to_jpg(file_path)

        # Process the image using your image_scanner module
        result = image_scanner.predict_fakeness(file_path)

        # Delete the file after processing
        os.remove(file_path)
        print(f"Deleted file: {file_path}")

        return jsonify(result), 200
    
    except Exception as e:
        # Ensure file is deleted even if an error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension (e.g., png, jpg, jpeg, gif).
    """
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Starting point for the Flask app
if __name__ == '__main__':
    app.run(debug=True)
