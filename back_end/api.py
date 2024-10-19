from flask import Flask, request, jsonify
from fake_news_text_detector.detector import load_model, predict_fakeness, get_fake_news_label

app = Flask(__name__)

# Load the model once when the API starts
model_dir = './fake_news_text_detector/fakenews_model'  # Make sure the model is in this directory
tokenizer, model, device = load_model(model_dir)

@app.route('/scantext', methods=['POST'])
def predict():
    """
    API endpoint to predict fake news based on input text.
    Accepts a JSON payload with a 'text' field containing a string or list of texts.

    Example request body:
    {
        "text": "This is an example news article."
    }

    Returns a JSON response with the fakeness score and label.
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input, 'text' field is required."}), 400

    texts = data['text']
    if isinstance(texts, str):
        texts = [texts]  # Convert a single string into a list

    # Run the fake news detection
    scores = predict_fakeness(tokenizer, model, device, texts)
    
    # Prepare the response
    response = []
    for score in scores:
        response.append(float(score))
    
    return jsonify(response), 200

# Starting point for the Flask app
if __name__ == '__main__':
    app.run(debug=True)
