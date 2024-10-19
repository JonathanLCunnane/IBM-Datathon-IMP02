from transformers import pipeline

def load_model():
    """
    Loads the fake news classification pipeline from the specified directory.

    Args:
        model_dir (str): Path to the directory containing the saved model and tokenizer.

    Returns:
        classifier: The Hugging Face pipeline for fake news classification.
    """
    # Load the pipeline with the model and tokenizer
    classifier = pipeline('text-classification', model="JTSR/BERT_Fake_news_classifier")
    return classifier

def predict_fakeness(classifier, texts):
    """
    Predicts the fakeness score for a list of texts.

    Args:
        classifier: The Hugging Face pipeline for text classification.
        texts (list): List of text strings to classify.

    Returns:
        list: Fakeness scores between 0 and 1.
    """
    # Get predictions from the classifier
    predictions = classifier(texts)
    
    # Extract the probabilities for the 'Fake' label
    fake_probs = [pred['score'] for pred in predictions]
    
    return fake_probs

# Additional helper function to determine the label
def get_fake_news_label(score):
    """
    Converts a fakeness score to a label (Fake or True).

    Args:
        score (float): Fakeness score.

    Returns:
        str: 'Fake' if the score is >= 0.5, otherwise 'True'.
    """
    return 'Fake' if score >= 0.5 else 'True'