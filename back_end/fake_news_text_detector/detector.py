import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model(model_dir):
    """
    Loads the fine-tuned BERT model and tokenizer from the specified directory.
    
    Args:
        model_dir (str): Path to the directory containing the saved model and tokenizer.
    
    Returns:
        tokenizer (BertTokenizer): The loaded tokenizer.
        model (BertForSequenceClassification): The loaded BERT model.
        device (torch.device): The device (GPU/CPU) the model will run on.
    """
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    
    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

def predict_fakeness(tokenizer, model, device, texts):
    """
    Predicts the fakeness score for a list of texts.
    
    Args:
        tokenizer (BertTokenizer): The tokenizer.
        model (BertForSequenceClassification): The BERT model.
        device (torch.device): The device to run the model on.
        texts (list): List of text strings to classify.
    
    Returns:
        list: Fakeness scores between 0 and 1.
    """
    inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        fake_probs = probabilities[:, 1].cpu().numpy()  # Probability of label '1' (fake)
    
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
