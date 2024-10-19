import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse

def load_model(model_dir):
    """
    Loads the fine-tuned BERT model and tokenizer from the specified directory.

    Args:
        model_dir (str): Path to the directory containing the saved model and tokenizer.

    Returns:
        tokenizer (BertTokenizer): The loaded tokenizer.
        model (BertForSequenceClassification): The loaded BERT model.
    """
    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    
    print(f"Loading model from {model_dir}...")
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
    # Tokenize the input texts
    inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Move tensors to the specified device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        fake_probs = probabilities[:, 1].cpu().numpy()  # Probability of label '1' (fake)
    
    return fake_probs

import argparse

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Fake News Detection using Fine-Tuned BERT Model")
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./fakenews_model',
        help='Path to the directory containing the fine-tuned model and tokenizer.'
    )
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        help='Input text(s) to classify as fake or true.'
    )
    parser.add_argument(
        '--file_input',
        type=str,
        help='Path to the input text file with one news article per line.'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run the script in interactive mode to input texts one by one.'
    )
    
    args = parser.parse_args()
    
    # Load the model and tokenizer
    tokenizer, model, device = load_model(args.model_dir)
    
    if args.input:
        # Batch prediction
        texts = args.input
        scores = predict_fakeness(tokenizer, model, device, texts)
        for text, score in zip(texts, scores):
            label = 'Fake' if score >= 0.5 else 'True'
            print(f"Text: {text}\nFakeness Score: {score:.4f} ({label})\n")
    
    elif args.file_input:
        # File input mode
        try:
            with open(args.file_input, 'r', encoding='utf-8') as file:
                texts = file.readlines()
                texts = [text.strip() for text in texts if text.strip()]  # Remove empty lines
                scores = predict_fakeness(tokenizer, model, device, texts)
                overall = predict_fakeness(tokenizer, model, device, "\n".join(texts))
                for text, score in zip(texts, scores):
                    label = 'Fake' if score >= 0.5 else 'True'
                    print(f"Text: {text}\nFakeness Score: {score:.4f} ({label})\n")
                score = overall[0]
                label = 'Fake' if score >= 0.5 else 'True'
                print(f"\nOverall fakeness: {score:.4f} ({label})")
        except FileNotFoundError:
            print(f"Error: File '{args.file_input}' not found.")
    
    elif args.interactive:
        # Interactive mode
        print("Enter your news article text (type 'exit' to quit):")
        while True:
            user_input = input(">> ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting.")
                break
            if user_input.strip() == '':
                print("Empty input. Please enter some text or type 'exit' to quit.")
                continue
            score = predict_fakeness(tokenizer, model, device, [user_input])[0]
            label = 'Fake' if score >= 0.5 else 'True'
            print(f"Fakeness Score: {score:.4f} ({label})\n")
    
    else:
        print("No input provided. Use --input, --file_input, or --interactive for different modes.")
        print("Example usage:")
        print("  python use_trained_model.py --input \"Sample news text 1\" \"Sample news text 2\"")
        print("  python use_trained_model.py --file_input input_file.txt")
        print("  python use_trained_model.py --interactive")

if __name__ == "__main__":
    main()
