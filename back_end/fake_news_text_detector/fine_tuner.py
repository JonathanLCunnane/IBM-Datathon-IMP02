import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import logging
import sys

def main():
    # -------------------------------
    # 1. Configuration and Parameters
    # -------------------------------
    
    # File path to CSV dataset
    DATA_PATH = 'llm_fine_tune\WELFake_Dataset.csv'
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]

    # Model configuration
    MODEL_NAME = 'bert-base-uncased'
    NUM_LABELS = 2  # 0: True, 1: Fake

    # Training configuration
    OUTPUT_DIR = './fakenews_model'
    BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 16
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_SEQ_LENGTH = 512
    RANDOM_SEED = 42
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1
    SAMPLE_FRACTION = 1

    # -------------------------------
    # 2. Setup Logging
    # -------------------------------
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # -------------------------------
    # 3. Check for GPU Availability
    # -------------------------------
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA is not available. Using CPU for training.")
    
    # -------------------------------
    # 4. Load and Preprocess Dataset
    # -------------------------------

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Drop rows with missing title, text, or label
    df = df.dropna(subset=['title', 'text', 'label']).reset_index(drop=True)

    # Select a sample of the dataset
    logger.info(f"Selecting a sample of {SAMPLE_FRACTION*100}% of the dataset...")
    df_sampled = df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_SEED)

    # Concatenate title and text into a single column
    df_sampled['content'] = df_sampled['title'] + ' ' + df_sampled['text']

    # Keep only necessary columns
    df_sampled = df_sampled[['content', 'label']]

    logger.info(f"Total samples after sampling: {len(df_sampled)}")

    # -------------------------------
    # 5. Split the Dataset
    # -------------------------------

    logger.info("Splitting dataset into train, validation, and test sets...")

    # First split: Train + Validation vs. Test
    train_val_df, test_df = train_test_split(
        df_sampled,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=df_sampled['label']
    )

    # Calculate validation size as a proportion of the train_val set
    val_size = VALIDATION_SPLIT / (1 - TEST_SPLIT)

    # Second split: Train vs. Validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=RANDOM_SEED,
        stratify=train_val_df['label']
    )

    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    logger.info(f"Test set size: {len(test_df)}")

    # -------------------------------
    # 6. Convert to Hugging Face Datasets
    # -------------------------------

    logger.info("Converting DataFrames to Hugging Face Datasets...")
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # -------------------------------
    # 7. Tokenization
    # -------------------------------

    logger.info("Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples['content'],
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )

    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Remove the 'content' column as it's no longer needed
    train_dataset = train_dataset.remove_columns(['content', '__index_level_0__'])
    val_dataset = val_dataset.remove_columns(['content', '__index_level_0__'])
    test_dataset = test_dataset.remove_columns(['content', '__index_level_0__'])

    # Rename 'label' to 'labels' as expected by the model
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    # Set the format to PyTorch tensors
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # -------------------------------
    # 8. Define Metrics
    # -------------------------------

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', pos_label=1
        )
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    # -------------------------------
    # 9. Initialize the Model
    # -------------------------------

    logger.info("Loading pre-trained BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    # Move the model to the appropriate device
    model.to(device)

    # -------------------------------
    # 10. Set Up Training Arguments
    # -------------------------------

    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        seed=RANDOM_SEED,
        fp16=torch.cuda.is_available()  # Enable mixed precision if using GPU
    )

    # -------------------------------
    # 11. Initialize Trainer
    # -------------------------------

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # -------------------------------
    # 12. Train the Model
    # -------------------------------

    logger.info("Starting training...")
    trainer.train()

    # -------------------------------
    # 13. Evaluate the Model
    # -------------------------------

    logger.info("Evaluating the model on the test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info("Test Set Results:")
    for key, value in test_results.items():
        if key.startswith('eval_'):
            logger.info(f"{key}: {value:.4f}")

    # -------------------------------
    # 14. Assign Fakeness Scores
    # -------------------------------

    logger.info("Assigning fakeness scores to the test set...")
    predictions = trainer.predict(test_dataset)
    probabilities = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()

    # Normalize fakeness scores between 0 and 1 (already probabilities)
    fakeness_scores = probabilities

    # Append fakeness scores to the test dataframe
    test_df['fakeness_score'] = fakeness_scores

    # Save the test results with fakeness scores to a CSV file
    logger.info("Saving test results with fakeness scores...")
    test_df.to_csv('test_results_with_fakeness_scores.csv', index=False)

    # -------------------------------
    # 15. Save the Fine-Tuned Model
    # -------------------------------

    logger.info("Saving the fine-tuned model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("Fine-tuning and evaluation complete.")

if __name__ == "__main__":
    main()