import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse
import os
from huggingface_hub import login

class ModelFinetuner:
    def __init__(
        self,
        model_name: str,
        csv_path: str,
        text_column: str,
        label_column: str,
        output_dir: str,
        hf_token: str = None,
        task_type: str = "classification"
    ):
        self.model_name = model_name
        self.csv_path = csv_path
        self.text_column = text_column
        self.label_column = label_column
        self.output_dir = output_dir
        self.task_type = task_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Login to Hugging Face if token provided
        if hf_token:
            login(hf_token)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_data(self):
        """Load and prepare the dataset"""
        try:
            # Read CSV
            print(f"Reading CSV file from {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            print(f"Dataset shape: {df.shape}")
            
            # Verify columns exist
            if self.text_column not in df.columns or self.label_column not in df.columns:
                raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}")
            
            # Create label mapping for classification tasks
            if self.task_type == "classification":
                unique_labels = df[self.label_column].unique()
                print(f"Unique labels found: {unique_labels}")
                self.label2id = {label: i for i, label in enumerate(unique_labels)}
                self.id2label = {i: label for label, i in self.label2id.items()}
                df['label_ids'] = df[self.label_column].map(self.label2id)
                
                # Save label mappings
                pd.DataFrame({
                    'label': list(self.label2id.keys()),
                    'id': list(self.label2id.values())
                }).to_csv(f"{self.output_dir}/label_mappings.csv", index=False)
            
            # Split data
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
            print(f"Train set size: {len(train_df)}, Validation set size: {len(val_df)}")
            
            # Convert to HF datasets
            self.train_dataset = Dataset.from_pandas(train_df)
            self.val_dataset = Dataset.from_pandas(val_df)
            
            # Initialize tokenizer
            print(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # Use AutoTokenizer for general compatibility
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except Exception as e:
            raise Exception(f"Error in prepare_data: {str(e)}")
            
    def tokenize_function(self, examples):
        """Tokenize text data"""
        return self.tokenizer(
            examples[self.text_column],
            padding="max_length",
            truncation=True,
            max_length=128,  # Adjust based on your needs
            return_tensors=None  # Return as lists
        )
            
    def prepare_features(self, examples):
        """Prepare features for training"""
        features = self.tokenize_function(examples)
        features["labels"] = examples["label_ids"]
        return features
        
    def tokenize_data(self):
        """Tokenize the datasets"""
        print("Tokenizing training data...")
        self.tokenized_train = self.train_dataset.map(
            self.prepare_features,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        print("Tokenizing validation data...")
        self.tokenized_val = self.val_dataset.map(
            self.prepare_features,
            batched=True,
            remove_columns=self.val_dataset.column_names
        )
        
        print("Tokenization complete!")
        
    def train(self, num_epochs: int = 3, batch_size: int = 8):
        """Train the model"""
        print("Initializing model...")
        # Initialize model
        if self.task_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            ).to(self.device)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            push_to_hub=False,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10
        )
        
        # Initialize trainer
        print("Initializing trainer...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer)
        )
        
        # Train
        print("Starting training...")
        self.trainer.train()
        
        # Save the model
        print("Saving model...")
        model_save_path = f"{self.output_dir}/final_model"
        self.trainer.save_model(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path}")

def main():
    parser = argparse.ArgumentParser(description='Finetune a Hugging Face model')
    parser.add_argument('--model_name', type=str, default="distilbert-base-uncased",
                      help='Name of the Hugging Face model to finetune')
    parser.add_argument('--csv_path', type=str, required=True,
                      help='Path to the CSV file containing the dataset')
    parser.add_argument('--text_column', type=str, required=True,
                      help='Name of the column containing the text data')
    parser.add_argument('--label_column', type=str, required=True,
                      help='Name of the column containing the labels')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Training batch size')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save the model')
    parser.add_argument('--hf_token', type=str, default=None,
                      help='Hugging Face API token for accessing private models')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"finetuned_model_{timestamp}"
    
    try:
        print("\nInitializing fine-tuning process...")
        print(f"Model: {args.model_name}")
        print(f"Output directory: {args.output_dir}")
        
        # Initialize finetuner
        finetuner = ModelFinetuner(
            model_name=args.model_name,
            csv_path=args.csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
            output_dir=args.output_dir,
            hf_token=args.hf_token
        )
        
        # Prepare and train
        print("\nPreparing data...")
        finetuner.prepare_data()
        
        print("\nTokenizing data...")
        finetuner.tokenize_data()
        
        print(f"\nTraining model for {args.epochs} epochs...")
        finetuner.train(num_epochs=args.epochs, batch_size=args.batch_size)
        
        print("\nTraining complete!")
        print(f"Model saved to: {args.output_dir}/final_model")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you're using a valid model name from Hugging Face")
        print("2. Verify the CSV file exists and has the correct columns")
        print("3. If using a private model, provide your Hugging Face token")
        print("4. Check your internet connection")

if __name__ == "__main__":
    main()
