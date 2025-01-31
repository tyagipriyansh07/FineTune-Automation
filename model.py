from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

class ModelInference:
    def __init__(self, model_path: str):
        """Initialize with path to finetuned model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        
        # Load label mappings
        mappings_path = f"{model_path}/../label_mappings.csv"
        mappings_df = pd.read_csv(mappings_path)
        self.id2label = dict(zip(mappings_df['id'], mappings_df['label']))
        
    def predict(self, texts):
        """Make predictions on new texts"""
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.softmax(dim=-1)
            
        # Process results
        results = []
        for text, probs in zip(texts, predictions):
            label_id = probs.argmax().item()
            confidence = probs.max().item()
            results.append({
                "text": text,
                "predicted_label": self.id2label[label_id],
                "confidence": confidence
            })
            
        return results

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Use a finetuned model for prediction')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the finetuned model')
    parser.add_argument('--text', type=str, required=True,
                      help='Text to classify')
    
    args = parser.parse_args()
    
    # Initialize model
    classifier = ModelInference(args.model_path)
    
    # Make prediction
    results = classifier.predict([args.text])
    
    # Print results
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Predicted Label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2f}")