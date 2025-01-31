import google.generativeai as genai
import json
import csv
from datetime import datetime
from typing import List, Dict
import os
import re

class DatasetGenerator:
    def __init__(self, api_key: str):
        """Initialize the dataset generator with Google API key"""
        self.dataset = []
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def clean_json_response(self, content: str) -> str:
        """Clean and validate JSON response with enhanced cleaning"""
        try:
            # Remove any markdown formatting and extra text
            content = re.sub(r'```json\s*|\s*```', '', content)
            content = re.sub(r'```\s*|\s*```', '', content)
            
            # Find the JSON array
            content = content.strip()
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx + 1]
            else:
                return "[]"
            
            # Enhanced JSON cleaning
            # Remove comments and whitespace
            content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.DOTALL)
            content = re.sub(r'\s+', ' ', content)
            
            # Fix quotes and ensure proper JSON formatting
            content = content.replace("'", '"')
            content = re.sub(r'(\w+)(?=\s*:)', r'"\1"', content)
            content = re.sub(r':\s*"?\b(true|false|null)\b"?', r': \1', content)
            content = re.sub(r':\s*"?(-?\d+\.?\d*)"?([,}\]])', r': \1\2', content)
            
            # Remove trailing commas
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            
            # Validate JSON
            json.loads(content)
            return content
            
        except Exception as e:
            print(f"Error cleaning JSON: {str(e)}")
            return "[]"

    def parse_examples(self, content: str) -> List[Dict]:
        """Parse examples from model output and validate format"""
        try:
            # Parse JSON
            cleaned_content = self.clean_json_response(content)
            examples = json.loads(cleaned_content)
            
            if not isinstance(examples, list):
                return []
            
            # Basic validation of required fields
            valid_examples = []
            for example in examples:
                if isinstance(example, dict):
                    valid_examples.append(example)
            
            return valid_examples
        except Exception as e:
            print(f"Error parsing examples: {str(e)}")
            return []

    def validate_example(self, example: Dict, task_type: str) -> bool:
        """Validate the structure of generated examples"""
        try:
            if task_type == "sentiment_analysis":
                required_fields = {"text", "sentiment"}
                return all(field in example for field in required_fields)
                
            elif task_type == "topic_classification":
                required_fields = {"text", "topic"}
                return all(field in example for field in required_fields)
                
            elif task_type == "question_answering":
                required_fields = {"question", "answer"}
                return all(field in example for field in required_fields)
                
            elif task_type == "recipe_generation":
                required_fields = {"name", "ingredients", "instructions", "prep_time"}
                return (all(field in example for field in required_fields) and
                        isinstance(example["ingredients"], list) and
                        isinstance(example["instructions"], list))
                
            elif task_type == "fitness_plan":
                required_fields = {"name", "duration", "exercises", "diet"}
                if not all(field in example for field in required_fields):
                    return False
                if not isinstance(example["exercises"], list):
                    return False
                if not isinstance(example["diet"], dict):
                    return False
                return True
                
            elif task_type == "weather_prediction":
                required_fields = {
                    "location", "current_temp", "conditions", "humidity", "wind", "forecast"
                }
                forecast_fields = {
                    "tomorrow", "temp_high", "temp_low", "rain_chance"
                }
                
                # Check main fields
                if not all(field in example for field in required_fields):
                    return False
                    
                # Check forecast fields
                if not all(field in example["forecast"] for field in forecast_fields):
                    return False
                    
                # Validate data types
                if not (isinstance(example["current_temp"], (int, float)) and
                        isinstance(example["humidity"], (int, float)) and
                        isinstance(example["wind"], (int, float))):
                    return False
                
                return True
                
            elif task_type == "travel_itinerary":
                required_fields = {
                    "destination", "days", "total_budget", "hotel",
                    "day1_morning", "day1_afternoon", "day1_evening",
                    "transport", "notes"
                }
                
                # Check required fields
                if not all(field in example for field in required_fields):
                    return False
                    
                # Validate data types
                if not (isinstance(example["days"], int) and
                        isinstance(example["total_budget"], (int, float))):
                    return False
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    def generate_examples(self, task_type: str, num_samples: int = 10) -> None:
        """Generate task-specific examples using Gemini"""
        task_prompts = {
            "sentiment_analysis": """
Generate exactly one JSON array containing sentiment analysis examples. Format:
[
    {
        "text": "This movie was fantastic!",
        "sentiment": "positive"
    },
    {
        "text": "Terrible service, would not recommend.",
        "sentiment": "negative"
    }
]
""",
            "topic_classification": """
Generate exactly one JSON array containing topic classification examples. Format:
[
    {
        "text": "NASA launches new Mars mission",
        "topic": "science"
    },
    {
        "text": "Stock market hits record high",
        "topic": "finance"
    }
]
""",
            "question_answering": """
Generate exactly one JSON array containing QA examples. Format:
[
    {
        "question": "What is the capital of France?",
        "answer": "Paris"
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare"
    }
]
""",
            "recipe_generation": """
Generate exactly one JSON array containing recipe examples. Format:
[
    {
        "name": "Chocolate Chip Cookies",
        "ingredients": ["flour", "sugar", "butter", "chocolate chips"],
        "instructions": ["Mix ingredients", "Bake at 350F"],
        "prep_time": "20 minutes"
    }
]
""",
            "fitness_plan": """
Generate exactly one JSON array containing simple fitness plans. Format:
[
    {
        "name": "Beginner Weight Loss",
        "duration": "4 weeks",
        "exercises": [
            {
                "name": "Walking",
                "duration": "30 minutes",
                "frequency": "3 times per week"
            }
        ],
        "diet": {
            "calories": 2000,
            "protein": "150g",
            "carbs": "200g"
        }
    }
]
""",
            "weather_prediction": """
Generate exactly one JSON array containing weather predictions. Follow this exact format without any modifications:
[
    {
        "location": "New York",
        "current_temp": 72,
        "conditions": "sunny",
        "humidity": 65,
        "wind": 10,
        "forecast": {
            "tomorrow": "partly cloudy",
            "temp_high": 75,
            "temp_low": 60,
            "rain_chance": 20
        }
    }
]
""",
            "travel_itinerary": """
Generate exactly one JSON array containing travel plans. Follow this exact format without any modifications:
[
    {
        "destination": "Paris",
        "days": 3,
        "total_budget": 1500,
        "hotel": "City Center Hotel",
        "day1_morning": "Visit Eiffel Tower",
        "day1_afternoon": "Louvre Museum Tour",
        "day1_evening": "Seine River Cruise",
        "day2_morning": "Notre Dame Cathedral",
        "day2_afternoon": "Shopping at Champs-Élysées",
        "day2_evening": "French Restaurant Dinner",
        "day3_morning": "Palace of Versailles",
        "day3_afternoon": "Arc de Triomphe",
        "day3_evening": "Food and Wine Tasting",
        "transport": "Metro and Walking",
        "notes": "Book museum tickets in advance"
    }
]
"""
        }

        if task_type not in task_prompts:
            raise ValueError(f"Unsupported task type: {task_type}")

        remaining_samples = num_samples
        batch_size = 2  # Smaller batch size for complex tasks
        max_retries = 5  # Increased retries

        while remaining_samples > 0:
            current_batch = min(batch_size, remaining_samples)
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    # Enhanced prompt with strict formatting instructions
                    base_prompt = task_prompts[task_type]
                    prompt = f"""
Generate exactly {current_batch} examples following the format below.
Important instructions:
- Generate ONLY a valid JSON array
- Do not include any explanation or additional text
- Do not use special characters or emojis
- Use only the exact fields shown in the example
- Ensure all values are properly quoted
- Do not add new fields or nested structures

Format:
{base_prompt}
"""
                    response = self.model.generate_content(prompt)
                    
                    if not response.text:
                        print("Empty response received")
                        retry_count += 1
                        continue

                    examples = self.parse_examples(response.text)
                    
                    if examples:
                        valid_examples = 0
                        for example in examples:
                            if self.validate_example(example, task_type):
                                self.add_example(example, task_type)
                                valid_examples += 1
                                remaining_samples -= 1
                        if valid_examples > 0:
                            success = True
                    
                    if not success:
                        retry_count += 1
                        print(f"Retry {retry_count}: No valid examples generated")

                except Exception as e:
                    print(f"Error in generation: {str(e)}")
                    retry_count += 1

            if not success:
                print(f"Failed to generate batch after {max_retries} retries")
                break

    def add_example(self, example: Dict, task_type: str) -> None:
        """Add a single example to the dataset with metadata"""
        example_with_metadata = {
            "data": example,
            "metadata": {
                "task_type": task_type,
                "timestamp": datetime.now().isoformat(),
                "generation_model": "gemini-pro"
            }
        }
        self.dataset.append(example_with_metadata)

    def save_csv(self, filename: str) -> None:
        """Save dataset to CSV format"""
        if not self.dataset:
            print("No data to save!")
            return
        
        try:
            # Determine fields based on the first example's structure
            first_example = self.dataset[0]["data"]
            fields = list(first_example.keys()) + ["timestamp", "task_type", "generation_model"]
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                
                for item in self.dataset:
                    row = {**item["data"], 
                          "timestamp": item["metadata"]["timestamp"],
                          "task_type": item["metadata"]["task_type"],
                          "generation_model": item["metadata"]["generation_model"]}
                    writer.writerow(row)
            print(f"Successfully saved {filename}")
        except Exception as e:
            print(f"Error saving CSV: {str(e)}")

    def save_jsonl(self, filename: str) -> None:
        """Save dataset in JSONL format"""
        if not self.dataset:
            print("No data to save!")
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for item in self.dataset:
                    f.write(json.dumps(item) + '\n')
            print(f"Successfully saved {filename}")
        except Exception as e:
            print(f"Error saving JSONL: {str(e)}")

    def print_sample(self, num_samples: int = 3) -> None:
        """Print a few samples from the generated dataset"""
        if not self.dataset:
            print("No samples to show!")
            return
            
        print(f"\nShowing {min(num_samples, len(self.dataset))} samples:")
        for item in self.dataset[:num_samples]:
            print("\nExample:")
            print(json.dumps(item["data"], indent=2))

def main():
    # Get API key from environment variable or use default
    api_key = 'enter you api key'
    
    # Initialize generator
    generator = DatasetGenerator(api_key)
    
    # Available task types
    task_types = {
        "1": "sentiment_analysis",
        "2": "topic_classification",
        "3": "question_answering",
        "4": "recipe_generation",
        "5": "fitness_plan",
        "6": "weather_prediction",
        "7": "travel_itinerary"
    }
    
    # Print available tasks
    print("\nAvailable task types:")
    for key, value in task_types.items():
        print(f"{key}. {value.replace('_', ' ').title()}")
    
    # Get task type
    while True:
        choice = input("\nPlease select a task type (1-7): ")
        if choice in task_types:
            task_type = task_types[choice]
            break
        print("Invalid choice. Please select 1-7.")
    
    # Get number of samples
    while True:
        try:
            num_samples = int(input("\nHow many samples do you want to generate? (1-100): "))
            if 1 <= num_samples <= 100:
                break
            print("Please enter a number between 1 and 100.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nGenerating {num_samples} examples for {task_type}...")
    generator.generate_examples(task_type=task_type, num_samples=num_samples)
    
    # Print samples
    generator.print_sample(3)
    
    # Save results
    if generator.dataset:
        generator.save_csv(f'{task_type}_dataset.csv')
        generator.save_jsonl(f'{task_type}_dataset.jsonl')
        print(f"\nGenerated {len(generator.dataset)} samples")
        print(f"Files saved: {task_type}_dataset.csv and {task_type}_dataset.jsonl")
    else:
        print("\nNo examples were generated successfully.")

if __name__ == "__main__":
    main()
