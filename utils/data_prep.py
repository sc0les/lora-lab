from datasets import Dataset
import json
import logging
from typing import List, Dict
import os
from config import config

logger = logging.getLogger(__name__)

def prepare_dataset(file_path: str) -> Dataset:
    """
    Prepare dataset from a JSON file containing training examples.
    Expected format: List[Dict] where each dict has a 'text' key
    """
    logger.info(f"Loading data from {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to datasets format
    dataset = Dataset.from_dict({
        'text': [item['text'] for item in data]
    })
    
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset

def create_test_data():
    """Create a small test dataset."""
    test_data = [
        {
            "text": "This is a test sentence for fine-tuning."
        },
        {
            "text": "Another test sentence to verify training works."
        }
    ]
    
    test_file = os.path.join(config.data_path, "test_data.json")
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"Created test data at {test_file}")
    return test_file

def format_conversation_with_system_prompt(system_prompt: str, conversations: List[Dict]) -> List[Dict]:
    """
    Format conversations with a system prompt for training.
    """
    formatted_data = []
    
    for conv in conversations:
        # Format in Llama chat style
        formatted_text = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        
        # Add the conversation messages
        for i, message in enumerate(conv['messages']):
            if message['role'] == 'user':
                formatted_text += f"{message['content']} [/INST] "
            elif message['role'] == 'assistant':
                formatted_text += f"{message['content']} "
                # Add [INST] for next turn if not last message
                if i < len(conv['messages']) - 1:
                    formatted_text += "[INST] "
        
        formatted_data.append({
            "text": formatted_text.strip()
        })
    
    return formatted_data

def create_training_data(system_prompt: str):
    """Create training data with system prompt."""
    # Example conversations
    conversations = [
        {
            "messages": [
                {"role": "user", "content": "Hey, how's it going?"},
                {"role": "assistant", "content": "Pretty good! What's on your mind?"}
            ]
        }
        # Add more conversations as needed
    ]
    
    formatted_data = format_conversation_with_system_prompt(system_prompt, conversations)
    
    training_file = os.path.join(config.data_path, "training_data.json")
    with open(training_file, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    logger.info(f"Created training data at {training_file}")
    return training_file

if __name__ == "__main__":
    # Create test data when run directly
    create_test_data()