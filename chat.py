import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import config
import re
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation parameters."""
    temperature: float = 0.6
    top_p: float = 0.8
    max_new_tokens: int = 150
    min_new_tokens: int = 10
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    do_sample: bool = True
    num_beams: int = 1

class ConversationManager:
    """Manages the conversation state and model interactions."""
    
    def __init__(self, model_path: str, adapter_path: str):
        """Initialize the conversation manager with model paths."""
        self.device = self._get_device()
        self.model, self.tokenizer = self._initialize_model(model_path, adapter_path)
        self.generation_config = GenerationConfig()
        
    @staticmethod
    def _get_device() -> str:
        """Determine the appropriate device for model execution."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _initialize_model(self, model_path: str, adapter_path: str) -> tuple:
        """Initialize the model and tokenizer with error handling."""
        try:
            logger.info(f"Loading base model from {model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Loading fine-tuned adapters from {adapter_path}")
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=False
            )
            model.eval()
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def _clean_response(self, text: str) -> str:
        """Clean the model's response text."""
        # Remove any system prompts or instruction tokens
        patterns = [
            r'\[INST\].*?\[/INST\]',
            r'<<SYS>>.*?<</SYS>>',
            r'You:.*?$',
            r'User:.*?$',
            r'Assistant:.*?$',
            r'System:.*?$',
            r'<br>.*?$',
            r':/+.*?$',
            r'\[.*?\]',
        ]
        
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        # Remove multiple spaces and clean up whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove any trailing incomplete sentences
        if cleaned.count('.') > 0:
            last_period = cleaned.rindex('.')
            if len(cleaned) - last_period > 30:  # If there's a long incomplete sentence
                cleaned = cleaned[:last_period + 1]
        
        return cleaned.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(self, prompt: str) -> str:
        """Generate a response with retry logic for robustness."""
        try:
            # Construct the prompt with system context
            system_prompt = "You're a chill, casual AI assistant. Keep responses natural and conversational."
            formatted_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.max_new_tokens,
                    min_new_tokens=self.generation_config.min_new_tokens,
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    do_sample=self.generation_config.do_sample,
                    num_beams=self.generation_config.num_beams,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=self.generation_config.repetition_penalty,
                    no_repeat_ngram_size=self.generation_config.no_repeat_ngram_size
                )
            
            # Decode and clean response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            clean_response = full_response.replace(formatted_prompt, '').strip()
            final_response = self._clean_response(clean_response)
            
            # Validate response
            if len(final_response.split()) < 3:
                raise ValueError("Response too short")
                
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        # Initialize conversation manager
        manager = ConversationManager(
            model_path=config.model_name,
            adapter_path=config.output_dir
        )
        
        print("\n🤖 AI Assistant initialized! (Type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == 'quit':
                    print("\nPeace out! ✌️")
                    break
                    
                if not user_input:
                    continue
                
                response = manager.generate_response(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nExiting gracefully...")
                break
                
            except Exception as e:
                logger.error(f"Error in conversation loop: {str(e)}")
                print("\nMy bad, something went wrong. Let's try again!")
                
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print("\nFailed to initialize the assistant. Please check the logs.")

if __name__ == "__main__":
    main()