import os
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model
)
from datasets import Dataset
from config import config
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
    """Load the base model and tokenizer."""
    logger.info(f"Loading model and tokenizer: {config.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.float32
    )
    
    return model, tokenizer

def prepare_model(model):
    """Prepare the model with LoRA for training."""
    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    logger.info("Preparing model for training")
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    
    return model

def prepare_dataset(tokenizer):
    """Load and prepare the dataset with system prompt."""
    logger.info("Loading dataset")
    
    # Load the data from your JSON file
    with open(os.path.join(config.data_path, "training_data.json"), 'r') as f:
        data = json.load(f)
    
    # Create dataset
    dataset = Dataset.from_dict({"text": [item["text"] for item in data]})
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        ),
        remove_columns=["text"]
    )
    
    return tokenized_dataset

def train(model, tokenizer, dataset):
    """Configure and run the training."""
    logger.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=False,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_safetensors=True,
        optim="adamw_torch",
        logging_dir=f"{config.output_dir}/logs",
    )
    
    logger.info("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    
    logger.info("Starting training")
    trainer.train()
    
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model()

def main():
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Prepare model for training
    model = prepare_model(model)
    
    # Load and prepare dataset
    dataset = prepare_dataset(tokenizer)
    
    # Train
    train(model, tokenizer, dataset)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()