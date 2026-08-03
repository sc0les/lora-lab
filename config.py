from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model settings
    model_name: str = "NousResearch/Hermes-3-Llama-3.1-8B"
    
    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 20
    weight_decay: float = 0.001
    
    # Training monitoring settings
    logging_steps: int = 1
    save_steps: int = 5
    
    # Paths
    output_dir: str = "./results"
    data_path: str = "./data"

config = TrainingConfig()