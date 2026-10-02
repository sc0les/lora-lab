import torch
from transformers import AutoTokenizer
from config import config
import sys
import os

def verify_environment():
    """Verify Python environment and dependencies"""
    print("\n=== Environment Information ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check CUDA/MPS availability
    print("\n=== Device Information ===")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Test device functionality
    try:
        x = torch.randn(2, 2).to(device)
        y = torch.randn(2, 2).to(device)
        z = x @ y
        print(f"\nTest computation successful on {device}")
    except Exception as e:
        print(f"\nError testing device: {e}")
        
    # Verify model access
    print("\n=== Model Access ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        print(f"Successfully accessed model tokenizer: {config.model_name}")
    except Exception as e:
        print(f"Error accessing model: {e}")
    
    # Verify directory structure
    print("\n=== Project Structure ===")
    required_dirs = ['data', 'results', 'utils']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ directory found")
        else:
            print(f"✗ {dir_name}/ directory missing")

if __name__ == "__main__":
    print("Starting environment verification...\n")
    verify_environment()