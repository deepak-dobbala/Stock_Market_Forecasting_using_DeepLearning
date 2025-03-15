import torch
import sys

def check_gpu():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Current GPU device: {torch.cuda.get_device_name()}")
        # Test GPU with a simple tensor operation
        x = torch.rand(5, 3)
        print("Testing GPU tensor:")
        print(x.to('cuda'))
    else:
        print("No CUDA GPU available. Please check your installation.")

if __name__ == "__main__":
    check_gpu()