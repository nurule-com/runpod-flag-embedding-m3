"""
Model loader for BGE-M3 embedding model.
Handles loading the model and cleanup of resources.
"""

import os
import atexit
from FlagEmbedding import BGEM3FlagModel

# Global model variable
model = None

def get_model():
    """
    Get the BGE-M3 model instance.
    If the model is not initialized, initialize it first.
    
    Returns:
        The model instance
    """
    global model
    
    # Initialize the model if it's not already loaded
    if model is None:
        load_model()
    
    return model

def load_model():
    """
    Load the BGE-M3 model.
    This ensures it's only loaded once when the worker starts.
    """
    global model
    
    # Get GPU device from environment variable or use default
    gpu_device = os.environ.get("GPU_DEVICE", "cuda:0")
    model_name = "BAAI/bge-m3"
    
    # Disable hf_transfer if the package is not available
    # This prevents the error when HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not installed
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        try:
            import hf_transfer
            print("Using hf_transfer for fast downloads")
        except ImportError:
            print("hf_transfer package not found, disabling fast downloads")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    try:
        print(f"Loading {model_name} model on {gpu_device}...")
        
        # Use the specific FlagEmbedding implementation for BGE-M3
        # Explicitly specify the GPU device to use
        model = BGEM3FlagModel(
            model_name, 
            use_fp16=True,
            device=gpu_device
        )
        
        print(f"Successfully loaded {model_name} model on {gpu_device}")
        
        # Register the cleanup function to be called at exit
        atexit.register(cleanup_model)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def cleanup_model():
    """
    Properly clean up model resources before shutdown.
    This prevents errors during Python's shutdown process.
    """
    global model
    
    if model is not None:
        try:
            # Access model attributes that might be needed during cleanup
            # This prevents them from being garbage collected too early
            if hasattr(model, '_pool') and model._pool is not None:
                model._pool = None
            if hasattr(model, '_tokenizer') and model._tokenizer is not None:
                model._tokenizer = None
            
            # Set model to None to avoid further cleanup attempts
            model = None
            print("Model resources cleaned up successfully")
        except Exception as e:
            print(f"Error during model cleanup: {e}") 