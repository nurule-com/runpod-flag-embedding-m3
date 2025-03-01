"""
Model loader for BGE-M3 embedding model.
Handles loading the model and cleanup of resources.
"""

import os
import atexit
import threading
from FlagEmbedding import BGEM3FlagModel

# Global model pool
model_pool = []
model_pool_lock = threading.Lock()
MAX_MODELS = 4  # Number of model instances to load (adjust based on GPU memory)

def load_model_pool():
    """
    Load multiple instances of the BGE-M3 model to better utilize GPU memory.
    This is called once at startup.
    """
    global model_pool
    
    # Get GPU device from environment variable or use default
    gpu_device = os.environ.get("GPU_DEVICE", "cuda:0")
    model_name = "BAAI/bge-m3"
    
    try:
        print(f"Loading {MAX_MODELS} instances of {model_name} model on {gpu_device}...")
        
        for i in range(MAX_MODELS):
            # Use the specific FlagEmbedding implementation for BGE-M3
            # Explicitly specify the GPU device to use
            model = BGEM3FlagModel(
                model_name, 
                use_fp16=True,
                device=gpu_device
            )
            model_pool.append(model)
            print(f"Successfully loaded model instance {i+1}/{MAX_MODELS}")
        
        # Register the cleanup function to be called at exit
        atexit.register(cleanup_model_pool)
        
        print(f"Model pool initialized with {len(model_pool)} instances")
        return True
        
    except Exception as e:
        print(f"Error loading model pool: {e}")
        raise

def get_model():
    """
    Get a model instance from the pool.
    If the pool is not initialized, initialize it first.
    
    Returns:
        A model instance from the pool
    """
    global model_pool
    
    # Initialize the pool if it's empty
    if not model_pool:
        with model_pool_lock:
            if not model_pool:  # Double-check to avoid race condition
                load_model_pool()
    
    # Get the first available model from the pool
    # In a more sophisticated implementation, you could implement a checkout/checkin system
    # For now, we'll use a simple round-robin approach
    with model_pool_lock:
        if model_pool:
            # Move the first model to the end of the list and return it
            model = model_pool[0]
            model_pool = model_pool[1:] + [model]
            return model
        else:
            raise RuntimeError("Failed to initialize model pool")

def cleanup_model_pool():
    """
    Properly clean up all model resources before shutdown.
    This prevents errors during Python's shutdown process.
    """
    global model_pool
    
    print(f"Cleaning up {len(model_pool)} model instances...")
    
    for i, model in enumerate(model_pool):
        try:
            # Access model attributes that might be needed during cleanup
            # This prevents them from being garbage collected too early
            if hasattr(model, '_pool') and model._pool is not None:
                model._pool = None
            if hasattr(model, '_tokenizer') and model._tokenizer is not None:
                model._tokenizer = None
            print(f"Cleaned up model instance {i+1}/{len(model_pool)}")
        except Exception as e:
            print(f"Error during model cleanup for instance {i+1}: {e}")
    
    # Clear the pool
    model_pool = []
    print("Model pool resources cleaned up successfully") 