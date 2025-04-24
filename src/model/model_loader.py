import os
import atexit
import logging
from FlagEmbedding import BGEM3FlagModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

# Monkey-patch torch.nn.Module.to_empty to support both positional and keyword args
if hasattr(torch.nn.Module, 'to_empty'):
    _orig_to_empty = torch.nn.Module.to_empty
    def _safe_to_empty(self, *args, **kwargs):
        # Try original call first
        try:
            return _orig_to_empty(self, *args, **kwargs)
        except TypeError as e:
            msg = str(e)
            # Handle missing keyword-only 'device' error
            if "required keyword-only argument: 'device'" in msg:
                # Extract device and dtype from args or kwargs
                if args:
                    device = args[0]
                    dtype = args[1] if len(args) > 1 else None
                else:
                    device = kwargs.get('device') or os.environ.get('GPU_DEVICE', 'cuda:0')
                    dtype = kwargs.get('dtype', None)
                return _orig_to_empty(self, device=device, dtype=dtype)
            # Re-raise other TypeErrors
            raise
    torch.nn.Module.to_empty = _safe_to_empty

def get_model():
    """
    Get the BGE-M3 model instance.
    If the model is not initialized, initialize it first.
    
    Returns:
        BGEM3FlagModel: The model instance
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
    
    # Get amount of models from environment variable or use default
    amount_of_models = int(os.environ.get("AMOUNT_OF_MODELS", "10"))
    model = [None] * amount_of_models

    # Disable hf_transfer if the package is not available
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        try:
            import hf_transfer
            logger.info("Using hf_transfer for fast downloads")
        except ImportError:
            logger.warning("hf_transfer package not found, disabling fast downloads")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    try:
        logger.info(f"Loading {model_name} model on {gpu_device}...")
        
        for i in range(amount_of_models):
            # Use the specific FlagEmbedding implementation for BGE-M3
            model[i] = BGEM3FlagModel(
                model_name, 
                use_fp16=True,
                device=gpu_device
            )
        
        logger.info(f"Successfully loaded {model_name} model on {gpu_device}")
        
        # Register the cleanup function to be called at exit
        atexit.register(cleanup_model)
        
    except Exception as e:
        logger.exception("Error loading model")
        raise

def cleanup_model():
    """
    Properly clean up model resources before shutdown.
    This prevents errors during Python's shutdown process.
    """
    global model
    
    if model is not None:
        try:
            # Call cleanup method if it exists
            if hasattr(model, 'cleanup'):
                model.cleanup()
            
            # Set model to None to avoid further cleanup attempts
            model = None
            logger.info("Model resources cleaned up successfully")
        except Exception as e:
            logger.exception("Error during model cleanup")