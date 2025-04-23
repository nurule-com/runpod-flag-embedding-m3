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

# Monkey-patch torch.nn.Module.to to safely move meta modules using to_empty
if hasattr(torch.nn.Module, 'to_empty'):
    _orig_to = torch.nn.Module.to
    def _safe_to(self, *args, **kwargs):
        # If any parameter is on meta device, use to_empty to allocate properly
        if any(p.device.type == 'meta' for p in self.parameters()):
            return torch.nn.Module.to_empty(self, *args, **kwargs)
        return _orig_to(self, *args, **kwargs)
    torch.nn.Module.to = _safe_to

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
        
        # Use the specific FlagEmbedding implementation for BGE-M3
        model = BGEM3FlagModel(
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