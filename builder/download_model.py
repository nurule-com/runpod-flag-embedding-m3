import os
from FlagEmbedding import BGEM3FlagModel

def download_model():
    """
    Download the BGE-M3 model during Docker build time.
    This ensures the model is cached in the container image.
    """
    model_name = "BAAI/bge-m3"
    
    # Enable hf_transfer for faster downloads if available
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    try:
        import hf_transfer
        print("Using hf_transfer for fast downloads")
    except ImportError:
        print("hf_transfer package not found, falling back to standard download")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    print(f"Downloading {model_name} model...")
    
    # Temporarily force CPU device for model download during build
    # This won't initialize the full model but will download all files
    try:
        # Initialize model class to trigger downloads
        # Setting download_only=True to avoid full model initialization
        model = BGEM3FlagModel(
            model_name,
            use_fp16=True,
            device="cpu"
        )
        print(f"Successfully downloaded {model_name} model")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

if __name__ == "__main__":
    download_model() 