""" 
RunPod worker for generating embeddings using BGE-M3 model.
This worker takes an array of texts and returns sparse, dense, and colbert vectors for each text.
"""

import os
import runpod
import numpy as np
import atexit
from FlagEmbedding import BGEM3FlagModel

# Load the BGE-M3 model outside the handler function
# This ensures it's only loaded once when the worker starts
MODEL_NAME = "BAAI/bge-m3"

# Get GPU device from environment variable or use default
GPU_DEVICE = os.environ.get("GPU_DEVICE", "cuda:0")

# Global model variable
model = None

try:
    # Use the specific FlagEmbedding implementation for BGE-M3
    # Explicitly specify the GPU device to use
    model = BGEM3FlagModel(
        MODEL_NAME, 
        use_fp16=True,
        device=GPU_DEVICE
    )
    print(f"Successfully loaded {MODEL_NAME} model on {GPU_DEVICE}")
    
    # Define a cleanup function to properly handle model resources
    def cleanup_model():
        global model
        if model is not None:
            # Explicitly clean up resources before the interpreter shuts down
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
    
    # Register the cleanup function to be called at exit
    atexit.register(cleanup_model)
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def process_texts(texts, is_passage=False, batch_size=8):
    """
    Process a list of texts and return sparse, dense, and colbert embeddings for each.
    
    Args:
        texts: List of text strings to encode
        is_passage: Whether the texts are passages (True) or queries (False)
        batch_size: Number of texts to process at once
        
    Returns:
        List of dictionaries containing the embeddings for each text
    """
    results = []
    
    # Process texts in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        try:
            # Use the appropriate encoding method based on text type
            if is_passage:
                # For passages/documents
                embeddings = model.encode_corpus(
                    batch_texts,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True
                )
            else:
                # For queries
                embeddings = model.encode_queries(
                    batch_texts,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True
                )
            
            # Process each text's embeddings
            for j, text in enumerate(batch_texts):
                try:
                    # Extract embeddings for this text
                    text_result = {
                        "text": text,
                        "dense": embeddings["dense_vecs"][j].tolist()
                    }
                    
                    # Handle sparse embeddings with proper error handling
                    try:
                        # The lexical_weights key contains the sparse embeddings
                        if "lexical_weights" in embeddings:
                            sparse_weights = embeddings["lexical_weights"][j]
                            
                            # Check if sparse_weights is a dictionary-like object (common for BGE-M3)
                            if hasattr(sparse_weights, 'items'):
                                # For dictionary-like sparse weights (token_id -> weight)
                                indexes = []
                                values = []
                                
                                # Convert all keys to integers and values to floats
                                for token_id, weight in sparse_weights.items():
                                    # Convert string token_id to integer if needed
                                    if isinstance(token_id, str):
                                        token_id = int(token_id)
                                    # Convert numpy float to regular float
                                    if hasattr(weight, 'item'):  # Check if it's a numpy type
                                        weight = float(weight.item())
                                    else:
                                        weight = float(weight)
                                        
                                    indexes.append(token_id)
                                    values.append(weight)
                                
                                text_result["sparse"] = {
                                    "indices": indexes,
                                    "values": values
                                }
                            else:
                                # Handle the sparse weights as a numpy array
                                # Convert to numpy array if it's not already
                                if not isinstance(sparse_weights, np.ndarray):
                                    sparse_weights = np.array(sparse_weights)
                                
                                # Ensure we're working with at least a 1D array
                                sparse_weights = np.atleast_1d(sparse_weights)
                                
                                # Get nonzero indices and values
                                nonzero_indices = np.nonzero(sparse_weights)[0]
                                nonzero_values = sparse_weights[nonzero_indices]
                                
                                # Convert numpy types to Python native types for JSON serialization
                                text_result["sparse"] = {
                                    "indices": nonzero_indices.tolist(),
                                    "values": [float(v) for v in nonzero_values.tolist()]
                                }
                        else:
                            # Fallback if lexical_weights is not available
                            print(f"Warning: No lexical_weights found for text {j} in batch {i//batch_size}")
                            text_result["sparse"] = {
                                "indices": [],
                                "values": []
                            }
                    except Exception as e:
                        print(f"Warning: Error processing sparse embeddings for text {j} in batch {i//batch_size}: {e}")
                        print(f"Sparse weights type: {type(sparse_weights)}")
                        print(f"Sparse weights: {sparse_weights}")
                        # Provide empty sparse embeddings as fallback
                        text_result["sparse"] = {
                            "indices": [],
                            "values": []
                        }
                    
                    # Add colbert embeddings if available
                    if "colbert_vecs" in embeddings:
                        text_result["colbert"] = embeddings["colbert_vecs"][j].tolist()
                    
                    results.append(text_result)
                except Exception as e:
                    print(f"Error processing text {j} in batch {i//batch_size}: {e}")
                    print(f"Text: {text[:100]}...")
                    if "lexical_weights" in embeddings:
                        print(f"Lexical weights type: {type(embeddings['lexical_weights'])}")
                        print(f"Lexical weights shape: {np.array(embeddings['lexical_weights']).shape if isinstance(embeddings['lexical_weights'], (list, np.ndarray)) else 'Not array-like'}")
                    raise
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            print(f"Batch texts: {[t[:50] + '...' for t in batch_texts]}")
            if "embeddings" in locals():
                print(f"Embeddings keys: {embeddings.keys() if isinstance(embeddings, dict) else 'Not a dict'}")
            raise
    
    return results

def handler(job):
    """
    Handler function that processes incoming jobs.
    
    Expected input format:
    {
        "texts": ["text1", "text2", ...],
        "isPassage": false,  // Optional, defaults to false (query mode)
        "batchSize": 8       // Optional, defaults to 8
    }
    
    Returns:
        List of dictionaries containing the embeddings for each text
    """
    job_input = job["input"]
    
    # Validate input
    if "texts" not in job_input:
        return {"error": "Missing 'texts' field in input"}
    
    texts = job_input["texts"]
    
    # Validate texts is a list
    if not isinstance(texts, list):
        return {"error": "The 'texts' field must be a list of strings"}
    
    # Process empty list case
    if len(texts) == 0:
        return {"results": []}
    
    # Check if texts are passages or queries
    is_passage = job_input.get("isPassage", False)
    
    # Get batch size
    batch_size = job_input.get("batchSize", 8)
    
    try:
        # Process the texts
        print(f"Processing {len(texts)} texts (isPassage={is_passage}, batchSize={batch_size})")
        results = process_texts(texts, is_passage=is_passage, batch_size=batch_size)
        print(f"Successfully processed {len(results)} texts")

        print(results)
        
        return {"results": results}
    except Exception as e:
        return {"error": f"Error processing texts: {str(e)}"}

# Start the serverless worker
runpod.serverless.start({"handler": handler})
