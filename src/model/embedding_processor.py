"""
Embedding processor for BGE-M3 model.
Handles processing texts and extracting embeddings.
"""

import threading
import time
import numpy as np
import asyncio
from .model_loader import get_model
from runpod import RunPodLogger

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

logger = RunPodLogger()

def process_texts_sync(texts, is_passage=False, batch_size=0):
    """
    Synchronous version of process_texts for use with thread pools.
    Process a list of texts and return embeddings for each.
    
    Args:
        texts: List of text strings to encode
        is_passage: Whether the texts are passages (True) or queries (False)
        batch_size: Number of texts to process at once. If 0 or negative, all texts are processed at once (default).
        
    Returns:
        List of dictionaries containing the embeddings for each text
    """

    # Start timing
    start_time = time.time()

    model = get_model()
    tokenizer = model.tokenizer
    results = []
    
    if not texts:
        return []

    try:
        # Tokenize all texts first for optimal batching
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="np"
        )
        token_counts = [len(ids) for ids in tokenized["input_ids"]]
    except Exception as e:
        print(f"Tokenization failed: {str(e)}")
        return [{"error": "Text processing failed"} for _ in texts]

    # Determine batch size if not provided
    if batch_size <= 0:
        batch_size = max(1, 512 // max(token_counts)) if token_counts else 1
    
    # Process in optimized batches
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        
        # Create a thread to process the batch
        batch_results = []
        stop_event = threading.Event()  # Event to signal the thread to stop

        def process_batch():
            nonlocal batch_results
            try:
                # Use existing model methods for encoding
                if is_passage:
                    embeddings = model.encode_corpus(
                        batch_texts,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=True
                    )
                else:
                    embeddings = model.encode_queries(
                        batch_texts,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=True
                    )
                
                # Process embeddings
                for j, text in enumerate(batch_texts):
                    text_result = {
                        "text": text,
                        "dense": embeddings["dense_vecs"][j].tolist()
                    }

                    # Process sparse embeddings
                    sparse_weights = embeddings.get("lexical_weights", [None])[j]
                    if sparse_weights is not None:
                        if isinstance(sparse_weights, dict):
                            indexes = []
                            values = []
                            for token_id, weight in sparse_weights.items():
                                indexes.append(int(token_id))
                                values.append(float(weight))
                        else:
                            # Handle array format
                            sparse_weights = np.array(sparse_weights)
                            nonzero = sparse_weights.nonzero()
                            indexes = nonzero[0].tolist()
                            values = sparse_weights[nonzero].tolist()
                        
                        if stop_event.is_set():  # Check if we should stop processing
                            logger.error(f"{RED}Stopping processing {batch_texts} {j}{RESET}")
                            return
                        
                        text_result["sparse"] = {
                            "indices": indexes,
                            "values": values
                        }
                    else:
                        text_result["sparse"] = {
                            "indices": [],
                            "values": []
                        }

                    if stop_event.is_set():  # Check if we should stop processing
                        logger.error(f"{RED}Stopping processing {batch_texts} {j}{RESET}")
                        return

                    # Add ColBERT embeddings if available
                    if "colbert_vecs" in embeddings:
                        text_result["colbert"] = embeddings["colbert_vecs"][j].tolist()

                    if stop_event.is_set():  # Check if we should stop processing
                        logger.error(f"{RED}Stopping processing {batch_texts} {j}{RESET}")
                        return

                    batch_results.append(text_result)

            except Exception as e:
                print(f"Batch {batch_idx // batch_size} failed: {str(e)}")
                batch_results.extend(
                    {"error": "Processing failed", "text": text} 
                    for text in batch_texts
                )

        logger.info(f"{GREEN}Process started{RESET}")

        # Start the thread
        thread = threading.Thread(target=process_batch)
        thread.start()
        
        # Wait for the thread to finish with a timeout
        thread.join(timeout=4)  # Wait for 4 seconds

        if thread.is_alive():
            print(f"Batch {batch_idx // batch_size} processing timed out.")
            stop_event.set()  # Signal the thread to stop
            results.extend(
                {"error": "Processing timed out", "text": text} 
                for text in batch_texts
            )
        else:
            # If the thread finished successfully, add the results
            results.extend(batch_results)

    elapsed_time = time.time() - start_time
    logger.info(f"{GREEN}Elapsed time: {elapsed_time:.2f} seconds{RESET}")

    return results

async def process_texts(texts, is_passage=False, batch_size=0):
    """
    Process a list of texts and return sparse, dense, and colbert embeddings for each.
    
    Args:
        texts: List of text strings to encode
        is_passage: Whether the texts are passages (True) or queries (False)
        batch_size: Number of texts to process at once. If 0 or negative, all texts are processed at once (default).
        
    Returns:
        List of dictionaries containing the embeddings for each text
    """
    # Get a model instance from the pool
    model = get_model()
    
    results = []
    
    # Determine if we should use batches
    use_batches = batch_size > 0
    
    if use_batches:
        # Process texts in batches to avoid memory issues
        batch_ranges = [(i, min(i + batch_size, len(texts))) for i in range(0, len(texts), batch_size)]
    else:
        # Process all texts at once
        batch_ranges = [(0, len(texts))] if texts else []
    
    for start_idx, end_idx in batch_ranges:
        batch_texts = texts[start_idx:end_idx]
        
        try:
            # Use the appropriate encoding method based on text type
            # Note: The model encoding itself is not async, but we can make the function async
            # to allow other async operations to run concurrently
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
                            # print(f"Warning: No lexical_weights found for text {j} in batch {start_idx//batch_size if use_batches else 0}")
                            text_result["sparse"] = {
                                "indices": [],
                                "values": []
                            }
                    except Exception as e:
                        # print(f"Warning: Error processing sparse embeddings for text {j} in batch {start_idx//batch_size if use_batches else 0}: {e}")
                        # print(f"Sparse weights type: {type(sparse_weights)}")
                        # print(f"Sparse weights: {sparse_weights}")
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
                    # print(f"Error processing text {j} in batch {start_idx//batch_size if use_batches else 0}: {e}")
                    # print(f"Text: {text[:100]}...")
                    if "lexical_weights" in embeddings:
                        print(f"Lexical weights type: {type(embeddings['lexical_weights'])}")
                        # print(f"Lexical weights shape: {np.array(embeddings['lexical_weights']).shape if isinstance(embeddings['lexical_weights'], (list, np.ndarray)) else 'Not array-like'}")
                    raise
        except Exception as e:
            # print(f"Error processing batch {start_idx//batch_size if use_batches else 0}: {e}")
            # print(f"Batch texts: {[t[:50] + '...' for t in batch_texts]}")
            if "embeddings" in locals():
                print(f"Embeddings keys: {embeddings.keys() if isinstance(embeddings, dict) else 'Not a dict'}")
            raise
        
        # Add a small delay to allow other async tasks to run if needed
        await asyncio.sleep(0)
    
    return results