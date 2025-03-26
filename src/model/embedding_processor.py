"""
Embedding processor for BGE-M3 model.
Handles processing texts and extracting embeddings.
"""

import numpy as np
import asyncio
import time
from .model_loader import get_model

def get_batch_ranges(texts, batch_size):
    """
    Generate batch ranges for processing texts.
    
    Args:
        texts: List of text strings to encode.
        batch_size: Number of texts to process at once. If 0 or negative, all texts are processed at once.
        
    Returns:
        List of tuples representing the start and end indices for each batch.
    """
    if batch_size > 0:
        return [(i, min(i + batch_size, len(texts))) for i in range(0, len(texts), batch_size)]
    else:
        return [(0, len(texts))] if texts else []

def process_batch_embeddings(model, batch_texts, is_passage, start_idx, use_batches):
    """
    Process a batch of texts and extract embeddings.
    
    Args:
        model: The model instance used for encoding.
        batch_texts: List of text strings to encode in the current batch.
        is_passage: Whether the texts are passages (True) or queries (False).
        start_idx: The starting index of the current batch.
        use_batches: Boolean indicating if batches are being used.
        
    Returns:
        List of dictionaries containing the embeddings for each text in the batch.
    """
    results = []
    
    # Use the appropriate encoding method based on text type
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
    
    for j, text in enumerate(batch_texts):
        try:
            text_result = {
                "text": text,
                "dense": embeddings["dense_vecs"][j].tolist()
            }
            
            # Handle sparse embeddings
            if "lexical_weights" in embeddings:
                sparse_weights = embeddings["lexical_weights"][j]
                if hasattr(sparse_weights, 'items'):
                    # Dictionary-like sparse weights
                    indexes = []
                    values = []
                    for token_id, weight in sparse_weights.items():
                        if isinstance(token_id, str):
                            token_id = int(token_id)
                        weight = float(weight.item()) if hasattr(weight, 'item') else float(weight)
                        indexes.append(token_id)
                        values.append(weight)
                    text_result["sparse"] = {"indices": indexes, "values": values}
                else:
                    # Handle as numpy array
                    sparse_weights = np.atleast_1d(np.array(sparse_weights))
                    nonzero_indices = np.nonzero(sparse_weights)[0]
                    nonzero_values = sparse_weights[nonzero_indices]
                    text_result["sparse"] = {
                        "indices": nonzero_indices.tolist(),
                        "values": [float(v) for v in nonzero_values.tolist()]
                    }
            else:
                print(f"Warning: No lexical_weights found for text {j} in batch {start_idx//batch_size if use_batches else 0}")
                text_result["sparse"] = {"indices": [], "values": []}
            
            if "colbert_vecs" in embeddings:
                text_result["colbert"] = embeddings["colbert_vecs"][j].tolist()
            
            results.append(text_result)
        except Exception as e:
            print(f"Error processing text {j} in batch {start_idx//batch_size if use_batches else 0}: {e}")
            print(f"Text: {text[:100]}...")
            raise
    
    return results

def process_texts_sync(texts, is_passage=False, batch_size=0):
    """
    Synchronous version of process_texts for use with thread pools.
    Process a list of texts and return sparse, dense, and colbert embeddings for each.
    
    Args:
        texts: List of text strings to encode.
        is_passage: Whether the texts are passages (True) or queries (False).
        batch_size: Number of texts to process at once. If 0 or negative, all texts are processed at once (default).
        
    Returns:
        List of dictionaries containing the embeddings for each text.
    """
    model = get_model()
    results = []
    batch_ranges = get_batch_ranges(texts, batch_size)
    
    for start_idx, end_idx in batch_ranges:
        batch_texts = texts[start_idx:end_idx]
        try:
            results.extend(process_batch_embeddings(model, batch_texts, is_passage, start_idx, batch_size > 0))
        except Exception as e:
            print(f"Error processing batch {start_idx//batch_size if batch_size > 0 else 0}: {e}")
            print(f"Batch texts: {[t[:50] + '...' for t in batch_texts]}")
            raise
        
        time.sleep(0.001)  # Small delay to prevent CPU hogging
    
    return results

async def process_texts(texts, is_passage=False, batch_size=0):
    """
    Process a list of texts and return sparse, dense, and colbert embeddings for each.
    
    Args:
        texts: List of text strings to encode.
        is_passage: Whether the texts are passages (True) or queries (False).
        batch_size: Number of texts to process at once. If 0 or negative, all texts are processed at once (default).
        
    Returns:
        List of dictionaries containing the embeddings for each text.
    """
    model = get_model()
    results = []
    batch_ranges = get_batch_ranges(texts, batch_size)
    
    for start_idx, end_idx in batch_ranges:
        batch_texts = texts[start_idx:end_idx]
        try:
            # Use asyncio.gather to process batches concurrently
            results.extend(await asyncio.gather(
                *[process_batch_embeddings(model, batch_texts, is_passage, start_idx, batch_size > 0)]
            ))
        except Exception as e:
            print(f"Error processing batch {start_idx//batch_size if batch_size > 0 else 0}: {e}")
            print(f"Batch texts: {[t[:50] + '...' for t in batch_texts]}")
            raise
        
        await asyncio.sleep(0)  # Allow other async tasks to run
    
    return results