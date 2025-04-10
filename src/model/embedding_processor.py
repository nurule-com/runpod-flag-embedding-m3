from collections import defaultdict
import json
import math
import multiprocessing
import os
import threading
import time
import numpy as np
import asyncio
from .model_loader import get_model
from runpod import RunPodLogger

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

logger = RunPodLogger()
model_instance = None
condition = threading.Condition()
textsBeingProcessed = 0

def get_model_instance():
    global model_instance
    if model_instance is None:
        model_instance = get_model()
    return model_instance

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
    model = get_model_instance()
    results = []
    tokenizer = model.tokenizer

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
        logger.error(f"Tokenization failed: {str(e)}")
        return [{"error": "Tokenization failed", "text": text} for text in texts]

    # Determine batch size if not provided
    if batch_size <= 0:
        max_token_count = max(token_counts) if token_counts else 1
        batch_size = max(1, 512 // max_token_count) if max_token_count > 0 else 512

    # Process in optimized batches
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        if not batch_texts:
            continue
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

            # Iterate through the texts *and their corresponding embeddings* in the current batch
            for j, text in enumerate(batch_texts):
                # Check if embeddings were generated successfully for this index
                if not embeddings or "dense_vecs" not in embeddings or j >= len(embeddings["dense_vecs"]):
                    logger.error(f"Missing dense_vecs for index {j} in batch starting at {batch_idx}")
                    results.append({"error": "Embedding generation failed", "text": text})
                    continue

                text_result = {
                    "text": text,
                    # Access the embedding for the specific text using index j
                    "dense": embeddings["dense_vecs"][j].tolist()
                }

                # Process sparse embeddings for the specific text using index j
                sparse_weights = None
                if "lexical_weights" in embeddings and embeddings["lexical_weights"] is not None and j < len(embeddings["lexical_weights"]):
                    sparse_weights = embeddings["lexical_weights"][j]

                if sparse_weights is not None:
                    indexes, values = process_sparse_weights(sparse_weights)
                    text_result["sparse"] = {
                        "indices": indexes,
                        "values": values
                    }
                else:
                    text_result["sparse"] = {
                        "indices": [],
                        "values": []
                    }

                # Add colbert embeddings if available for the specific text using index j
                if "colbert_vecs" in embeddings and embeddings["colbert_vecs"] is not None and j < len(embeddings["colbert_vecs"]):
                    text_result["colbert"] = embeddings["colbert_vecs"][j].tolist()

                # Append the complete dictionary to the results list
                results.append(text_result)

        except Exception as e:
            logger.error(f"Batch starting at index {batch_idx} failed: {str(e)}")
            # Append error dicts for each text in the failed batch
            results.extend([{"error": "Batch processing failed", "text": text} for text in batch_texts])

    # Return the list of dictionaries
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
    model = get_model_instance()
    results = []
    use_batches = batch_size > 0

    # Determine batch ranges
    batch_ranges = [(i, min(i + batch_size, len(texts))) for i in range(0, len(texts), batch_size)] if use_batches else [(0, len(texts))]

    for start_idx, end_idx in batch_ranges:
        batch_texts = texts[start_idx:end_idx]
        try:
            # Use the appropriate encoding method based on text type
            embeddings = model.encode_corpus(batch_texts, return_dense=True, return_sparse=True, return_colbert_vecs=True) if is_passage else model.encode_queries(batch_texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)

            for j, text in enumerate(batch_texts):
                try:
                    text_result = {
                        "text": text,
                        "dense": embeddings["dense_vecs"][j].tolist()
                    }

                    # Handle sparse embeddings
                    if "lexical_weights" in embeddings:
                        sparse_weights = embeddings["lexical_weights"][j]
                        indexes, values = process_sparse_weights(sparse_weights)
                        text_result["sparse"] = {"indices": indexes, "values": values}
                    else:
                        text_result["sparse"] = {"indices": [], "values": []}

                    # Add colbert embeddings if available
                    if "colbert_vecs" in embeddings:
                        text_result["colbert"] = embeddings["colbert_vecs"][j].tolist()

                    results.append(text_result)

                except Exception as e:
                    # Keep detailed error messages for debugging
                    logger.error(f"Error processing text {j} in batch {start_idx // batch_size if use_batches else 0}: {e}")

        except Exception as e:
            logger.error(f"Error processing batch {start_idx // batch_size if use_batches else 0}: {e}")

        # Allow other async tasks to run
        await asyncio.sleep(0)

    return results

def process_sparse_weights(sparse_weights):
    """
    Process sparse weights to extract indices and values.
    
    Args:
        sparse_weights: The sparse weights to process.
    
    Returns:
        Tuple of lists containing indices and values.
    """
    indexes = []
    values = []

    if hasattr(sparse_weights, 'items'):
        for token_id, weight in sparse_weights.items():
            if isinstance(token_id, str):
                token_id = int(token_id)
            # Extract value safely from numpy types
            values.append(float(weight.item()) if hasattr(weight, 'item') else float(weight))
            indexes.append(token_id)
    else:
        sparse_weights = np.array(sparse_weights) if not isinstance(sparse_weights, np.ndarray) else sparse_weights
        nonzero_indices = np.nonzero(sparse_weights)[0]
        nonzero_values = sparse_weights[nonzero_indices]
        indexes = nonzero_indices.tolist()  # Already JSON-safe

        # Process values to ensure they're Python-native floats
        for d in nonzero_values:
            if isinstance(d, defaultdict):
                # Handle defaultdict values
                for weight in d.values():
                    try:
                        # Convert numpy scalars to Python floats
                        values.append(float(weight.item()) if hasattr(weight, 'item') else float(weight))
                    except (ValueError, TypeError) as e:
                        print(f"Could not convert value {weight} to float: {e}")
            else:
                # Directly handle numpy floats/ints
                try:
                    # Use .item() to extract Python scalar from numpy dtype
                    values.append(float(d.item()) if hasattr(d, 'item') else float(d))
                except (ValueError, TypeError) as e:
                    print(f"Could not convert value {d} to float: {e}")
    return indexes, values