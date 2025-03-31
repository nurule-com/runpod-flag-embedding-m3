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
model_instance = None

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
    bool_stop = False
    start_time = time.time()

    model = get_model_instance()
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
        logger.error(f"Tokenization failed: {str(e)}")
        return [{"error": "Text processing failed"} for _ in texts]

    # Determine batch size if not provided
    if batch_size <= 0:
        batch_size = max(1, 512 // max(token_counts)) if token_counts else 1
    
    # Process in optimized batches
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_results = []

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

            for j, text in enumerate(batch_texts):
                """
                if (time.time() - start_time) > 1:  # Check if we should stop processing
                    logger.error(f"{RED}Stopping processing {batch_texts} {j}{RESET}")
                    bool_stop = True
                    break
                """
                
                text_result = {
                    "text": text,
                    "dense": embeddings["dense_vecs"][j].tolist()
                }

                # Process sparse embeddings
                sparse_weights = embeddings.get("lexical_weights", [None])[j]
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

                batch_results.append(text_result)

            if bool_stop:
                logger.error(f"Batch {batch_idx // batch_size} processing timed out.")
                results.extend({"error": "Processing timed out", "text": text} for text in batch_texts)
            else:
                results.extend(batch_results)

        except Exception as e:
            logger.error(f"Batch {batch_idx // batch_size} failed: {str(e)}")
            results.extend({"error": "Processing failed", "text": text} for text in batch_texts)

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
            values.append(float(weight.item()) if hasattr(weight, 'item') else float(weight))
            indexes.append(token_id)
    else:
        sparse_weights = np.array(sparse_weights) if not isinstance(sparse_weights, np.ndarray) else sparse_weights
        nonzero_indices = np.nonzero(sparse_weights)[0]
        nonzero_values = sparse_weights[nonzero_indices]
        indexes = nonzero_indices.tolist()
        values = [float(v) for v in nonzero_values.tolist()]

    return indexes, values

# Example usage
# results_sync = process_texts_sync(texts, is_passage=True, batch_size=32)
# results_async = await process_texts(texts, is_passage=True, batch_size=32)