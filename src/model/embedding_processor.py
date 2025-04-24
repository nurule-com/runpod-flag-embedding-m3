import random
from .model_loader import get_model

model_instance = None

def get_model_instance():
    global model_instance
    if model_instance is None:
        model_instance = get_model()
    return model_instance

def process_texts_sync(texts):
    """
    Synchronous version of process_texts for use with thread pools.
    Process a list of texts and return embeddings for each.
    
    Args:
        texts: List of text strings to encode
        batch_size: Number of texts to process at once. If 0 or negative, all texts are processed at once (default).
        
    Returns:
        List of dictionaries containing the embeddings for each text
    """
    models = get_model_instance()
    model = random.choice(models)
    results = []

    if not texts:
        return []

    results = [None] * len(texts)

    embeddings = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True
    )

    for i, text in enumerate(texts):
        text_result = {
            "text": text,
            "dense": embeddings["dense_vecs"][i].tolist()
        }

        # Process sparse embeddings
        sparse_weights = embeddings["lexical_weights"][i]
        indexes, values = process_sparse_weights(sparse_weights)
        text_result["sparse"] = {
            "indices": indexes,
            "values": values
        }

        text_result["colbert"] = embeddings["colbert_vecs"][i].tolist()

        results[i] = text_result

    return list(results)

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

    for token_id, weight in sparse_weights.items():
        token_id = int(token_id) if isinstance(token_id, str) else token_id
        values.append(float(weight.item()) if hasattr(weight, 'item') else float(weight))
        indexes.append(token_id)

    return indexes, values