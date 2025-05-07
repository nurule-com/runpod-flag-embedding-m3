import base64
import io
import random
import numpy as np
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
        
    Returns:
        List of dictionaries containing the embeddings for each text
    """
    dense = []
    sparse_indices = []
    sparse_values = []
    colbert = []
    models = get_model_instance()
    model = random.choice(models)

    if not texts:
        return None

    embeddings = model.encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True
    )

    for i in range(len(texts)):
        dense.append(embeddings["dense_vecs"][i])
        colbert.append(embeddings["colbert_vecs"][i])
        sparse_weights = embeddings["lexical_weights"][i]
        sparse_index, sparse_value = process_sparse_weights(sparse_weights)
        sparse_indices.append(sparse_index)
        sparse_values.append(sparse_value)

    buf = io.BytesIO()
    np.savez_compressed(buf,
        dense=np.array(dense, dtype=np.float16),
        sparse_indices=np.array(sparse_indices, dtype=object),
        sparse_values=np.array(sparse_values, dtype=object),
        colbert=np.array(colbert, dtype=object)
    )
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded

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
        values.append(float(weight))
        indexes.append(token_id)
        
    return indexes, values