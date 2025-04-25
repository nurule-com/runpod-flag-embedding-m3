import io
import json
import random
import sys
import time

import numpy as np
from .model_loader import get_model
from runpod import RunPodLogger

model_instance = None

logger = RunPodLogger()

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
    start_time = time.time()
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

    # Solo manejamos el primer texto
    dense = embeddings["dense_vecs"][0]
    sparse_weights = embeddings["lexical_weights"][0]
    colbert = embeddings["colbert_vecs"][0]

    sparse_indices, sparse_values = process_sparse_weights(sparse_weights)

    # Crear archivo .npz en memoria
    buf = io.BytesIO()
    np.savez(buf,
        dense=np.array(dense),
        sparse_indices=np.array(sparse_indices),
        sparse_values=np.array(sparse_values),
        colbert=np.array(colbert)
    )
    buf.seek(0)

    end_time = time.time()
    logger.info(f"Tiempo total (incluyendo generación .npz): {end_time - start_time} segundos")
    return buf  # este buffer será devuelto como respuesta

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