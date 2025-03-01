""" 
RunPod worker for generating embeddings using BGE-M3 model.
This worker takes an array of texts and returns sparse, dense, and colbert vectors for each text.
"""

import runpod
import asyncio
import time
from utils.validation import validate_input
from model.embedding_processor import process_texts

# Track metrics for concurrency adjustment
last_request_time = time.time()
request_count = 0
request_rate = 0
last_rate_update = time.time()

async def handler(job):
    """
    Asynchronous handler function that processes incoming jobs.
    
    Expected input format:
    {
        "texts": ["text1", "text2", ...],
        "isPassage": false,  // Optional, defaults to false (query mode)
        "batchSize": 0       // Optional, defaults to 0 (no batching). Set to a positive number to enable batching.
    }
    
    Returns:
        List of dictionaries containing the embeddings for each text
    """
    # Update request metrics
    global request_count, last_request_time
    request_count += 1
    last_request_time = time.time()
    
    job_input = job["input"]
    
    # Validate input
    is_valid, result = validate_input(job_input)
    if not is_valid:
        return result
    
    # Handle empty list case
    if result["empty"]:
        return {"results": []}
    
    try:
        # Process the texts
        batch_mode = "no batching" if result["batch_size"] <= 0 else f"batch size {result['batch_size']}"
        print(f"Processing {len(result['texts'])} texts (isPassage={result['is_passage']}, {batch_mode})")
        
        results = await process_texts(
            result["texts"], 
            is_passage=result["is_passage"], 
            batch_size=result["batch_size"]
        )
        
        print(f"Successfully processed {len(results)} texts")
        
        return {"results": results}
    except Exception as e:
        return {"error": f"Error processing texts: {str(e)}"}

def concurrency_modifier(current_concurrency):
    """
    Dynamically adjusts the concurrency level based on the observed request rate.
    
    Args:
        current_concurrency: The current concurrency level
        
    Returns:
        The adjusted concurrency level
    """
    global request_count, last_rate_update, request_rate
    
    # Update request rate calculation every 10 seconds
    current_time = time.time()
    if current_time - last_rate_update >= 10:
        # Calculate requests per second
        time_diff = current_time - last_rate_update
        if time_diff > 0:
            request_rate = request_count / time_diff
        
        # Reset counters
        request_count = 0
        last_rate_update = current_time
        
        print(f"Current request rate: {request_rate:.2f} req/s, Concurrency: {current_concurrency}")
    
    # Define concurrency thresholds - match with number of model instances
    max_concurrency = 4  # Match with MAX_MODELS in model_loader.py
    min_concurrency = 1  # Minimum concurrency level
    
    # More aggressive scaling to utilize our model pool
    if request_rate > 0.05 and current_concurrency < max_concurrency:
        # Even low request rates should scale up to use available models
        return current_concurrency + 1
    elif request_rate == 0 and current_concurrency > min_concurrency:
        # Only scale down if there are no requests
        return current_concurrency - 1
    
    # No change needed
    return current_concurrency

# Start the serverless worker with the async handler and concurrency modifier
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier
})
