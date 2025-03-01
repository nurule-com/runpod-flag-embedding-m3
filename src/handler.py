""" 
RunPod worker for generating embeddings using BGE-M3 model.
This worker takes an array of texts and returns sparse, dense, and colbert vectors for each text.
"""

import runpod
import asyncio
import time
import os
from utils.validation import validate_input
from model.embedding_processor import process_texts

# Track metrics for concurrency adjustment
last_request_time = time.time()
request_count = 0
request_rate = 0
last_rate_update = time.time()

# Get concurrency settings from environment variables
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))  # Default to 4
MIN_CONCURRENCY = int(os.environ.get("MIN_CONCURRENCY", "1"))  # Default to 1
SCALE_UP_THRESHOLD = float(os.environ.get("SCALE_UP_THRESHOLD", "0.05"))  # Default to 0.05 req/s
SCALE_DOWN_THRESHOLD = float(os.environ.get("SCALE_DOWN_THRESHOLD", "0.0"))  # Default to 0 req/s

# Print configuration on startup
print(f"Concurrency settings: MAX={MAX_CONCURRENCY}, MIN={MIN_CONCURRENCY}")
print(f"Scaling thresholds: UP={SCALE_UP_THRESHOLD} req/s, DOWN={SCALE_DOWN_THRESHOLD} req/s")

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
    
    # Use environment variable settings for concurrency thresholds
    # More aggressive scaling to utilize our model pool
    if request_rate > SCALE_UP_THRESHOLD and current_concurrency < MAX_CONCURRENCY:
        # Request rate above threshold, increase concurrency
        return current_concurrency + 1
    elif request_rate <= SCALE_DOWN_THRESHOLD and current_concurrency > MIN_CONCURRENCY:
        # Request rate below threshold, decrease concurrency
        return current_concurrency - 1
    
    # No change needed
    return current_concurrency

# Start the serverless worker with the async handler and concurrency modifier
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier
})
