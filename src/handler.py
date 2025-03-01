""" 
RunPod worker for generating embeddings using BGE-M3 model.
This worker takes an array of texts and returns sparse, dense, and colbert vectors for each text.
"""

import runpod
import asyncio
from utils.validation import validate_input
from model.embedding_processor import process_texts

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

# Start the serverless worker with the async handler directly
runpod.serverless.start({"handler": handler})
