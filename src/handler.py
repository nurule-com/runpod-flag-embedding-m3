import runpod
import os
from model.model_loader import load_model
from utils.validation import validate_input
from model.embedding_processor import process_texts_sync

# Get concurrency settings from environment variables
CONCURRENCY = int(os.environ.get("CONCURRENCY", "10"))

load_model()

async def handler(job):  
    job_input = job["input"]
    
    # Validate input
    is_valid, result = validate_input(job_input)
    if not is_valid:
        return result
    
    if result["empty"]:
        return {"results": []}

    results = process_texts_sync(
        result["texts"]
    )

    return {"results": results}

def concurrency_modifier(current_concurrency):
    current_concurrency = CONCURRENCY
    return current_concurrency

# Start the serverless worker with the async handler and concurrency modifier
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier
})