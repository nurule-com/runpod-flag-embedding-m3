import runpod
import os
from model.embedding_processor import process_texts_sync

# Get concurrency settings from environment variables
CONCURRENCY = int(os.environ.get("CONCURRENCY", "10"))

async def handler(job):  
    job_input = job["input"]
    
    results = process_texts_sync(
        job_input["texts"]
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