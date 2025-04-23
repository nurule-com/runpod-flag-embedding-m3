import runpod
import time
import os
import concurrent.futures
import logging
from utils.validation import validate_input
from model.embedding_processor import process_texts_sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track metrics for concurrency adjustment
last_request_time = time.time()
request_count = 0
request_rate = 0
last_rate_update = time.time()

# Get concurrency settings from environment variables
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
MIN_CONCURRENCY = int(os.environ.get("MIN_CONCURRENCY", "1"))
SCALE_UP_THRESHOLD = float(os.environ.get("SCALE_UP_THRESHOLD", "0.05"))
SCALE_DOWN_THRESHOLD = float(os.environ.get("SCALE_DOWN_THRESHOLD", "0.0"))
THREAD_POOL_SIZE = int(os.environ.get("THREAD_POOL_SIZE", str(MAX_CONCURRENCY * 2)))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "10"))

# Create a thread pool executor
thread_pool = concurrent.futures.ProcessPoolExecutor(max_workers=THREAD_POOL_SIZE)

logger.info(f"Concurrency settings: MAX={MAX_CONCURRENCY}, MIN={MIN_CONCURRENCY}")
logger.info(f"Scaling thresholds: UP={SCALE_UP_THRESHOLD} req/s, DOWN={SCALE_DOWN_THRESHOLD} req/s")
logger.info(f"Thread pool size: {THREAD_POOL_SIZE}")

async def handler(job):
    global request_count, last_request_time
    request_count += 1
    last_request_time = time.time()
    
    job_input = job["input"]
    
    # Validate input
    is_valid, result = validate_input(job_input)
    if not is_valid:
        return result
    
    if result["empty"]:
        return {"results": []}

    results = process_texts_sync(
        result["texts"],
        result["is_passage"],
        result["batch_size"]
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