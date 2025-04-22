import json
import math
import multiprocessing
import runpod
import asyncio
import time
import os
import concurrent.futures
import logging
from utils.validation import validate_input
from model.embedding_processor import GREEN, RED, RESET, process_texts_sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track metrics for concurrency adjustment
last_request_time = time.time()
request_count = 0
request_rate = 0
last_rate_update = time.time()

# Get concurrency settings from environment variables
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "15"))
MIN_CONCURRENCY = int(os.environ.get("MIN_CONCURRENCY", "1"))
SCALE_UP_THRESHOLD = float(os.environ.get("SCALE_UP_THRESHOLD", "0.05"))
SCALE_DOWN_THRESHOLD = float(os.environ.get("SCALE_DOWN_THRESHOLD", "0.0"))
THREAD_POOL_SIZE = int(os.environ.get("THREAD_POOL_SIZE", str(MAX_CONCURRENCY * 2)))

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
    
    cores = 30

    # Validate input
    is_valid, result = validate_input(job_input)
    if not is_valid:
        return result
    
    if result["empty"]:
        return {"results": []}
    
    try:
        with multiprocessing.Pool(cores) as pool:
            # Split work into chunks
            args_list = [
                (result["texts"][start:end], result["is_passage"], result["batch_size"])
                for start, end in generate_chunks(len(result["texts"]), cores)
            ]

            # Process all chunks in parallel
            results = pool.starmap(process_texts_sync, args_list)

        return {"results": json.dumps(results)}
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}

def generate_chunks(total_items: int, num_chunks: int):
    """Split a list into `num_chunks` roughly equal segments."""
    chunk_size = math.ceil(total_items / num_chunks)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_items)
        yield (start, end)
        if end >= total_items:
            break

def concurrency_modifier(current_concurrency):
    global request_count, last_rate_update, request_rate
    
    current_time = time.time()
    if current_time - last_rate_update >= 10:
        time_diff = current_time - last_rate_update
        if time_diff > 0:
            request_rate = request_count / time_diff
        
        request_count = 0
        last_rate_update = current_time
        
        logger.info(f"Current request rate: {request_rate:.2f} req/s, Concurrency: {current_concurrency}")
    
    if request_rate > SCALE_UP_THRESHOLD and current_concurrency < MAX_CONCURRENCY:
        return current_concurrency + 1
    elif request_rate <= SCALE_DOWN_THRESHOLD and current_concurrency > MIN_CONCURRENCY:
        return current_concurrency - 1
    
    return current_concurrency

# Start the serverless worker with the async handler and concurrency modifier
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier
})