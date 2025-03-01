"""
Input validation utilities for the BGE-M3 embedding worker.
"""

def validate_input(job_input):
    """
    Validate the input for the embedding job.
    
    Args:
        job_input: The input dictionary from the job
        
    Returns:
        tuple: (is_valid, result_or_error)
            - If valid: (True, dict with validated parameters)
            - If invalid: (False, error message)
    """
    # Check if texts field exists
    if "texts" not in job_input:
        return False, {"error": "Missing 'texts' field in input"}
    
    texts = job_input["texts"]
    
    # Validate texts is a list
    if not isinstance(texts, list):
        return False, {"error": "The 'texts' field must be a list of strings"}
    
    # Process empty list case
    if len(texts) == 0:
        return True, {"texts": [], "is_passage": False, "batch_size": 0, "empty": True}
    
    # Check if texts are passages or queries
    is_passage = job_input.get("isPassage", False)
    
    # Get batch size (0 means no batching, which is the default)
    batch_size = job_input.get("batchSize", 0)
    
    # Return validated parameters
    return True, {
        "texts": texts,
        "is_passage": is_passage,
        "batch_size": batch_size,
        "empty": False
    } 