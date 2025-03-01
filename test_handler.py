"""
Simple test script to verify that the modularized code structure is working correctly.
This script doesn't rely on the runpod module.
"""

import asyncio
from src.utils.validation import validate_input

async def test_validation():
    """Test the validation function."""
    # Test with valid input
    job_input = {
        "texts": ["This is a test query"],
        "isPassage": False,
        "batchSize": 0
    }
    is_valid, result = validate_input(job_input)
    print(f"Valid input test: {is_valid}")
    print(f"Result: {result}")
    
    # Test with missing texts field
    job_input = {
        "isPassage": False,
        "batchSize": 0
    }
    is_valid, result = validate_input(job_input)
    print(f"Missing texts test: {is_valid}")
    print(f"Result: {result}")
    
    # Test with empty texts list
    job_input = {
        "texts": [],
        "isPassage": False,
        "batchSize": 0
    }
    is_valid, result = validate_input(job_input)
    print(f"Empty texts test: {is_valid}")
    print(f"Result: {result}")
    
    # Test with non-list texts
    job_input = {
        "texts": "This is not a list",
        "isPassage": False,
        "batchSize": 0
    }
    is_valid, result = validate_input(job_input)
    print(f"Non-list texts test: {is_valid}")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_validation()) 