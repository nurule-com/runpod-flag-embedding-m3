# BGE-M3 Embedding Worker for RunPod

This RunPod worker generates embeddings using the BGE-M3 model from FlagEmbedding. It takes an array of texts and returns three types of vectors for each text:

1. **Dense Embeddings**: Fixed-length, continuous representations
2. **Sparse Embeddings**: High-dimensional but mostly zeros (lexical information)
3. **ColBERT Embeddings**: Token-level embeddings for fine-grained matching

## Input Format

The worker expects a JSON input with the following structure:

```json
{
  "texts": ["text1", "text2", "text3", ...],
  "isPassage": false,
  "batchSize": 0
}
```

### Parameters

- `texts`: Array of text strings to encode (required)
- `isPassage`: Boolean flag indicating whether the texts are passages/documents (optional, defaults to `false`)
  - `false`: Uses `encode_queries()` - optimized for short queries
  - `true`: Uses `encode_corpus()` - optimized for longer passages/documents
- `batchSize`: Number of texts to process in each batch (optional, defaults to `0`)
  - `0` (default): Disables batching and processes all texts at once for maximum efficiency
  - Positive values: Enables batch processing with the specified batch size
  - Smaller batch sizes use less memory but may be slower
  - Larger batch sizes are more efficient but require more memory

## Output Format

The worker returns a JSON response with the following structure:

```json
{
  "results": [
    {
      "text": "text1",
      "dense": [...],  // Dense vector representation
      "sparse": {
        "indices": [...],  // Indices of non-zero elements (integers)
        "values": [...]    // Values of non-zero elements (floats)
      },
      "colbert": [...]  // ColBERT token-level embeddings
    },
    // Results for other texts...
  ]
}
```

### Qdrant Compatibility

The sparse vector format is compatible with Qdrant's requirements for sparse vectors. The format follows Qdrant's specification:

```json
{
  "indices": [1, 3, 5, 7],  // Integer indices of non-zero elements
  "values": [0.1, 0.2, 0.3, 0.4]  // Float values of non-zero elements
}
```

This allows you to directly use the sparse vectors with Qdrant's sparse vector search capabilities without any additional transformation.

## Error Handling

If an error occurs, the worker returns a JSON response with an error message:

```json
{
  "error": "Error message"
}
```

## Implementation Details

- Multiple BGE-M3 model instances are loaded at startup to maximize GPU utilization
- **Concurrent request handling** with dynamic concurrency adjustment
- **Asynchronous handler** for improved concurrency and efficiency
- By default, all texts are processed in a single model call for maximum efficiency
- Optional batch processing can be enabled by setting a positive `batchSize` value
- Different encoding methods are used based on the text type:
  - `encode_queries()` for short queries (default)
  - `encode_corpus()` for longer passages/documents
- All three embedding types (dense, sparse, and colbert) are explicitly requested
- Robust error handling with fallbacks for missing embedding types
- Results are converted to Python lists for JSON serialization

## Performance Considerations

- The worker uses an asynchronous handler to improve concurrency
- **Multiple model instances** are loaded to better utilize available GPU memory
- The worker can handle multiple concurrent requests, each using a different model instance
- Dynamic concurrency adjustment based on request rate
- While the model inference itself is synchronous, the async implementation allows for better handling of multiple requests
- By default, all texts are processed in a single model call, which is more efficient but requires more memory
- Batch processing can be enabled by setting a positive `batchSize` value
- The batch processing includes small async yields to prevent blocking the event loop
- For very large numbers of texts, consider enabling batching or splitting into multiple API calls

## Environment Variables

The worker supports the following environment variables:

- `GPU_DEVICE`: Specifies which GPU device to use (e.g., "cuda:0", "cuda:1"). Defaults to "cuda:0" if not specified.
- `MAX_MODELS`: Number of model instances to load in the pool. Defaults to 4. Adjust based on available GPU memory.
- `MAX_CONCURRENCY`: Maximum number of concurrent requests the worker can handle. Defaults to 4 (should match MAX_MODELS).
- `MIN_CONCURRENCY`: Minimum number of concurrent requests the worker will maintain. Defaults to 1.
- `SCALE_UP_THRESHOLD`: Request rate threshold (requests per second) above which concurrency will increase. Defaults to 0.05.
- `SCALE_DOWN_THRESHOLD`: Request rate threshold (requests per second) below which concurrency will decrease. Defaults to 0.0.


## Dependencies

- RunPod SDK
- PyTorch
- Transformers
- FlagEmbedding

## Known Issues

### FlagEmbedding Cleanup Error

You may see the following error message when the worker completes a job:

```
Exception ignored in: <function AbsEmbedder.__del__ at 0x...>
Traceback (most recent call last):
  File ".../FlagEmbedding/abc/inference/AbsEmbedder.py", line 286, in __del__
  File ".../FlagEmbedding/abc/inference/AbsEmbedder.py", line 94, in stop_self_pool
TypeError: 'NoneType' object is not callable
```

This error occurs during Python's shutdown process when the `AbsEmbedder` class attempts to clean up resources that have already been garbage collected. This is a known issue with the FlagEmbedding library and does not affect the functionality or results of the worker. We've implemented a custom cleanup function using `atexit` to mitigate this issue.

## References

- [RunPod Serverless Documentation](https://docs.runpod.io/serverless/workers/handlers/overview)
- [FlagEmbedding GitHub Repository](https://github.com/FlagOpen/FlagEmbedding)
- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)
