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
  "batchSize": 8
}
```

### Parameters

- `texts`: Array of text strings to encode (required)
- `isPassage`: Boolean flag indicating whether the texts are passages/documents (optional, defaults to `false`)
  - `false`: Uses `encode_queries()` - optimized for short queries
  - `true`: Uses `encode_corpus()` - optimized for longer passages/documents
- `batchSize`: Number of texts to process in each batch (optional, defaults to `8`)
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

- The BGE-M3 model is loaded once at startup and moved to GPU
- Texts are processed in batches to optimize GPU utilization
- Different encoding methods are used based on the text type:
  - `encode_queries()` for short queries (default)
  - `encode_corpus()` for longer passages/documents
- All three embedding types (dense, sparse, and colbert) are explicitly requested
- Robust error handling with fallbacks for missing embedding types
- Results are converted to Python lists for JSON serialization

## Environment Variables

The worker supports the following environment variables:

- `GPU_DEVICE`: Specifies which GPU device to use (e.g., "cuda:0", "cuda:1"). Defaults to "cuda:0" if not specified.

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
