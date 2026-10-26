# Embedding models

Model-specific serving notes. The mechanism these run on is described in
{doc}`../architecture/serving`; this page is only about what a given family
needs and how to address it.

## EmbeddingGemma 2

Both artifact formats serve **text** on `/v1/embeddings`, not chat or extraction:

| Hub repository | Family | Runtime |
|---|---|---|
| `google/embeddinggemma-2` | `embeddinggemma2` | `docie dense-embedding` / SentenceTransformers |
| `ggml-org/embeddinggemma-2-GGUF` | `embedding` | llama-server with `--embedding --pooling mean` |

Select the repository in Hub onboarding, seed its snapshot or GGUF, then deploy
the detected family. For GGUF text embeddings no `mmproj` is required. The
serving image pins llama.cpp to the v0.7.0 / b11669 commit
`620d2225cc436ab0b001e3200e590e3b286ef9ab`; older binaries may not recognize
`gemma-embedding2`. Rebuild the serving image to pick up the pin.

The native checkpoint requires SentenceTransformers >=6.1 and Transformers
>=5.19. These conflict with the analyzer stack's Transformers constraint.
The serving image therefore installs the `embeddings` extra into
`/opt/docie-embeddings`, separate from `encoders`, and selects its CLI via
`DOCIE_DENSE_EMBEDDING_EXECUTABLE`. Do not install both extras in one environment.
The extra includes `torchvision`: the checkpoint's shared processor requires it
even when only the text encoder is loaded.
For standalone native serving in a separate virtual environment:

```sh
pip install 'small-doc-ie-bench[embeddings]'
docie dense-embedding --model google/embeddinggemma-2 --port 8093
```

Native inference disables the unused vision/audio encoders and uses bfloat16
on supported CUDA hardware, otherwise float32. It never uses float16.
The API accepts a string or a batch of strings; image/audio/video objects and
token-ID inputs are not supported by this text endpoint.

### Retrieval prefixes and output dimensions

For either runtime, format query and document inputs explicitly:

```json
{
  "model": "store:your-embedding-deployment",
  "input": [
    "task: search result | query: What causes the northern lights?",
    "title: none | text: The northern lights are caused by charged particles from the sun."
  ]
}
```

Use `title: {title} | text: {content}` when a document has a title. The native
runtime also accepts `prompt_name: "SearchQuery"` or `"Document"` for an entire
batch of unformatted strings. Omit `prompt_name` for already formatted inputs;
no task prefix is chosen implicitly. For GGUF use the explicit text prefixes,
not the native runtime's `prompt_name` extension.

Native output defaults to 768 dimensions. `dimensions` accepts 128, 256, 512,
or 768 and the result is L2-normalized **after** truncation. Use the same
dimension for queries and indexed documents. `encoding_format` accepts
`"float"` or `"base64"` (little-endian float32). Native inputs exceeding the
8,192-token context, including prefixes and special tokens, are rejected rather
than silently truncated.

Sources: [Google model card](https://huggingface.co/google/embeddinggemma-2),
[GGUF conversion](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF).
