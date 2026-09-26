# ResearchMind

## arXiv metadata

The ingestion connector reads `common-pile/arxiv_abstracts` from Hugging Face's
Datasets Server rows API, avoiding a full parquet download.
The dataset contains approximately 2.54 million arXiv abstracts in the `train`
split. Streaming is enabled by default, so the complete dataset is not downloaded.

Start with a bounded connectivity check:

```bash
uv run python -m src.ingestion.connectors --limit 1
uv run python -m src.ingestion.connectors --limit 5
```

Optional settings are `HF_DATASET_ID`, `HF_DATASET_SPLIT`, `HF_DATASET_TEXT_FIELD`,
and `HF_ROWS_API_URL`. `HF_TOKEN` is loaded from `.env` and sent with each request
to improve access reliability and rate limits.

Documents are recursively chunked with a 600-token chunk size and 60-token overlap
using the `cl100k_base` tokenizer. Paper metadata is copied to every chunk, along
with its `start_index`.

Run a bounded ingestion into persistent Chroma storage:

```bash
uv run python -m src.ingestion.pipeline --limit 10 --batch-size 500
```

The default collection is `arxiv_abstracts_bge_base` in `.chroma`. Documents are
embedded with normalized `BAAI/bge-base-en-v1.5` vectors (768 dimensions). Increase
`--limit` only when you are ready to ingest more of the dataset. Set
`BGE_EMBEDDING_MODEL` before creating a new collection if you choose another model.
