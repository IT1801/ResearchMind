# ResearchMind

## arXiv metadata

The ingestion connector streams `common-pile/arxiv_abstracts` from Hugging Face.
The dataset contains approximately 2.54 million arXiv abstracts in the `train`
split. Streaming is enabled by default, so the complete dataset is not downloaded.

Start with a bounded connectivity check:

```bash
uv run python -m src.ingestion.connectors --limit 1
uv run python -m src.ingestion.connectors --limit 5
```

Optional settings are `HF_DATASET_ID`, `HF_DATASET_SPLIT`, and
`HF_DATASET_TEXT_FIELD`. A Hugging Face token is only needed for private datasets.

Documents are recursively chunked with a 600-token chunk size and 60-token overlap
using the `cl100k_base` tokenizer. Paper metadata is copied to every chunk, along
with its `start_index`.
