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

Documents are split into 600-token parent chunks with a 60-token overlap, then
into 200-token child chunks with a 20-token overlap using the `cl100k_base`
tokenizer. Child chunks carry a stable `parent_id` and the parent text. Hybrid
retrieval searches child chunks, collapses matches to linked parents, and uses a
cross-encoder to rerank the parent contexts. A single Gemini compression call
then removes parent chunks that are not required and extracts only the evidence
needed for the final LLM prompt. Final compressed results are also stored in a
separate semantic query cache; similar future questions within the configured
distance threshold can skip retrieval, reranking, and compression.

Run a bounded ingestion into persistent Chroma storage:

```bash
uv run python -m src.ingestion.pipeline --limit 10 --batch-size 500
```

To ingest 500,000 source documents, run the same streaming pipeline with an
explicit limit. It keeps only one embedding batch in memory and retries
transient Hugging Face rows API failures with exponential backoff:

```bash
uv run python -m src.ingestion.pipeline --limit 500000 --batch-size 500 --progress-every 20
```

The run is safe to restart against the same collection because chunk IDs are
deterministic and Chroma uses upserts. Set `HF_MAX_RETRIES`,
`HF_RETRY_BACKOFF_SECONDS`, `HF_REQUEST_TIMEOUT`, or `HF_PAGE_SIZE` in `.env`
when tuning a long-running job. Expect roughly 5,000 source-page requests;
embedding and disk capacity are the main runtime constraints.

If a run stops after a source offset, resume with `--start-offset` and reduce
`--limit` by the number of documents already processed. Set `HF_CA_BUNDLE` to
your organization’s PEM CA bundle when HTTPS is intercepted by a corporate
proxy; TLS verification remains enabled.

The default collection is `arxiv_abstracts_parent_child` in `.chroma`. Documents are
embedded with normalized `BAAI/bge-base-en-v1.5` vectors (768 dimensions). Increase
`--limit` only when you are ready to ingest more of the dataset. Set
`BGE_EMBEDDING_MODEL` before creating a new collection if you choose another model.
New collections use Chroma HNSW `M=16` and `ef=100` for a speed and recall
balance. HNSW settings are fixed when a collection is created; rebuild into a
new collection to apply these settings to an existing index.

## Query API

Start the FastAPI service:

```bash
env -u VIRTUAL_ENV uv run uvicorn src.api.app:app --reload
```

Query the hybrid retriever:

```bash
curl -X POST http://127.0.0.1:8000/query \
	-H 'Content-Type: application/json' \
	-d '{"query":"quantum chromodynamics","top_k":10}'
```

The response contains compressed parent contexts, retrieval scores, vector and
BM25 ranks, and source metadata. Interactive API documentation is available at
`/docs`.

## Guardrails

NeMo Guardrails protect both sides of `/query`. Input and generated output are
checked for PII, secrets, prompt injection, malware or credential theft,
violent wrongdoing, self-harm instructions, and sexual content involving
minors. Blocked requests return a custom safety message, `blocked: true`, and
no source contexts.

## Logging and errors

Application logs are emitted through the `researchmind` logger. Configure them with
`LOG_LEVEL=DEBUG` for detailed diagnostics or `LOG_JSON=true` for structured JSON
records. Expected failures use typed exceptions such as `DataSourceError`,
`EmbeddingError`, `VectorStoreError`, and `IngestionError`.
