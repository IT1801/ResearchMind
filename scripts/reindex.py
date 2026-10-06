from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pipeline import IngestionPipeline
from src.vector_store.client import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-embed documents with the configured model.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()
    if args.embedding_model:
        os.environ["BGE_EMBEDDING_MODEL"] = args.embedding_model
    os.environ["HF_START_OFFSET"] = str(args.start_offset)
    stats = IngestionPipeline(store=ChromaVectorStore(collection_name=args.collection)).run(args.limit)
    print(stats)


if __name__ == "__main__":
    main()
