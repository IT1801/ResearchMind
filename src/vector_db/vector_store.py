import chromadb
chroma_client = chromadb.Client()

collection_name = "test_collection"

collection = chroma_client.get_or_create_collection(collection_name)

documents = [
    {
        "id": "doc1",
        "text": "This is the first document.",
        "metadata": {"source": "source1"}
    }
]

for doc in documents:
    collection.upsert(ids = [doc["id"]], documents = [doc["text"]], metadatas = [doc["metadata"]])

query = "Hello World"

results = collection.query(
    query_texts= [query],
    n_results=1
)

print(results)


