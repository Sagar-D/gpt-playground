import json
import chromadb
from chromadb.utils import embedding_functions
from chromadb import Collection

from food_data_helper import get_formatted_food_data, get_metatdat_filters


embedder = embedding_functions.SentenceTransformerEmbeddingFunction()
client = chromadb.Client()


def create_or_get_collection(collection_name):

    if collection_name in [c.name for c in client.list_collections()]:
        return client.get_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedder,
        configuration={"hnsw": {"space": "cosine"}},
    )
    return collection


def add_documents(collection: Collection, data_list: list):

    documents = [data["document"] for data in data_list]
    metadatas = [data["metadata"] for data in data_list]
    ids = [data["id"] for data in data_list]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)


def query_docs(collection: Collection, prompt: str, enable_metadata_filter=True):

    metadata_filter = None
    if enable_metadata_filter:
        metadata_filter = get_metatdat_filters(prompt, collection)
    print(f"metadata_filter : {metadata_filter}")

    results = collection.query(query_texts=[prompt], n_results=5, where=metadata_filter)

    return results["documents"][0]


def delete_collection_if_exists(collection_name: str):

    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
