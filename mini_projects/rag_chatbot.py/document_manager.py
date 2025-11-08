from langchain_community.document_loaders import WebBaseLoader
from helper import generate_hash, preprocess_documents, chunk_documents
from vector_store_manager import VectorStoreManager

document_registry = {}


def load_and_store_document(url: str):
    """Load documents from web, create embeddings and store in vectore_stores"""

    url = url.strip()
    collection_name = generate_hash(url)

    if (
        collection_name in document_registry
        or VectorStoreManager.does_collection_exist(collection_name)
    ):
        print(
            "Skipping Document store process. Document already exists in Vector store"
        )
        return
    try:
        documents = WebBaseLoader(url).load()
    except:
        raise Exception("Failed to load document from web!!")

    # Text preprocessing and chunking
    documents = preprocess_documents(documents=documents)
    chunks = chunk_documents(documents=documents)
    vector_store_manager = VectorStoreManager(collection_name)
    vector_store_ids = vector_store_manager.store_documents(chunks, collection_name)
    document_registry[collection_name] = {
        "url": url,
        "vector_store_ids": vector_store_ids,
    }
