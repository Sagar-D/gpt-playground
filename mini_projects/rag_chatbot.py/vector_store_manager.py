from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreManager:

    VECTOR_STORE_DIRECTORY = "vector_db/rag_chatbot"

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.embedding = HuggingFaceEmbeddings()
        self.vector_store = Chroma(
            embedding_function=self.embedding,
            persist_directory=self.VECTOR_STORE_DIRECTORY,
            collection_name=self.collection_name,
        )

    def store_documents(
        self, chunks: list, collection_name: str = "default_collection"
    ):
        """Add document chunks to Vector DB"""
        vectore_store_ids = self.vector_store.add_documents(documents=chunks)
        self.vector_store.persist()
        return vectore_store_ids

    def get_retriever(self, k: int = 5):
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    @staticmethod
    def does_collection_exist(collection_name):
        temp_store = Chroma(
            persist_directory=VectorStoreManager.VECTOR_STORE_DIRECTORY,
        )
        collections = [
            collection.name for collection in temp_store._client.list_collections()
        ]
        if collection_name in collections:
            return True
        return False
