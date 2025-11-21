import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever

chroma_db = chromadb.PersistentClient("vector_db/youtube_qna_bot")
youtube_bot_chroma_collection = chroma_db.get_or_create_collection("youtube-summerizer")

youtube_bot_vector_store = ChromaVectorStore(
    chroma_collection=youtube_bot_chroma_collection
)
youtube_bot_storage_context = StorageContext.from_defaults(
    vector_store=youtube_bot_vector_store
)

embedding_model = HuggingFaceEmbedding()


def create_vector_index(transcript, doc_id):

    document = Document(text=transcript, doc_id=doc_id)
    nodes = SentenceSplitter(chunk_size=400, chunk_overlap=50).get_nodes_from_documents(
        documents=[document]
    )

    for i, node in enumerate(nodes):
        node.id_ = f"{doc_id}_chunk_{i}"

    youtube_bot_vector_index = VectorStoreIndex(
        nodes=nodes,
        storage_context=youtube_bot_storage_context,
        embed_model=embedding_model,
    )

    return youtube_bot_vector_index


def retrieve_docs(vector_index: VectorStoreIndex, query_prompt: str):

    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
    return retriever.retrieve(query_prompt)
