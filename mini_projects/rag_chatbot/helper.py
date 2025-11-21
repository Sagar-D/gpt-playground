import hashlib
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


def generate_hash(url: str):
    """Generate a hash value for an URL"""

    url = url.encode("utf-8")
    hash_obj = hashlib.sha256()
    hash_obj.update(url)
    return hash_obj.hexdigest()


def preprocess_documents(documents: list):
    """Preprocess text in documents"""

    for page in documents:
        page.page_content = re.sub("\n\n+", "\n\n", page.page_content)
        page.page_content = re.sub("  +", "  ", page.page_content)
    return documents


def chunk_documents(documents: list):
    """Chunk documents and return list of chunks"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " "]
    )
    return text_splitter.split_documents(documents)
