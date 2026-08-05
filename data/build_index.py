"""
Build FAISS Index - तिम्रो document collection लाई searchable बनाउने
Run this BEFORE starting the API
"""

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from app.config import config
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index_from_documents():
    """
    Step-by-step index building:
    1. Load documents (txt, pdf, etc.)
    2. Split into chunks (for better retrieval)
    3. Create embeddings
    4. Build FAISS index
    5. Save locally
    """

    # Step 1: Load documents
    # You can put .txt files in data/documents/
    docs_dir = config.DATA_DIR / "documents"

    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        logger.info("Creating sample document...")
        docs_dir.mkdir(parents=True)

        # Create sample document about AI
        sample_doc = docs_dir / "sample_ai.txt"
        sample_doc.write_text("""
        Artificial Intelligence (AI) is intelligence demonstrated by machines.
        Machine Learning is a subset of AI where systems learn from data.
        Deep Learning uses neural networks with many layers.
        Natural Language Processing (NLP) helps AI understand human language.
        """)

    # Load all text files
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")

    # Step 2: Split into chunks
    # Why split? Documents can be very long
    # Better to retrieve smaller, relevant chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 500 characters per chunk
        chunk_overlap=50,  # Overlap to maintain context
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")

    # Step 3: Create embeddings and index
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

    logger.info("Building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Step 4: Save index
    index_path = config.VECTOR_STORE_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))

    logger.info(f"Index saved to {index_path}")
    logger.info(f"Total vectors: {vector_store.index.ntotal}")

    # Step 5: Test retrieval
    test_query = "What is AI?"
    results = vector_store.similarity_search(test_query, k=2)

    logger.info(f"\nTest query: {test_query}")
    logger.info(f"Top result: {results[0].page_content[:200]}...")

    return vector_store


if __name__ == "__main__":
    build_index_from_documents()
