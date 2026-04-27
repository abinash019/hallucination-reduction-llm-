"""
Retriever Service - Document खोजी प्रणाली
How it works: Convert query to vector → Search similar vectors → Return relevant docs
"""

from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from app.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieverService:
    def __init__(self):
        """
        Initialize embedding model and load vector database
        Embedding: Converts text to 384-dimensional vector
        FAISS: Facebook's similarity search library
        """
        logger.info("Initializing Retriever Service...")

        # Step 1: Load embedding model
        # This model converts any text to a mathematical vector
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Step 2: Load FAISS vector store
        # FAISS stores vectors and enables fast similarity search
        try:
            self.vector_store = FAISS.load_local(
                str(config.VECTOR_STORE_PATH),
                self.embeddings,
                allow_dangerous_deserialization=True  # Required for FAISS
            )
            logger.info(
                f"Loaded vector store with {self.vector_store.index.ntotal} documents")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            logger.info(
                "Creating empty vector store. Please run build_index.py first")
            self.vector_store = None

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """
        Retrieve most relevant documents for a query

        How similarity search works:
        1. Convert query to vector (embedding)
        2. FAISS computes cosine similarity with all document vectors
        3. Return top_k most similar documents

        Args:
            query: User's question
            top_k: Number of documents to retrieve

        Returns:
            List of Document objects with page_content and metadata
        """
        if self.vector_store is None:
            logger.warning("Vector store not initialized")
            return []

        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL

        logger.info(f"Retrieving for query: {query[:50]}...")

        # similarity_search returns list of Document objects
        # Each document contains: page_content (text), metadata (source, etc.)
        docs = self.vector_store.similarity_search(query, k=top_k)

        logger.info(f"Retrieved {len(docs)} documents")

        # Optional: Add relevance scores
        # docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)

        return docs

    def add_documents(self, documents: List[Document]):
        """
        Add new documents to the vector store
        Used when you want to expand knowledge base
        """
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        # Save updated index
        self.vector_store.save_local(str(config.VECTOR_STORE_PATH))
        logger.info(
            f"Added {len(documents)} documents, total: {self.vector_store.index.ntotal}")


# Singleton instance
retriever_service = RetrieverService()
