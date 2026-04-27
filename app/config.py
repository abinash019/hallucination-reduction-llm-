"""
Configuration file - सबै settings एक ठाउँमा
Environment variables, model names, paths etc.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"

    # Retriever configuration
    # Small, fast, good quality
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_STORE_PATH = DATA_DIR / "embeddings" / "faiss_index"
    TOP_K_RETRIEVAL = 5  # Number of documents to retrieve

    # Generator configuration
    # Good balance of quality/speed
    GENERATOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
    # Alternative: "meta-llama/Llama-2-7b-chat-hf" (need approval)
    MAX_GENERATION_LENGTH = 300
    TEMPERATURE = 0.3  # Lower = more deterministic, less hallucination

    # Verifier configuration
    # Best NLI model for hallucination detection
    VERIFIER_MODEL = "roberta-large-mnli"
    # Alternatives: "microsoft/deberta-v3-base-mnli", "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

    # Confidence scoring
    RETRIEVAL_WEIGHT = 0.3
    VERIFICATION_WEIGHT = 0.7

    # API
    API_HOST = "0.0.0.0"
    API_PORT = 8000

    # Evaluation
    EVALUATION_DATASET = "truthful_qa"  # Best for hallucination testing
    SAMPLE_SIZE = 100  # For quick evaluation


config = Config()
