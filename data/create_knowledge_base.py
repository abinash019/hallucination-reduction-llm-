"""
Create a real knowledge base from scratch
No external downloads needed
"""

from app.config import config
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


# Create 10+ documents with diverse content
documents_content = {
    "ai_basics.txt": """
Artificial Intelligence (AI) is the simulation of human intelligence in machines.
AI systems can learn, reason, perceive, and understand language.
Key branches of AI include Machine Learning, Deep Learning, Natural Language Processing, and Computer Vision.
AI is used in healthcare, finance, transportation, and entertainment.
""",

    "machine_learning.txt": """
Machine Learning (ML) is a subset of AI where systems learn from data without explicit programming.
Supervised learning uses labeled data to train models.
Unsupervised learning finds patterns in unlabeled data.
Reinforcement learning learns through trial and error with rewards.
Examples include spam detection, recommendation systems, and fraud detection.
""",

    "deep_learning.txt": """
Deep Learning uses neural networks with multiple layers to learn hierarchical representations.
Neural networks are inspired by the human brain's structure.
Convolutional Neural Networks (CNNs) excel at image recognition.
Recurrent Neural Networks (RNNs) work well with sequential data like text and time series.
Transformers are a recent architecture powering large language models like GPT and BERT.
""",

    "nlp.txt": """
Natural Language Processing (NLP) helps computers understand, interpret, and generate human language.
Key NLP tasks include tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis.
Large Language Models (LLMs) like GPT, BERT, and Llama are trained on massive text datasets.
Applications include chatbots, machine translation, text summarization, and question answering.
""",

    "llm_hallucination.txt": """
Hallucination in LLMs occurs when the model generates false or misleading information confidently.
Common causes include insufficient training data, ambiguous prompts, and the model's tendency to guess.
Reduction techniques include Retrieval-Augmented Generation (RAG), prompt engineering, and verification.
RAG grounds the model's answers in retrieved documents, reducing hallucinations significantly.
Verification using NLI (Natural Language Inference) can detect contradictions with source material.
""",

    "rag_systems.txt": """
Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.
The retriever fetches relevant documents from a knowledge base using embeddings and similarity search.
The generator uses these documents as context to produce grounded answers.
RAG reduces hallucinations, improves factuality, and allows updating knowledge without retraining.
Applications include chatbots, enterprise search, and medical question answering.
""",

    "computer_vision.txt": """
Computer Vision enables machines to interpret and understand visual information from images and videos.
Key tasks include image classification, object detection, segmentation, and facial recognition.
Convolutional Neural Networks (CNNs) are the foundation of modern computer vision systems.
Applications include autonomous vehicles, medical imaging, security systems, and augmented reality.
""",

    "transformers.txt": """
Transformers are a neural network architecture introduced in the paper 'Attention Is All You Need'.
The self-attention mechanism allows the model to weigh the importance of different words in a sequence.
Unlike RNNs, transformers process all tokens in parallel, making them much faster to train.
BERT (Bidirectional Encoder Representations from Transformers) excels at understanding tasks.
GPT (Generative Pre-trained Transformer) excels at generation tasks.
""",

    "ethics_ai.txt:": """
AI ethics addresses the moral implications of artificial intelligence systems.
Key concerns include bias, fairness, transparency, accountability, and privacy.
Bias can be introduced through unbalanced training data or flawed algorithms.
Explainable AI (XAI) aims to make model decisions understandable to humans.
Responsible AI development requires diverse teams, rigorous testing, and ongoing monitoring.
""",

    "future_ai.txt": """
The future of AI includes artificial general intelligence (AGI) — systems that match human reasoning.
Multimodal models combine text, image, audio, and video understanding.
Edge AI runs models on devices rather than cloud servers for privacy and speed.
AI alignment research focuses on ensuring AI systems pursue human-compatible goals.
Regulatory frameworks for AI safety and ethics are being developed globally.
"""
}


def build_knowledge_base():
    """Create documents and build FAISS index"""

    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Write all documents
    for filename, content in documents_content.items():
        filepath = docs_dir / filename
        filepath.write_text(content)
        print(f"✓ Created: {filename}")

    # Load all documents
    from langchain.document_loaders import DirectoryLoader
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    print(f"\n📄 Loaded {len(documents)} documents")
    print(f"🔪 Split into {len(chunks)} chunks")

    # Create embeddings and index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save
    index_path = Path("data/embeddings/faiss_index")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))

    print(f"✅ Index saved to {index_path}")
    print(f"📊 Total vectors: {vector_store.index.ntotal}")

    # Test retrieval
    test_query = "What is RAG?"
    results = vector_store.similarity_search(test_query, k=2)
    print(f"\n🔍 Test: '{test_query}'")
    print(f"   → Found: {results[0].page_content[:100]}...")

    return vector_store


if __name__ == "__main__":
    build_knowledge_base()
