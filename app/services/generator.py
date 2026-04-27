"""
Generator Service - LLM ले context पढेर answer बनाउँछ
Key: Forces LLM to use only provided context
"""

from transformers import pipeline
from typing import List
from langchain.schema import Document
from app.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneratorService:
    def __init__(self):
        """
        Initialize text generation pipeline

        Pipeline: 
        - Tokenizer (convert text to numbers)
        - Model (generate next tokens)
        - Decoder (convert numbers back to text)
        """
        logger.info(f"Loading generator model: {config.GENERATOR_MODEL}")

        # text-generation pipeline handles everything
        # device=0 for GPU, device=-1 for CPU
        self.generator = pipeline(
            "text-generation",
            model=config.GENERATOR_MODEL,
            device=-1,  # -1 = CPU, 0 = GPU
            torch_dtype="auto",
            trust_remote_code=True
        )

        logger.info("Generator model loaded successfully")

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate answer using retrieved documents as context

        The prompt is CRITICAL - it forces LLM to:
        1. Use ONLY the context
        2. Admit when it doesn't know
        3. Not add external knowledge

        Args:
            query: User's question
            context_docs: Retrieved documents from retriever

        Returns:
            Generated answer string
        """
        # Step 1: Combine context documents
        # Usually keep first 2000 chars to avoid token limit
        context_text = "\n\n".join([doc.page_content[:2000]
                                   for doc in context_docs])

        # Truncate if too long (Mistral has 8192 token limit ~ 6000 words)
        if len(context_text) > 6000:
            context_text = context_text[:6000]

        # Step 2: Create prompt with strict instructions
        # This is RESEARCH-GRADE prompt engineering
        prompt = f"""You are a truthful AI assistant. You MUST follow these rules STRICTLY:

1. ONLY use information from the CONTEXT below
2. If the CONTEXT does NOT contain the answer, say "I don't have enough information to answer this"
3. Do NOT add any external knowledge or make assumptions
4. Be concise - answer in 1-2 sentences if possible

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

        logger.info(f"Generating answer for: {query[:50]}...")

        # Step 3: Generate
        # Parameters controlled by config
        response = self.generator(
            prompt,
            max_new_tokens=config.MAX_GENERATION_LENGTH,
            temperature=config.TEMPERATURE,
            do_sample=True,  # Enable sampling when temp > 0
            top_p=0.95,      # Nucleus sampling
            repetition_penalty=1.1  # Discourage repetition
        )

        # Step 4: Extract generated part
        # Response format: [{'generated_text': 'full prompt + answer'}]
        full_text = response[0]['generated_text']

        # Remove the prompt part to get only answer
        answer = full_text.split("ANSWER:")[-1].strip()

        # Step 5: Filter out "I don't know" if needed
        # This is actually GOOD - means LLM is being honest

        logger.info(f"Generated answer: {answer[:100]}...")

        return answer


# Singleton instance
generator_service = GeneratorService()
