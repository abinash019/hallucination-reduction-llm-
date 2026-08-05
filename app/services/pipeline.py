"""
Main Pipeline - सबै services लाई जोडेर final system बनाउँछ
This is where the magic happens
"""

from app.services.retriever import retriever_service
from app.services.generator import generator_service
from app.services.verifier import verifier_service
from app.services.scorer import confidence_scorer
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HallucinationReductionPipeline:
    def __init__(self):
        self.retriever = retriever_service
        self.generator = generator_service
        self.verifier = verifier_service
        self.scorer = confidence_scorer

    def process_query(self, query: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Main entry point - process user query through all stages

        Flow:
        1. Retrieve relevant documents
        2. Generate answer from context
        3. Verify answer against context
        4. Compute confidence score
        5. Return result with sources

        Args:
            query: User's question
            return_sources: Whether to return source documents

        Returns:
            Dictionary with answer, confidence, verification, sources
        """
        logger.info(f"Processing query: {query}")

        # STAGE 1: Retrieve
        docs = self.retriever.retrieve(query)

        if not docs:
            # No documents found
            return {
                "query": query,
                "answer": "I don't have relevant information to answer this question.",
                "confidence": 0.0,
                "confidence_level": "LOW",
                "verification": {"is_hallucination": True, "label": "NO_CONTEXT"},
                "sources": []
            }

        # STAGE 2: Generate
        answer = self.generator.generate_answer(query, docs)

        # STAGE 3: Verify against context
        # Use combined context from all docs
        combined_context = "\n\n".join(
            [doc.page_content[:1000] for doc in docs])
        verification = self.verifier.verify(combined_context, answer)

        # STAGE 4: Confidence scoring
        confidence = self.scorer.compute_confidence(docs, verification)
        confidence_level = self.scorer.get_confidence_level(confidence)

        # STAGE 5: Optional - Dynamic retrieval if confidence too low
        # RESEARCH CONTRIBUTION: Verification feedback loop
        if confidence < 0.4 and not self._is_valid_refusal(answer):
            logger.info(
                f"Low confidence ({confidence}), attempting refinement...")
            return self._refinement_loop(query, docs)

        # Prepare result
        result = {
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "verification": verification,
            "sources": [
                {
                    "text": doc.page_content[:300] + "...",
                    "metadata": doc.metadata
                }
                for doc in docs[:2]  # Return top 2 sources
            ] if return_sources else None
        }

        logger.info(f"Returning result with confidence: {confidence}")
        return result

    def _is_valid_refusal(self, answer: str) -> bool:
        """Check if answer is an honest 'I don't know'"""
        refusal_phrases = [
            "don't have enough information",
            "don't know",
            "cannot answer",
            "not in the context"
        ]
        return any(phrase in answer.lower() for phrase in refusal_phrases)

    def _refinement_loop(self, query: str, original_docs: list) -> Dict:
        """
        RESEARCH INNOVATION: Dynamic refinement
        If confidence is low, try to improve
        """
        # Try with more documents
        more_docs = self.retriever.retrieve(query, top_k=10)

        # Regenerate with more context
        better_answer = self.generator.generate_answer(query, more_docs)

        # Verify again
        combined_context = "\n\n".join(
            [doc.page_content[:1000] for doc in more_docs])
        verification = self.verifier.verify(combined_context, better_answer)

        # Recompute confidence
        confidence = self.scorer.compute_confidence(more_docs, verification)

        return {
            "query": query,
            "answer": better_answer,
            "confidence": confidence,
            "confidence_level": self.scorer.get_confidence_level(confidence),
            "verification": verification,
            "refinement_performed": True,
            "sources": [{"text": doc.page_content[:300], "metadata": doc.metadata}
                        for doc in more_docs[:2]]
        }


# Singleton pipeline instance
pipeline = HallucinationReductionPipeline()
