"""
Verifier Service - यसले answer सही छ कि गलत verify गर्छ
This is your MAIN RESEARCH innovation
Uses NLI (Natural Language Inference) to check if answer follows from context
"""

from transformers import pipeline
from typing import Dict, Tuple
from app.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerifierService:
    def __init__(self):
        """
        Initialize NLI (Natural Language Inference) model

        NLI: Determines logical relationship between two texts:
        - ENTAILMENT: Premise (context) implies Hypothesis (answer)
        - CONTRADICTION: Premise contradicts hypothesis
        - NEUTRAL: No logical relationship

        RoBERTa-large-MNLI is fine-tuned on:
        - MultiNLI (433k examples)
        - SNLI (570k examples)
        - Specialized for contradiction detection
        """
        logger.info(f"Loading verifier model: {config.VERIFIER_MODEL}")

        self.verifier = pipeline(
            "text-classification",
            model=config.VERIFIER_MODEL,
            device=-1,  # CPU
            top_k=None  # Return all classes with scores
        )

        logger.info("Verifier model loaded successfully")

    def verify(self, context: str, answer: str) -> Dict:
        """
        Verify if answer is consistent with context

        How it works:
        1. Combine context and answer with special token </s></s>
        2. NLI model processes the pair
        3. Returns probabilities for ENTAILMENT/CONTRADICTION/NEUTRAL

        Args:
            context: Retrieved document text
            answer: Generated answer

        Returns:
            Dictionary with verification result and scores
        """
        # Step 1: Prepare input pair
        # </s></s> is standard separator for NLI models
        # Some models use [SEP] token
        input_pair = f"{context} </s></s> {answer}"

        # Truncate to model's max length (512 tokens for RoBERTa)
        # ~2000 characters
        if len(input_pair) > 2000:
            input_pair = input_pair[:2000]

        # Step 2: Get prediction
        # Returns: [{'label': 'ENTAILMENT', 'score': 0.95}, ...]
        result = self.verifier(input_pair)[0]

        logger.info(f"Verification result: {result}")

        # Step 3: Structure the output
        verification = {
            'label': result['label'],
            'confidence': result['score'],
            'is_hallucination': result['label'] == 'CONTRADICTION',
            'is_correct': result['label'] == 'ENTAILMENT',
            'all_scores': {r['label']: r['score'] for r in self.verifier(input_pair)[0]}
        }

        return verification

    def batch_verify(self, context_answer_pairs: list) -> list:
        """
        Verify multiple context-answer pairs at once
        More efficient for evaluation
        """
        inputs = [f"{ctx} </s></s> {ans}" for ctx, ans in context_answer_pairs]

        # Batch prediction
        results = self.verifier(inputs, batch_size=16)

        return results


# Singleton instance
verifier_service = VerifierService()
