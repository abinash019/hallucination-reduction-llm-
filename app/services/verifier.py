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
        logger.info(f"Loading verifier model: {config.VERIFIER_MODEL}")
        self.verifier = pipeline(
            "text-classification",
            model=config.VERIFIER_MODEL,
            device=-1,
            top_k=None  # returns all 3 labels
        )
        logger.info("Verifier model loaded successfully")

    def verify(self, context: str, answer: str) -> Dict:
        # Truncate smartly — guarantee answer is included
        context_truncated = context[:1500]
        answer_truncated = answer[:400]
        input_pair = f"{context_truncated} </s></s> {answer_truncated}"

        # Single model call
        all_labels = self.verifier(input_pair)[0]

        # Build scores
        scores = {r['label']: r['score'] for r in all_labels}
        best_label = max(scores, key=scores.get)

        return {
            'label': best_label,
            'confidence': scores[best_label],
            'is_hallucination': best_label == 'CONTRADICTION',
            'is_correct': best_label == 'ENTAILMENT',
            'all_scores': scores  # useful for debugging / research
        }

    def batch_verify(self, context_answer_pairs: list) -> list:
        inputs = [
            f"{ctx[:1500]} </s></s> {ans[:400]}"
            for ctx, ans in context_answer_pairs
        ]
        results = self.verifier(inputs, batch_size=16)

        return [
            {
                'label': max(labels, key=lambda x: x['score'])['label'],
                'scores': {r['label']: r['score'] for r in labels}
            }
            for labels in results
        ]


# Singleton instance
verifier_service = VerifierService()
