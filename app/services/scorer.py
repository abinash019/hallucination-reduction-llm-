"""
Confidence Scorer - कति प्रतिशत विश्वास गर्ने?
Combines multiple signals into final confidence score
"""

from typing import Dict
import numpy as np
from app.config import config


class ConfidenceScorer:
    def __init__(self):
        self.retrieval_weight = config.RETRIEVAL_WEIGHT
        self.verification_weight = config.VERIFICATION_WEIGHT

    def compute_confidence(self,
                           retrieval_docs: list,
                           verification_result: Dict) -> float:
        """
        Compute final confidence score (0.0 to 1.0)

        Factors:
        1. Retrieval quality: How relevant are the retrieved docs?
           - More docs = higher chance of correct answer
           - Less than 3 docs → penalty

        2. Verification score: NLI entailment confidence

        3. Bonus/penalty based on special cases

        Returns:
            Confidence score (0.0 = no trust, 1.0 = fully trust)
        """
        # Factor 1: Retrieval confidence
        # If we retrieved 5 documents, higher confidence than 1 document
        num_docs = len(retrieval_docs)
        retrieval_confidence = min(1.0, num_docs / config.TOP_K_RETRIEVAL)

        # Factor 2: Verification confidence
        verif_label = verification_result['label']
        verif_score = verification_result['confidence']

        # Map NLI labels to confidence
        if verif_label == 'ENTAILMENT':
            verification_confidence = verif_score  # Near 0.95
        elif verif_label == 'NEUTRAL':
            verification_confidence = 0.3  # Low confidence
        else:  # CONTRADICTION
            verification_confidence = 0.05  # Very low

        # Factor 3: Penalty for long answers (more likely to hallucinate)
        # We'll implement this in pipeline.py

        # Combine with weights
        confidence = (self.retrieval_weight * retrieval_confidence +
                      self.verification_weight * verification_confidence)

        # Round to 2 decimal places
        return round(confidence, 2)

    def get_confidence_level(self, confidence: float) -> str:
        """Convert numeric confidence to human-readable level"""
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"


# Singleton instance
confidence_scorer = ConfidenceScorer()
