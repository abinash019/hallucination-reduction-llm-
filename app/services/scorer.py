def compute_confidence(retrieval_score, verification_score):
    return 0.5 * retrieval_score + 0.5 * verification_score
