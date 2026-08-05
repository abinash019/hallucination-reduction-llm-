"""
Evaluation Script - तिम्रो system कति राम्रो छ मापन गर्ने
Compare baseline vs your system
"""

import logging
import json
import pandas as pd
from tqdm import tqdm
from app.services.retriever import retriever_service
from app.services.generator import generator_service
from app.services.pipeline import pipeline
from datasets import load_dataset
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


logging.basicConfig(level=logging.WARNING)  # Reduce verbosity


def evaluate_hallucination_rate(dataset_samples, use_verification=True):
    """
    Evaluate hallucination rate on TruthfulQA dataset

    Returns:
        Statistics about correctness and hallucination
    """
    results = []

    for sample in tqdm(dataset_samples, desc="Evaluating"):
        question = sample['question']
        correct_answer = sample['correct_answer']

        if use_verification:
            # Your full system
            result = pipeline.process_query(question, return_sources=False)
            generated_answer = result['answer']
            confidence = result['confidence']
            is_hallucinating = result['verification']['is_hallucination']
        else:
            # Baseline: LLM without verification
            docs = retriever_service.retrieve(question)
            context = "\n".join([d.page_content for d in docs])
            generated_answer = generator_service.generate_answer(
                question, docs)
            is_hallucinating = None  # Can't detect without verifier

        results.append({
            'question': question,
            'generated_answer': generated_answer,
            'correct_answer': correct_answer,
            'confidence': confidence if use_verification else None,
            'detected_hallucination': is_hallucinating
        })

    return results


def compute_metrics(results):
    """Compute evaluation metrics"""
    total = len(results)

    # For manual evaluation (you need to check)
    # Here we use verification as proxy

    detected_hallucinations = sum(
        1 for r in results if r['detected_hallucination'])
    refused_answers = sum(
        1 for r in results if "don't know" in r['generated_answer'].lower())

    metrics = {
        'total_questions': total,
        'detected_hallucination_rate': detected_hallucinations / total,
        'refusal_rate': refused_answers / total,
        'avg_confidence': sum(r['confidence'] for r in results if r['confidence']) / total
    }

    return metrics


def main():
    # Load TruthfulQA dataset
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    # Use subset for evaluation
    sample_size = 50  # Start small, increase later
    samples = dataset.select(range(min(sample_size, len(dataset))))

    print(f"Evaluating on {len(samples)} questions")

    # Run evaluation
    results = evaluate_hallucination_rate(samples, use_verification=True)
    metrics = compute_metrics(results)

    print("\n📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Questions: {metrics['total_questions']}")
    print(
        f"Hallucination Rate: {metrics['detected_hallucination_rate']*100:.1f}%")
    print(
        f"Refusal Rate (honest 'I don\'t know'): {metrics['refusal_rate']*100:.1f}%")
    print(f"Average Confidence: {metrics['avg_confidence']:.2f}")

    # Save results
    output_path = Path(__file__).parent / "results" / "evaluation_results.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'detailed_results': results[:10]  # First 10 for inspection
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
