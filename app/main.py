from fastapi import FastAPI
from services import retriever, generator, verifier, scorer

app = FastAPI()


@app.post("/ask")
def ask(query: str):
    docs = retriever.retrieve(query)
    context = " ".join([d.page_content for d in docs])

    answer = generator.generate_answer(query, context)

    verification = verifier.verify(context, answer)

    confidence = scorer.compute_confidence(0.8, 0.9)

    return {
        "answer": answer,
        "confidence": confidence,
        "verification": verification
    }
