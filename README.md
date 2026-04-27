<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Hallucination Reduction System</title>

  <!-- Bootstrap CDN -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">

  <style>
    body {
      background: #0f172a;
      color: #e2e8f0;
      font-family: Arial, sans-serif;
    }
    .container {
      margin-top: 40px;
      margin-bottom: 40px;
    }
    .card {
      background: #1e293b;
      border: none;
      border-radius: 12px;
      margin-bottom: 20px;
    }
    .title {
      font-size: 2.2rem;
      font-weight: bold;
    }
    pre {
      background: #0b1220;
      padding: 15px;
      border-radius: 10px;
      color: #38bdf8;
    }
    .badge-custom {
      background: #38bdf8;
      color: #000;
      margin-right: 5px;
    }
  </style>
</head>

<body>

<div class="container">

  <div class="text-center mb-5">
    <h1 class="title">🧠 Hallucination Reduction System</h1>
    <p class="lead">
      Retrieval-Augmented Generation (RAG) + NLI Verification System for grounded AI responses
    </p>

    <span class="badge badge-custom">Python</span>
    <span class="badge badge-custom">LangChain</span>
    <span class="badge badge-custom">FAISS</span>
    <span class="badge badge-custom">HuggingFace</span>
    <span class="badge badge-custom">FastAPI</span>
  </div>

  <div class="card p-4">
    <h3>🚀 Overview</h3>
    <p>
      This system reduces hallucinations in LLMs using a 3-stage pipeline:
      Retrieval → Generation → Verification.
    </p>
  </div>

  <div class="card p-4">
    <h3>⚙️ Architecture</h3>
    <pre>
User Query
   ↓
Retriever (FAISS + Embeddings)
   ↓
Generator (LLM)
   ↓
Verifier (NLI - RoBERTa)
   ↓
Final Verified Answer
    </pre>
  </div>

  <div class="card p-4">
    <h3>🧩 Core Components</h3>

    <h5>🔎 Retriever</h5>
    <ul>
      <li>FAISS vector search</li>
      <li>HuggingFace embeddings</li>
    </ul>

    <h5>🧠 Generator</h5>
    <ul>
      <li>LLM-based response generation</li>
      <li>Strict context-only prompting</li>
    </ul>

    <h5>✅ Verifier</h5>
    <ul>
      <li>NLI (Natural Language Inference)</li>
      <li>Detects hallucination vs grounded answers</li>
    </ul>
  </div>

  <div class="card p-4">
    <h3>🛠️ Tech Stack</h3>
    <ul>
      <li>Python 3.10+</li>
      <li>LangChain</li>
      <li>FAISS</li>
      <li>Transformers (HuggingFace)</li>
      <li>PyTorch</li>
      <li>FastAPI</li>
    </ul>
  </div>

  <div class="card p-4">
    <h3>📁 Project Structure</h3>
    <pre>
app/
 ├── services/
 │    ├── retriever.py
 │    ├── generator.py
 │    ├── verifier.py
 │    └── pipeline.py
 ├── models/
 ├── utils/
 ├── main.py
 ├── config.py

data/
 ├── raw/
 ├── embeddings/
    </pre>
  </div>

  <div class="card p-4">
    <h3>🔥 Features</h3>
    <ul>
      <li>RAG-based retrieval system</li>
      <li>Hallucination detection using NLI</li>
      <li>Context-grounded LLM responses</li>
      <li>FAISS semantic search</li>
      <li>FastAPI backend</li>
    </ul>
  </div>

  <div class="card p-4">
    <h3>📊 Workflow</h3>
    <ol>
      <li>User asks question</li>
      <li>Retriever finds documents</li>
      <li>Generator creates answer</li>
      <li>Verifier validates correctness</li>
      <li>Final response returned</li>
    </ol>
  </div>

  <div class="card p-4">
    <h3>🎯 Goal</h3>
    <p>
      To eliminate hallucinations in LLMs by enforcing:
      retrieval grounding, controlled generation, and verification layer.
    </p>
  </div>

  <div class="card p-4 text-center">
    <h3>⭐ Support</h3>
    <p>If you like this project, give it a star on GitHub.</p>
  </div>

</div>

</body>
</html>