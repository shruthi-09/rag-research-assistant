"""
RAGAS Evaluation Pipeline
Metrics: Faithfulness · Answer Relevancy · Context Precision · Context Recall
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

from app.rag_pipeline import RAGPipeline, EMBEDDING_MODEL, GROQ_MODEL

EVAL_DATASET = [
    {
        "question": "What is the self-attention mechanism in transformers?",
        "ground_truth": "Self-attention allows each position in the sequence to attend to all positions in the previous layer, computing a weighted sum of values based on query-key similarity scores.",
    },
    {
        "question": "What does BERT stand for and what is its key innovation?",
        "ground_truth": "BERT stands for Bidirectional Encoder Representations from Transformers. Its key innovation is pre-training deep bidirectional representations by jointly conditioning on both left and right context.",
    },
    {
        "question": "How does Retrieval-Augmented Generation work?",
        "ground_truth": "RAG combines a retrieval component that fetches relevant documents from a knowledge base with a generative model that uses those documents as context to produce grounded answers.",
    },
    {
        "question": "What are the main components of a transformer architecture?",
        "ground_truth": "A transformer consists of an encoder and decoder, each built from multi-head self-attention layers and position-wise feed-forward networks, with residual connections and layer normalization.",
    },
    {
        "question": "What is the purpose of positional encoding in transformers?",
        "ground_truth": "Positional encoding injects information about the relative or absolute position of tokens in the sequence, since the transformer has no built-in notion of token order.",
    },
]


def run_evaluation(pipeline: RAGPipeline, output_dir: str = "evaluation") -> Dict:
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\n[RAGAS] Running evaluation on {len(EVAL_DATASET)} questions...\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in EVAL_DATASET:
        result   = pipeline.query(item["question"])
        raw_docs = pipeline.vectorstore.similarity_search(item["question"], k=4)
        ctx      = [doc.page_content for doc in raw_docs]

        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(ctx)
        ground_truths.append(item["ground_truth"])
        print(f"  ✓ {item['question'][:60]}...")

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    ragas_llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"]
    )
    ragas_emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    scores = {
        "faithfulness":      round(float(result["faithfulness"]),      4),
        "answer_relevancy":  round(float(result["answer_relevancy"]),  4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall":    round(float(result["context_recall"]),    4),
        "overall_score":     round(sum([
            float(result["faithfulness"]),
            float(result["answer_relevancy"]),
            float(result["context_precision"]),
            float(result["context_recall"]),
        ]) / 4, 4),
    }

    output = {
        "timestamp":     datetime.utcnow().isoformat() + "Z",
        "model":         GROQ_MODEL,
        "embedding":     EMBEDDING_MODEL,
        "num_questions": len(EVAL_DATASET),
        "scores":        scores,
    }

    out_path = Path(output_dir) / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print("  RAGAS EVALUATION RESULTS")
    print(f"{'='*50}")
    for k, v in scores.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<25} {v:.4f}  {bar}")
    print(f"{'='*50}")
    print(f"  Results saved → {out_path}\n")

    return output


if __name__ == "__main__":
    rag = RAGPipeline()
    rag.setup()
    run_evaluation(rag)