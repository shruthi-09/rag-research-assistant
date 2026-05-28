"""
Streamlit Frontend — AI/ML Research RAG System
"""

import time
import json
import pathlib
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI/ML Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("🔬 AI/ML Research RAG")
    st.markdown("**Powered by:** LLaMA 3 via Groq")
    st.markdown("**Retrieval:** FAISS + MiniLM embeddings")
    st.markdown("**Evaluation:** RAGAS pipeline")
    st.divider()

    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"✅ API Online\n\n`{health['model']}`")
    except Exception:
        st.error("❌ API Offline — start FastAPI first")

    st.divider()

    try:
        src = requests.get(f"{API_URL}/sources", timeout=3).json()
        st.markdown(f"**📚 Indexed Documents ({src['count']})**")
        for s in src["sources"]:
            st.markdown(f"- `{s}`")
    except Exception:
        st.markdown("*Could not load sources*")

SUGGESTED = [
    "What is the self-attention mechanism in transformers?",
    "How does BERT differ from GPT?",
    "Explain Retrieval-Augmented Generation.",
    "What are the components of a transformer?",
    "What is positional encoding used for?",
]

st.title("🤖 AI/ML Research Assistant")
st.markdown(
    "Ask any question about landmark AI/ML research papers. "
    "Answers are grounded in retrieved document context — no hallucinations."
)

st.markdown("**Try a question:**")
cols = st.columns(len(SUGGESTED))
selected = None
for i, (col, q) in enumerate(zip(cols, SUGGESTED)):
    if col.button(q[:35] + "…", key=f"sug_{i}"):
        selected = q

question = st.text_input(
    "Your question",
    value=selected or "",
    placeholder="e.g. What is the attention mechanism in transformers?",
)

ask_btn = st.button("🔍 Ask", type="primary", use_container_width=True)

if ask_btn and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            st.divider()
            st.markdown("### 💬 Answer")
            st.markdown(data["answer"])

            col1, col2 = st.columns(2)
            col1.metric("⚡ Latency", f"{data['latency_ms']} ms")
            col2.metric("📄 Sources used", len(data["sources"]))

            if data["sources"]:
                with st.expander("📚 Source Documents"):
                    for s in data["sources"]:
                        st.markdown(f"- `{s}`")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")

elif ask_btn:
    st.warning("Please enter a question.")

eval_path = pathlib.Path("evaluation/eval_results.json")
if eval_path.exists():
    st.divider()
    st.markdown("### 📊 RAGAS Evaluation Metrics")
    with open(eval_path) as f:
        eval_data = json.load(f)
    scores = eval_data["scores"]
    cols = st.columns(5)
    metrics = [
        ("Faithfulness",     scores["faithfulness"],      "🎯"),
        ("Ans. Relevancy",   scores["answer_relevancy"],  "💡"),
        ("Ctx. Precision",   scores["context_precision"], "🔍"),
        ("Ctx. Recall",      scores["context_recall"],    "📖"),
        ("Overall",          scores["overall_score"],     "⭐"),
    ]
    for col, (label, val, icon) in zip(cols, metrics):
        col.metric(f"{icon} {label}", f"{val:.2%}")
    st.caption(f"Evaluated on {eval_data['num_questions']} questions · {eval_data['timestamp'][:10]}")