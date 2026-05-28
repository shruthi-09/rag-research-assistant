# 🤖 Production RAG System — AI/ML Research Assistant

[![CI](https://github.com/shruthi-09/rag-research-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/shruthi-09/rag-research-assistant/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade Retrieval-Augmented Generation (RAG) system** that answers questions grounded in landmark AI/ML research papers. Built with a full evaluation pipeline using RAGAS metrics.

---

## 📊 RAGAS Evaluation Results

| Metric | Score | Description |
|---|---|---|
| **Faithfulness** | 0.89 | Answers are grounded in retrieved context |
| **Answer Relevancy** | 0.92 | Answers directly address the question |
| **Context Precision** | 0.86 | Retrieved chunks are relevant |
| **Context Recall** | 0.84 | All necessary context is retrieved |
| **Overall Score** | **0.878** | Average across all metrics |

---

## 🏗️ System Architecture
---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/shruthi-09/rag-research-assistant.git
cd rag-research-assistant
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add API Key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at: https://console.groq.com
```

### 3. Add Research Papers
Place PDF files in the `data/` directory:
- Attention Is All You Need — https://arxiv.org/pdf/1706.03762
- BERT — https://arxiv.org/pdf/1810.04805
- RAG Paper — https://arxiv.org/pdf/2005.11401

### 4. Run the API
```bash
uvicorn app.api:app --reload --port 8000
```

### 5. Run the Frontend
```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Tests
```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## 📡 API Reference

### `GET /health`
```json
{
  "status": "ok",
  "model": "llama3-8b-8192 via Groq",
  "indexed": true
}
```

### `POST /query`
```json
{ "question": "What is the attention mechanism in transformers?" }
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Retrieval** | FAISS + MiniLM-L6-v2 |
| **Generation** | LLaMA 3 8B via Groq |
| **Orchestration** | LangChain |
| **Evaluation** | RAGAS |
| **API** | FastAPI |
| **Frontend** | Streamlit |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Testing** | Pytest |

---

*Built by [Shruthi Manukonda](https://linkedin.com/in/shruthimanukonda)*