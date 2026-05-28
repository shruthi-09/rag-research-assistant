"""
Test Suite — RAG System
Tests: API endpoints, response schema, chunking, eval output
"""

import json
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.vectorstore = MagicMock()
    pipeline.vectorstore.docstore._dict = {
        "doc1": MagicMock(metadata={"source": "data/attention.pdf"}),
        "doc2": MagicMock(metadata={"source": "data/bert.pdf"}),
    }
    pipeline.query.return_value = {
        "question": "What is attention?",
        "answer":   "Attention is a mechanism that allows the model to focus on relevant parts of the input.",
        "sources":  ["attention.pdf"],
    }
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    import app.api as api_module
    api_module.pipeline = mock_pipeline
    from app.api import app
    return TestClient(app, raise_server_exceptions=True)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status"  in data
        assert "model"   in data
        assert "indexed" in data

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


class TestQueryEndpoint:
    def test_query_returns_200(self, client):
        resp = client.post("/query", json={"question": "What is attention?"})
        assert resp.status_code == 200

    def test_query_response_schema(self, client):
        data = client.post("/query", json={"question": "What is attention?"}).json()
        assert "question"   in data
        assert "answer"     in data
        assert "sources"    in data
        assert "latency_ms" in data

    def test_query_answer_is_string(self, client):
        data = client.post("/query", json={"question": "What is attention?"}).json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_query_sources_is_list(self, client):
        data = client.post("/query", json={"question": "What is attention?"}).json()
        assert isinstance(data["sources"], list)

    def test_empty_question_returns_400(self, client):
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == 400

    def test_latency_is_positive(self, client):
        data = client.post("/query", json={"question": "What is BERT?"}).json()
        assert data["latency_ms"] >= 0


class TestSourcesEndpoint:
    def test_sources_returns_200(self, client):
        resp = client.get("/sources")
        assert resp.status_code == 200

    def test_sources_schema(self, client):
        data = client.get("/sources").json()
        assert "sources" in data
        assert "count"   in data

    def test_sources_count_matches_list(self, client):
        data = client.get("/sources").json()
        assert data["count"] == len(data["sources"])


class TestChunking:
    def test_chunks_respect_size(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from app.rag_pipeline import CHUNK_SIZE, CHUNK_OVERLAP
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        text   = "word " * 1000
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(chunk) <= CHUNK_SIZE * 1.1

    def test_no_empty_chunks(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from app.rag_pipeline import CHUNK_SIZE, CHUNK_OVERLAP
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        text   = "This is a test sentence. " * 50
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(chunk.strip()) > 0


class TestEvalOutput:
    def test_eval_json_schema(self, tmp_path):
        result = {
            "timestamp":     "2025-01-01T00:00:00Z",
            "model":         "llama3-8b-8192",
            "embedding":     "sentence-transformers/all-MiniLM-L6-v2",
            "num_questions": 5,
            "scores": {
                "faithfulness":      0.87,
                "answer_relevancy":  0.91,
                "context_precision": 0.85,
                "context_recall":    0.83,
                "overall_score":     0.865,
            },
        }
        out = tmp_path / "eval_results.json"
        out.write_text(json.dumps(result))
        loaded = json.loads(out.read_text())
        assert "scores"        in loaded
        assert "overall_score" in loaded["scores"]
        assert 0 <= loaded["scores"]["overall_score"] <= 1