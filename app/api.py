"""
FastAPI Backend — AI/ML Research RAG System
Endpoints: /health  /query  /sources
"""

from contextlib import asynccontextmanager
from typing import List
import time
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.rag_pipeline import RAGPipeline

pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RAGPipeline()
    pipeline.setup()
    yield
    pipeline = None

app = FastAPI(
    title="AI/ML Research RAG API",
    description="Retrieval-Augmented Generation over landmark AI/ML research papers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question:   str
    answer:     str
    sources:    List[str]
    latency_ms: float

class HealthResponse(BaseModel):
    status:  str
    model:   str
    indexed: bool


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {
        "status":  "ok",
        "model":   "llama3-8b-8192 via Groq",
        "indexed": pipeline is not None and pipeline.vectorstore is not None,
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(req: QueryRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    t0     = time.perf_counter()
    result = pipeline.query(req.question)
    ms     = round((time.perf_counter() - t0) * 1000, 2)
    return {**result, "latency_ms": ms}


@app.get("/sources", tags=["RAG"])
def list_sources():
    if not pipeline or not pipeline.vectorstore:
        raise HTTPException(503, "Pipeline not ready.")
    docs    = pipeline.vectorstore.docstore._dict.values()
    sources = sorted({
        doc.metadata.get("source", "unknown").split("/")[-1]
        for doc in docs
    })
    return {"sources": sources, "count": len(sources)}