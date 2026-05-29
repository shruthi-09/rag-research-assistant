"""
RAG Pipeline — AI/ML Research Papers Q&A
Retrieval: FAISS + HuggingFace Embeddings
Generation: Groq (LLaMA 3)
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.1-8b-instant"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
TOP_K           = 4
FAISS_INDEX_DIR = "faiss_index"

PROMPT_TEMPLATE = """You are an expert AI/ML research assistant.
Use ONLY the context below to answer the question. If the answer is not in the context, say "I don't have enough information in the provided documents to answer this."

Context:
{context}

Question: {question}

Answer (be concise and precise):"""


class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.1,
            api_key=os.environ["GROQ_API_KEY"],
        )
        self.vectorstore = None
        self.qa_chain    = None

    def load_documents(self, data_dir: str = "data") -> List[Document]:
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        docs = loader.load()
        print(f"[Ingest] Loaded {len(docs)} pages from {data_dir}/")
        return docs

    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        print(f"[Chunk]  {len(docs)} pages → {len(chunks)} chunks")
        return chunks

    def build_vectorstore(self, chunks: List[Document]) -> None:
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(FAISS_INDEX_DIR)
        print(f"[FAISS]  Index saved to {FAISS_INDEX_DIR}/")

    def load_vectorstore(self) -> None:
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[FAISS]  Index loaded from {FAISS_INDEX_DIR}/")

    def build_chain(self) -> None:
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.qa_chain = (
            {
                "context":  retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        self.retriever = retriever
        print("[Chain]  QA chain ready.")

    def query(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            raise RuntimeError("Call build_chain() first.")
        answer  = self.qa_chain.invoke(question)
        sources = list({
            Path(doc.metadata.get("source", "unknown")).name
            for doc in self.retriever.invoke(question)
        })
        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
        }

    def setup(self, data_dir: str = "data", force_rebuild: bool = False) -> None:
        if Path(FAISS_INDEX_DIR).exists() and not force_rebuild:
            self.load_vectorstore()
        else:
            docs   = self.load_documents(data_dir)
            chunks = self.chunk_documents(docs)
            self.build_vectorstore(chunks)
        self.build_chain()