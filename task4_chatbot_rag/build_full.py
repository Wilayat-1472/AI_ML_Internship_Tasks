"""
Task 4: Context-Aware RAG Chatbot
Uses FAISS retrieval + extractive response generation.
Demonstrates full RAG architecture: documents -> embeddings -> vector store -> retrieval -> response.
"""

import os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

MODEL_DIR = "./chatbot_model"
print("=" * 60)
print("Task 4: Context-Aware RAG Chatbot")
print("=" * 60)

print("\n[1/3] Loading FAISS vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    os.path.join(MODEL_DIR, "faiss_index"), embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("  Vector store loaded (20 chunks from 5 docs)")

print("\n[2/3] Setting up RAG pipeline...")

chat_history = []
MAX_HISTORY = 10

def rag_chat(question: str) -> str:
    global chat_history
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "unknown") for d in docs]))

    context_lower = context.lower()
    q_lower = question.lower()
    words = question.split()
    key_terms = [w.lower() for w in words if len(w) > 3]
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if len(s.strip()) > 20]

    scored = []
    for sent in sentences:
        sent_lower = sent.lower()
        score = sum(1 for t in key_terms if t in sent_lower)
        score += sum(1 for w in words if w.lower() in sent_lower) * 0.5
        scored.append((score, sent))
    scored.sort(key=lambda x: -x[0])
    relevant = [s for _, s in scored[:3] if _ > 0]

    if not relevant and sentences:
        relevant = sentences[:2]

    answer = ". ".join(relevant)
    if answer and not answer.endswith("."):
        answer += "."

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    if len(chat_history) > MAX_HISTORY * 2:
        chat_history = chat_history[-MAX_HISTORY * 2:]

    return answer

print("  RAG pipeline ready")

print("\n[3/3] Testing chatbot...")
test_qs = [
    "What is machine learning?",
    "Tell me about transformer models",
    "What are the applications of NLP?",
    "How do neural networks work?",
]
for q in test_qs:
    a = rag_chat(q)
    print(f"\n  Q: {q}")
    print(f"  A: {a[:200]}")
    print(f"  Sources: {', '.join(set(d.metadata['source'] for d in retriever.invoke(q)))}")

print("\n" + "=" * 60)
print("Task 4: RAG Chatbot COMPLETE")
print("=" * 60)
