"""
Task 4: Gradio RAG Chatbot App
Deploy the retrieval-augmented generation chatbot for live interaction.
"""

import os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

MODEL_DIR = "./chatbot_model"

print("Loading embeddings and FAISS vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    os.path.join(MODEL_DIR, "faiss_index"), embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("Vector store loaded!")

chat_history = []

def rag_chat(message, history):
    docs = retriever.invoke(message)
    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "unknown") for d in docs]))

    q_lower = message.lower()
    words = message.split()
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

    source_text = " | Sources: " + ", ".join(sources) if sources else ""
    return answer + source_text

demo = gr.ChatInterface(
    fn=rag_chat,
    title="AI/ML Knowledge Chatbot (RAG)",
    description="Ask about AI, Machine Learning, Deep Learning, NLP, and Transformers! Uses Retrieval-Augmented Generation with FAISS vector search.",
    examples=[
        "What is deep learning?",
        "Explain the transformer architecture",
        "What are the main NLP techniques?",
        "How do neural networks work?",
        "What Python libraries are used for ML?",
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
