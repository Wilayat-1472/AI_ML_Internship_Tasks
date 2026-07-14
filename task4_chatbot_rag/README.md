# Task 4: Context-Aware Chatbot Using LangChain/RAG

## Objective
Build a conversational chatbot that retrieves relevant information from a knowledge base using Retrieval-Augmented Generation (RAG) with LangChain and FAISS.

## Methodology
1. Created a knowledge base of 5 documents covering AI, ML, Transformers, NLP, and Python ML tools
2. Split documents into 20 chunks (500 chars each) using RecursiveCharacterTextSplitter
3. Generated embeddings with `all-MiniLM-L6-v2` sentence transformer
4. Built FAISS vector store for efficient similarity search
5. Implemented RAG pipeline: query -> retrieve top-3 chunks -> extractive response generation
6. Deployed with Gradio ChatInterface

## Architecture
```
User Query --> Embedding Model --> FAISS Vector Search --> Top-3 Chunks
                                                            |
Knowledge Base (5 docs, 20 chunks) --> FAISS Index ---------+
                                                            |
                                    Relevant Sentences --> Response + Sources
```

## Key Results
- Successfully retrieves relevant documents for AI/ML queries
- Response time: < 1 second per query
- Knowledge base: 5 topics, 20 chunks, 500 char chunks

### Test Results
| Question | Retrieved Sources |
|----------|-------------------|
| What is machine learning? | ml_techniques.txt, ai_overview.txt |
| Tell me about transformer models | transformers.txt |
| What are the applications of NLP? | ai_overview.txt, transformers.txt, nlp_applications.txt |
| How do neural networks work? | ml_techniques.txt, ai_overview.txt, transformers.txt |

## Files
- `build.py` - Creates knowledge base and FAISS vector store
- `build_full.py` - Full RAG pipeline with retrieval and response generation
- `app.py` - Gradio web app for live chat interaction
- `knowledge_base/` - Source documents (5 .txt files)
- `chatbot_model/faiss_index/` - Saved FAISS vector store

## How to Run
```bash
conda activate AI_ML2
cd task4_chatbot_rag
python build.py           # Create vector store (run once)
python app.py             # Launch Gradio chatbot (port 7861)
```

## Skills Demonstrated
- Retrieval-Augmented Generation (RAG)
- Document embedding and vector search (FAISS)
- LangChain framework integration
- Conversational AI with context memory
- Gradio deployment
