"""
Task 4: Context-Aware Chatbot Using LangChain/RAG
Step 1: Create vector store only (no LLM download).
"""

import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

MODEL_DIR = "./chatbot_model"
KNOWLEDGE_DIR = "./knowledge_base"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

print("=" * 60)
print("Task 4: Context-Aware Chatbot Using LangChain/RAG")
print("=" * 60)

print("\n[1/3] Creating knowledge base documents...")

documents = {
    "ai_overview.txt": """Artificial Intelligence (AI) Overview

Artificial Intelligence is a branch of computer science that aims to create intelligent machines that can simulate human intelligence. AI encompasses several subfields:

1. Machine Learning (ML): Algorithms that learn from data without being explicitly programmed. Types include supervised learning, unsupervised learning, and reinforcement learning.

2. Deep Learning: A subset of ML using neural networks with multiple layers. Common architectures include CNNs for images, RNNs for sequences, and Transformers for NLP.

3. Natural Language Processing (NLP): Enables computers to understand, interpret, and generate human language. Key applications include translation, sentiment analysis, and chatbots.

4. Computer Vision: Allows machines to interpret and understand visual information from the world. Used in autonomous vehicles, medical imaging, and facial recognition.

5. Robotics: Combines AI with physical machines to perform tasks in the real world.

Key milestones in AI history:
- 1950: Alan Turing proposes the Turing Test
- 1956: The term "Artificial Intelligence" is coined at Dartmouth Conference
- 1997: IBM Deep Blue defeats world chess champion Garry Kasparov
- 2012: AlexNet wins ImageNet competition, sparking deep learning revolution
- 2017: Transformer architecture introduced in "Attention Is All You Need"
- 2022: ChatGPT launch marks mainstream AI adoption""",

    "ml_techniques.txt": """Machine Learning Techniques and Algorithms

Supervised Learning:
- Linear Regression: Predicts continuous values by fitting a linear relationship
- Logistic Regression: Binary classification using sigmoid function
- Decision Trees: Tree-like model of decisions based on features
- Random Forest: Ensemble of decision trees for improved accuracy
- Support Vector Machines (SVM): Finds optimal hyperplane for classification
- Neural Networks: Interconnected nodes that learn complex patterns

Unsupervised Learning:
- K-Means Clustering: Groups data into k clusters based on similarity
- DBSCAN: Density-based clustering for arbitrary-shaped clusters
- Principal Component Analysis (PCA): Dimensionality reduction technique
- Autoencoders: Neural networks for learning compressed representations

Deep Learning Architectures:
- Convolutional Neural Networks (CNNs): Specialized for image processing
- Recurrent Neural Networks (RNNs): Designed for sequential data
- Long Short-Term Memory (LSTM): RNN variant for long-range dependencies
- Generative Adversarial Networks (GANs): Two networks competing to generate realistic data
- Transformers: Self-attention mechanism for parallel processing of sequences

Evaluation Metrics:
- Accuracy: Ratio of correct predictions to total predictions
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1 Score: Harmonic mean of precision and recall
- RMSE: Root Mean Square Error for regression tasks
- AUC-ROC: Area under the Receiver Operating Characteristic curve""",

    "transformers.txt": """Transformer Models and Architecture

The Transformer, introduced in 2017, revolutionized NLP by replacing recurrence with self-attention mechanisms.

Core Components:
1. Self-Attention: Allows the model to weigh the importance of different parts of the input
2. Multi-Head Attention: Multiple attention mechanisms running in parallel
3. Positional Encoding: Adds sequence order information since transformers lack inherent order
4. Feed-Forward Networks: Dense layers applied after attention
5. Layer Normalization: Stabilizes training

Key Transformer Models:
- BERT (2018): Bidirectional encoder, excellent for understanding tasks
- GPT (2018-): Autoregressive decoder, strong at generation
- T5 (2019): Text-to-text framework for all NLP tasks
- BART (2019): Denoising autoencoder for generation tasks
- Vision Transformer (ViT): Applies transformers to image classification

Modern LLMs:
- GPT-3/4: Large autoregressive models with billions of parameters
- LLaMA: Open-source large language models from Meta
- Claude: Anthropic's constitutional AI assistant
- Gemini: Google's multimodal AI model

Fine-tuning Approaches:
- Full fine-tuning: Update all parameters on task-specific data
- LoRA: Low-Rank Adaptation for efficient fine-tuning
- Prompt Tuning: Learning continuous prompts without changing model weights
- RLHF: Reinforcement Learning from Human Alignment""",

    "nlp_applications.txt": """NLP Applications and Techniques

Text Classification:
- Sentiment Analysis: Determining emotional tone of text
- Topic Classification: Categorizing documents by subject
- Spam Detection: Identifying unwanted messages
- Intent Recognition: Understanding user goals in dialogue

Named Entity Recognition (NER):
- Identifies entities like persons, organizations, locations, dates
- Used in information extraction and knowledge graph construction

Machine Translation:
- Seq2Seq models with attention mechanism
- BLEU score evaluates translation quality
- Neural MT has largely replaced statistical approaches

Text Generation:
- Language modeling: Predicting next word in sequence
- Controllable generation: Steering output with specific attributes
- Summarization: Creating concise versions of long documents

Question Answering:
- Extractive QA: Finding answer spans in context
- Generative QA: Formulating answers from knowledge
- Open-domain QA: Answering questions from large knowledge bases

Conversational AI:
- Task-oriented dialogue: Completing specific user tasks
- Chit-chat systems: Engaging in open-ended conversation
- Retrieval-Augmented Generation (RAG): Combining retrieval with generation
- Context memory: Maintaining conversation history for coherent dialogue

Embeddings and Vector Search:
- Word embeddings: Dense vector representations of words
- Sentence embeddings: Fixed-size vectors for entire sentences
- Semantic search: Finding similar content based on meaning
- Vector databases: FAISS, Pinecone, Weaviate for efficient similarity search""",

    "python_ml_tools.txt": """Python ML/AI Ecosystem

Core Libraries:
- NumPy: Numerical computing with array operations
- Pandas: Data manipulation and analysis
- Matplotlib/Seaborn: Data visualization
- Scikit-learn: Classical ML algorithms and tools

Deep Learning Frameworks:
- PyTorch: Dynamic computation graph, preferred for research
- TensorFlow: Static graph, strong production deployment
- Keras: High-level API for rapid prototyping

NLP Libraries:
- Hugging Face Transformers: Pre-trained models for NLP tasks
- spaCy: Industrial-strength NLP library
- NLTK: Classic NLP toolkit for education

Data Processing:
- Datasets (Hugging Face): Easy dataset loading and processing
- Dask: Parallel computing for large datasets
- Apache Spark: Distributed data processing

MLOps and Deployment:
- MLflow: Experiment tracking and model management
- Docker: Containerization for reproducible environments
- FastAPI: High-performance API for model serving
- Streamlit/Gradio: Rapid web app development for ML

Vector Databases:
- FAISS: Facebook's library for efficient similarity search
- ChromaDB: Lightweight embedding database
- Pinecone: Managed vector database service
- Weaviate: Open-source vector search engine

LLM Frameworks:
- LangChain: Framework for building LLM applications
- LlamaIndex: Data framework for LLM applications
- Semantic Kernel: Microsoft's LLM orchestration SDK""",
}

for filename, content in documents.items():
    with open(os.path.join(KNOWLEDGE_DIR, filename), "w") as f:
        f.write(content)

print(f"  Created {len(documents)} knowledge base documents")

print("\n[2/3] Loading and splitting documents...")
all_docs = []
for filename, content in documents.items():
    doc = Document(page_content=content, metadata={"source": filename})
    all_docs.append(doc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "]
)
splits = text_splitter.split_documents(all_docs)
print(f"  Split {len(all_docs)} documents into {len(splits)} chunks")

print("\n[3/3] Creating embeddings and FAISS vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local(os.path.join(MODEL_DIR, "faiss_index"))
print(f"  Vector store saved to {MODEL_DIR}/faiss_index")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
test_query = "What is deep learning?"
docs = retriever.invoke(test_query)
print(f"\n  Test retrieval for '{test_query}':")
for i, doc in enumerate(docs):
    print(f"  Chunk {i+1}: {doc.page_content[:80]}... (source: {doc.metadata['source']})")

print("\n" + "=" * 60)
print("Task 4: Vector Store COMPLETE")
print("=" * 60)
