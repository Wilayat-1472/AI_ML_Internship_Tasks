<p align="center">
  <img src="https://img.icons8.com/3d-fluency/94/brain.png" width="80" alt="AI/ML Logo"/>
</p>

<h1 align="center">AI/ML Engineering — Advanced Internship Tasks</h1>

<p align="center">
  <em>Three advanced AI/ML projects — transformer fine-tuning, multimodal learning, and retrieval-augmented generation<br/>submitted for the DevelopersHub Corporation AI/ML Engineering Internship (Advanced Track).</em>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://huggingface.co"><img src="https://img.shields.io/badge/🤗%20Transformers-5.13-FFD21E?style=for-the-badge" alt="Transformers"/></a>
  <a href="https://python.langchain.com"><img src="https://img.shields.io/badge/LangChain-1.3-000000?style=for-the-badge" alt="LangChain"/></a>
  <a href="https://gradio.app"><img src="https://img.shields.io/badge/Gradio-6.20-FF5733?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio"/></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-completed-tasks">Tasks</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-submission">Submission</a>
</p>

---

## 📋 Overview

This repository contains **3 advanced AI/ML projects** completed as part of the DevelopersHub Corporation AI/ML Engineering Internship (Advanced Track). Each task demonstrates cutting-edge techniques in NLP, multimodal learning, and conversational AI.

| Task | Domain | Approach | Key Metric |
| :--- | :--- | :--- | :--- |
| **Task 1** — News Topic Classifier | NLP / Classification | BERT Fine-Tuning (Hugging Face) | Accuracy **90.4%**, F1 **90.37%** |
| **Task 3** — Multimodal Housing Price Prediction | Computer Vision + Tabular | ResNet-18 CNN + MLP Feature Fusion | MAE **$343K**, RMSE **$385K** |
| **Task 4** — Context-Aware RAG Chatbot | Conversational AI | LangChain + FAISS + Sentence Embeddings | Retrieval + Response in <1s |

---

## ✅ Completed Tasks

### Task 1: News Topic Classifier Using BERT

A transformer-based text classifier that categorizes news headlines into 4 topics (World, Sports, Business, Sci/Tech) by fine-tuning BERT on the AG News dataset.

**Pipeline:**
```
AG News Dataset → Tokenization (BERT) → Fine-tune bert-base-uncased → Evaluate (Accuracy, F1, Classification Report) → Gradio Deployment
```

| Capability | Details |
| :--- | :--- |
| **Dataset** | AG News — 120K train / 7.6K test samples, 4 classes |
| **Base Model** | `bert-base-uncased` (110M parameters) |
| **Fine-Tuning** | Hugging Face `Trainer` API · 2 epochs · lr 2e-5 · batch 32 |
| **Max Sequence Length** | 32 tokens (optimized for CPU training) |
| **Evaluation** | Accuracy, Weighted F1, Precision, Recall, Classification Report |
| **Deployment** | Gradio Interface (port 7860) |

| Metric | Value |
| :--- | :---: |
| Test Accuracy | **90.4%** |
| Test F1-Score | **90.37%** |
| Macro F1 | **90%** |
| Training Time | ~10 min (CPU) |

#### Classification Report

```
              precision    recall  f1-score   support
       World       0.96      0.85      0.90       266
      Sports       0.94      1.00      0.97       246
    Business       0.89      0.85      0.87       246
    Sci/Tech       0.83      0.93      0.88       242

    accuracy                           0.90      1000
   macro avg       0.91      0.91      0.90      1000
weighted avg       0.91      0.90      0.90      1000
```

> **Key Insight:** Sports classification achieves near-perfect recall (1.00), while Sci/Tech has the highest false positive rate — headlines with scientific terms overlap with business news.

> **Script:** [`task1_news_classifier/train.py`](task1_news_classifier/train.py) | **App:** [`task1_news_classifier/app.py`](task1_news_classifier/app.py)

---

### Task 3: Multimodal Housing Price Prediction

A multimodal deep learning model that predicts housing prices by fusing visual features from house images (CNN) with structured tabular data (MLP).

**Pipeline:**
```
Synthetic Dataset Generation (2000 samples) → Image Preprocessing (ResNet-18) + Tabular Preprocessing (StandardScaler) → Feature Fusion → Joint Training → MAE / RMSE Evaluation
```

| Capability | Details |
| :--- | :--- |
| **Dataset** | Synthetic — 2,000 samples with house images (128×128) + 6 tabular features |
| **Image Encoder** | ResNet-18 (pretrained on ImageNet, fine-tuned) → 512-d feature vector |
| **Tabular Encoder** | MLP — Linear(6→64) → ReLU → BN → Dropout → Linear(64→32) → ReLU → 32-d |
| **Fusion** | Concatenation (512 + 32 = 544) → FC layers → 1-d output (price) |
| **Training** | MSE Loss · Adam optimizer · lr 1e-4 · 15 epochs · ReduceLROnPlateau |
| **Evaluation** | MAE, RMSE, R², Training Curves, Actual vs Predicted Plot |

#### Model Architecture
```
┌─────────────────┐     ┌──────────────────┐
│  House Image     │     │  Tabular Features │
│  (128×128×3)    │     │  (6 features)     │
└────────┬────────┘     └────────┬─────────┘
         │                       │
    ┌────▼────┐            ┌─────▼─────┐
    │ResNet-18│            │    MLP    │
    │ (512-d) │            │  (32-d)   │
    └────┬────┘            └─────┬─────┘
         │                       │
         └───────────┬───────────┘
              ┌──────▼──────┐
              │   Fusion    │
              │  (544→128   │
              │   →64→1)    │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │    Price    │
              └─────────────┘
```

| Metric | Value |
| :--- | :---: |
| Mean Absolute Error (MAE) | **$343,358** |
| Root Mean Squared Error (RMSE) | **$384,713** |
| Final Train Loss | **5.1** (from 167.5) |
| Total Parameters | **11.2M** (9M trainable) |

> **Key Insight:** The model learns meaningful representations — image features contribute to price prediction alongside tabular data, demonstrating effective multimodal fusion.

> **Script:** [`task3_housing_multimodal/train.py`](task3_housing_multimodal/train.py) | **Results:** [`multimodal_model/results.png`](task3_housing_multimodal/multimodal_model/results.png)

---

### Task 4: Context-Aware RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers questions about AI/ML topics by searching a vectorized knowledge base using LangChain and FAISS.

**Pipeline:**
```
Knowledge Base (5 docs) → Chunking (RecursiveCharacterTextSplitter) → Embeddings (all-MiniLM-L6-v2) → FAISS Vector Store → Similarity Search (k=3) → Extractive Response → Gradio Deployment
```

| Capability | Details |
| :--- | :--- |
| **Knowledge Base** | 5 AI/ML documents → 20 chunks (500 chars each) |
| **Embeddings** | `all-MiniLM-L6-v2` (384-d sentence embeddings) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Retrieval** | Top-3 similar chunks per query |
| **Response** | Extractive — relevant sentences from retrieved context |
| **Sources** | Transparent source attribution for every answer |
| **Deployment** | Gradio ChatInterface (port 7861) |

#### Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  User Query  │────▶│  Embedding   │────▶│  FAISS      │
│              │     │  Model       │     │  Vector     │
└─────────────┘     │  (MiniLM)    │     │  Search     │
                    └──────────────┘     │  (k=3)      │
                                         └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Knowledge   │────▶│  Document    │────▶│  Top-3      │
│  Base (5 docs)│    │  Splitter    │     │  Chunks     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                         ┌──────▼──────┐
                                         │  Response   │
                                         │  Generation │
                                         │  + Sources  │
                                         └─────────────┘
```

| Capability | Details |
| :--- | :--- |
| **Topics Covered** | AI Overview, ML Techniques, Transformers, NLP, Python ML Tools |
| **Query Time** | < 1 second per query |
| **Retrieval Accuracy** | Correct source documents retrieved for all test queries |

#### Test Results

| Question | Retrieved Sources |
| :--- | :--- |
| What is machine learning? | `ml_techniques.txt`, `ai_overview.txt` |
| Tell me about transformer models | `transformers.txt` |
| What are the applications of NLP? | `ai_overview.txt`, `transformers.txt`, `nlp_applications.txt` |
| How do neural networks work? | `ml_techniques.txt`, `ai_overview.txt`, `transformers.txt` |
| What Python libraries are used for ML? | `python_ml_tools.txt`, `ml_techniques.txt` |

> **Key Insight:** The RAG architecture ensures answers are grounded in the knowledge base, with transparent source attribution — critical for trustworthy AI applications.

> **Script:** [`task4_chatbot_rag/build.py`](task4_chatbot_rag/build.py) | **App:** [`task4_chatbot_rag/app.py`](task4_chatbot_rag/app.py)

---

## 🛠 Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.10 |
| **Data Processing** | pandas 2.3, NumPy 2.2 |
| **Machine Learning** | scikit-learn 1.7 |
| **Deep Learning** | PyTorch 2.13 (CPU) |
| **NLP / Transformers** | Hugging Face Transformers 5.13, Datasets 5.0, Tokenizers |
| **Computer Vision** | torchvision 0.28 (ResNet-18) |
| **RAG / Conversational AI** | LangChain 1.3, FAISS 1.14, Sentence Transformers |
| **Visualization** | matplotlib 3.10, seaborn 0.13 |
| **Deployment** | Gradio 6.20 (Chat Interface) |
| **Environment** | Conda (AI_ML2 environment) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Conda (recommended) or pip
- 8GB+ RAM recommended for BERT training
- **Hugging Face Account** — AG News dataset requires HF token for download

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Wilayat-1472/AI_ML_Internship_Tasks.git
cd AI_ML_Internship_Tasks

# 2. Create and activate conda environment
conda create -n AI_ML2 python=3.10 -y
conda activate AI_ML2

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets scikit-learn pandas matplotlib seaborn python-dotenv
pip install gradio streamlit langchain langchain-community langchain-huggingface
pip install sentence-transformers faiss-cpu accelerate
```

### HF Token Setup

Task 1 downloads the **AG News dataset** from Hugging Face Hub, which requires authentication.

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens) and create a token
3. Copy `.env.example` and add your token:

```bash
cp .env.example .env
```

Edit `.env`:
```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Security:** `.env` is in `.gitignore` and will never be committed. Your token stays local.

### Run Tasks

```bash
# ─── Task 1: BERT News Classifier ───
cd task1_news_classifier
python train.py          # Train the model (~10 min on CPU)
python app.py            # Launch Gradio interface → http://localhost:7860

# ─── Task 3: Multimodal Housing Prediction ───
cd ../task3_housing_multimodal
python train.py          # Generate data + train model

# ─── Task 4: RAG Chatbot ───
cd ../task4_chatbot_rag
python build.py          # Build FAISS vector store (run once)
python app.py            # Launch Gradio chatbot → http://localhost:7861
```

---

## 📁 Project Structure

```
AI_ML_Internship_Tasks/
├── README.md
├── .gitignore
│
├── task1_news_classifier/
│   ├── train.py                         # BERT fine-tuning script
│   ├── app.py                           # Gradio deployment app
│   ├── README.md
│   └── news_topic_model/
│       ├── config.json                  # Model configuration
│       ├── metrics.json                 # Accuracy & F1 scores
│       ├── tokenizer.json               # BERT tokenizer
│       └── tokenizer_config.json
│
├── task3_housing_multimodal/
│   ├── train.py                         # Full multimodal pipeline
│   ├── README.md
│   ├── multimodal_model/
│   │   ├── multimodal_model.pt          # Trained model weights
│   │   ├── metrics.json                 # MAE, RMSE, R² scores
│   │   └── results.png                  # Training curves & predictions
│   └── housing_data/
│       └── images/                      # 2000 synthetic house images
│
└── task4_chatbot_rag/
    ├── build.py                         # Vector store builder
    ├── build_full.py                    # Full RAG pipeline test
    ├── app.py                           # Gradio chatbot app
    ├── README.md
    ├── chatbot_model/
    │   └── faiss_index/
    │       ├── index.faiss              # FAISS vector index
    │       └── index.pkl                # Document metadata
    └── knowledge_base/
        ├── ai_overview.txt              # AI overview document
        ├── ml_techniques.txt            # ML algorithms reference
        ├── transformers.txt             # Transformer architecture
        ├── nlp_applications.txt         # NLP techniques & applications
        └── python_ml_tools.txt          # Python ML ecosystem
```

---

## 📊 Results Summary

| Task | Model | Metric | Value |
| :--- | :--- | :--- | :---: |
| Task 1 | BERT (fine-tuned) | Accuracy | **90.4%** |
| Task 1 | BERT (fine-tuned) | F1-Score | **90.37%** |
| Task 3 | ResNet-18 + MLP | MAE | **$343K** |
| Task 3 | ResNet-18 + MLP | RMSE | **$385K** |
| Task 4 | FAISS + MiniLM-L6 | Query Time | **< 1s** |
| Task 4 | FAISS + MiniLM-L6 | Retrieval Accuracy | **100%** (correct sources) |

---

## 📄 Submission

| Detail | Info |
| :--- | :--- |
| **Organization** | DevelopersHub Corporation |
| **Program** | AI/ML Engineering Internship (Advanced Track) |
| **Due Date** | 21st July, 2026 |
| **Tasks Completed** | 3 of 5 (Tasks 1, 3, 4) |
| **GitHub** | [Wilayat-1472/AI_ML_Internship_Tasks](https://github.com/Wilayat-1472/AI_ML_Internship_Tasks) |
| **Submitted By** | Wilayat Ali |

---

## 🔗 Related Repos

| Repository | Description |
| :--- | :--- |
| [AI_ML](https://github.com/Wilayat-1472/AI_ML) | Basic Internship Tasks (Heart Disease, Mental Health Chatbot, House Price) |
| [AI_ML_Internship_Tasks](https://github.com/Wilayat-1472/AI_ML_Internship_Tasks) | Advanced Internship Tasks (BERT, Multimodal, RAG Chatbot) ← This Repo |

---

<p align="center">
  <strong>Built with ❤️ for the DevelopersHub AI/ML Engineering Internship (Advanced Track)</strong>
</p>
