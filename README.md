# AI/ML Engineering - Advanced Internship Tasks

Three portfolio-attractive AI/ML projects completed for the DevelopersHub Corporation internship.

## Tasks Completed

### Task 1: News Topic Classifier Using BERT
Fine-tuned BERT on AG News Dataset for 4-class news classification.
- **Accuracy**: 90.4% | **F1**: 90.37%
- **Tech**: Hugging Face Transformers, BERT, Gradio
- [View Task 1](task1_news_classifier/)

### Task 3: Multimodal Housing Price Prediction
Predicted housing prices using CNN (images) + MLP (tabular data) with feature fusion.
- **MAE**: $343K | **RMSE**: $385K
- **Tech**: PyTorch, ResNet-18, Multimodal Learning
- [View Task 3](task3_housing_multimodal/)

### Task 4: Context-Aware RAG Chatbot
Built a retrieval-augmented generation chatbot with LangChain and FAISS.
- **Stack**: LangChain, FAISS, all-MiniLM-L6-v2, Gradio
- **Knowledge Base**: 5 docs, 20 chunks
- [View Task 4](task4_chatbot_rag/)

## Environment Setup
```bash
conda activate AI_ML2
```

## Quick Start
```bash
# Task 1 - Train & Launch
cd task1_news_classifier
python train.py
python app.py  # Port 7860

# Task 3 - Train
cd task3_housing_multimodal
python train.py

# Task 4 - Build & Launch
cd task4_chatbot_rag
python build.py
python app.py  # Port 7861
```

## Project Structure
```
internship_tasks/
├── README.md
├── task1_news_classifier/
│   ├── train.py
│   ├── app.py
│   ├── README.md
│   └── news_topic_model/
├── task3_housing_multimodal/
│   ├── train.py
│   ├── README.md
│   ├── multimodal_model/
│   └── housing_data/
└── task4_chatbot_rag/
    ├── build.py
    ├── app.py
    ├── README.md
    ├── chatbot_model/
    └── knowledge_base/
```
