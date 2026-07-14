# Task 1: News Topic Classifier Using BERT

## Objective
Fine-tune a BERT transformer model to classify news headlines into 4 topic categories: World, Sports, Business, and Sci/Tech.

## Methodology
1. Used the AG News Dataset (120K headlines, 4 classes)
2. Tokenized with BERT tokenizer (max_length=32 for CPU efficiency)
3. Fine-tuned `bert-base-uncased` for sequence classification
4. Evaluated with accuracy, F1-score, and classification report

## Key Results
| Metric | Value |
|--------|-------|
| Test Accuracy | **90.4%** |
| Test F1-Score | **90.37%** |
| Training Samples | 3,000 |
| Epochs | 2 |

### Classification Report
```
              precision    recall  f1-score   support
       World       0.96      0.85      0.90       266
      Sports       0.94      1.00      0.97       246
    Business       0.89      0.85      0.87       246
    Sci/Tech       0.83      0.93      0.88       242
    accuracy                           0.90      1000
```

## Files
- `train.py` - Training script with data loading, tokenization, fine-tuning, and evaluation
- `app.py` - Gradio web app for live classification
- `news_topic_model/` - Saved model, tokenizer, and metrics

## How to Run
```bash
conda activate AI_ML2
cd task1_news_classifier
python train.py          # Train the model
python app.py            # Launch Gradio interface (port 7860)
```

## Skills Demonstrated
- NLP with Hugging Face Transformers
- Transfer learning & fine-tuning (BERT)
- Evaluation metrics (Accuracy, F1, Precision, Recall)
- Model deployment with Gradio
