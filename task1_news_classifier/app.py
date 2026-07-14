"""
Task 1: Gradio App for News Topic Classifier
Deploy the fine-tuned BERT model for live interaction.
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr
import numpy as np

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
MODEL_DIR = "./news_topic_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def classify_news(text):
    if not text.strip():
        return {label: 0.0 for label in LABEL_MAP.values()}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
    return {LABEL_MAP[i]: float(probs[i]) for i in range(4)}


demo = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(label="News Headline", placeholder="Enter a news headline..."),
    outputs=gr.Label(num_top_classes=4, label="Predicted Topic"),
    title="News Topic Classifier",
    description="Classify news headlines into World, Sports, Business, or Sci/Tech using fine-tuned BERT.",
    examples=[
        ["Apple launches new iPhone with advanced AI features"],
        ["Manchester United wins the Champions League final"],
        ["Stock markets rally as inflation fears ease"],
        ["Scientists discover new exoplanet orbiting nearby star"],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
