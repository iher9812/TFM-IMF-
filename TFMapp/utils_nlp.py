# utils_nlp.py
import re, html, unicodedata
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent  # TFMapp/

def load_model_hf(repo_id: str):
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=BASE_DIR / "hf_model_cache",
        allow_patterns=["*.json", "*.txt", "*.safetensors"]
    )
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSequenceClassification.from_pretrained(local_dir)
    return tokenizer, model


URL_RE = re.compile(r'(https?://\S+|www\.\S+)')
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def clean_for_model(text: str) -> str:
    if not isinstance(text, str): return ""
    t = html.unescape(text)
    t = URL_RE.sub('', t)
    t = unicodedata.normalize('NFKC', t)
    return re.sub(r'\s+', ' ', t).strip()

def limpiar_texto_dataset(texto: str) -> str:
    if not isinstance(texto, str): return ""
    t = texto.lower()
    t = URL_RE.sub('', t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def load_model(model_path="sentiment_model_full_offline"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def classify_sentiment_bert(text, tokenizer, model):
    if not isinstance(text, str) or not text.strip(): return "Neutral"
    inputs = tokenizer(text, padding=True, truncation=True, max_length=160, return_tensors="pt")
    with torch.no_grad():
        pred = torch.argmax(model(**inputs).logits, dim=1).item()
    return sentiment_map.get(pred, "Neutral")

def predict_sentiment_batch(texts, tokenizer, model, batch_size=64):
    labels = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors="pt")
        with torch.no_grad():
            preds = torch.argmax(model(**inputs).logits, dim=1).tolist()
        labels.extend([sentiment_map.get(p, "Neutral") for p in preds])
    return labels
