# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import torch
import numpy as np
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import os

# -------------------------
# 1) FastAPI instance + CORS
# -------------------------
app = FastAPI(title="Hybrid ALBERT Bias Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 2) Load models + artifacts
# -------------------------
SAVE_DIR = "hybrid_albert_model"

MODEL_NAME = f"{SAVE_DIR}/albert_finetuned"
ALBERT = AlbertForSequenceClassification.from_pretrained(MODEL_NAME)
ALBERT.eval()

tokenizer = AlbertTokenizer.from_pretrained(f"{SAVE_DIR}/tokenizer")

clf = joblib.load(f"{SAVE_DIR}/lr_classifier.joblib")
scaler = joblib.load(f"{SAVE_DIR}/scaler.joblib")
le = joblib.load(f"{SAVE_DIR}/label_encoder_lr.joblib")
biasTypes = joblib.load(f"{SAVE_DIR}/bias_types_finetuned.joblib")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALBERT.to(device)

vader = SentimentIntensityAnalyzer()

# -------------------------
# 3) Pydantic request model
# -------------------------
class AnalyzeRequest(BaseModel):
    text: str

# -------------------------
# 4) Utility functions
# -------------------------
def extract_sentiment_features(text_list):
    feats = []
    for t in text_list:
        tb = TextBlob(t).sentiment
        vd = vader.polarity_scores(t)
        feats.append([
            tb.polarity, tb.subjectivity,
            vd["neg"], vd["neu"], vd["pos"], vd["compound"]
        ])
    return np.array(feats, dtype=np.float32)

def albert_embedding_batch(text_list, batch_size=16):
    ALBERT.eval()
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out = ALBERT.albert(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = torch.sum(hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()
            all_vecs.append(mean_pooled)
    return np.vstack(all_vecs)

def predict_finetuned(texts: List[str]):
    ALBERT.eval()
    results = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer.encode_plus(t, add_special_tokens=True, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].to(device)
            masks = enc["attention_mask"].to(device)
            out = ALBERT(input_ids=ids, attention_mask=masks)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = biasTypes[pred_idx]
            results.append((pred_label, probs))
    return results

def predict_lr(texts: List[str]):
    sent = extract_sentiment_features(texts)
    emb = albert_embedding_batch(texts)
    X = np.concatenate([emb, sent], axis=1)
    Xs = scaler.transform(X)
    probs = clf.predict_proba(Xs)
    results = []
    for p in probs:
        idx = int(np.argmax(p))
        label = le.classes_[idx]
        results.append((label, p))
    return results

def hybrid_predict(texts: List[str]):
    ft_results = predict_finetuned(texts)
    lr_results = predict_lr(texts)
    final = []
    for t, (lbl_ft, prob_ft), (lbl_lr, prob_lr) in zip(texts, ft_results, lr_results):
        conf_ft = float(np.max(prob_ft))
        conf_lr = float(np.max(prob_lr))
        if lbl_ft == lbl_lr:
            chosen = lbl_ft
            chosen_conf = max(conf_ft, conf_lr)
        else:
            if conf_ft >= conf_lr:
                chosen = lbl_ft
                chosen_conf = conf_ft
            else:
                chosen = lbl_lr
                chosen_conf = conf_lr
        final.append({"text": t, "final_label": chosen, "final_confidence": chosen_conf})
    return final

# -------------------------
# 5) Rewrite prompt using OpenRouter API
# -------------------------
OPENROUTER_API_KEY = "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"  # <-- ADD YOUR KEY HERE

def rewrite_prompt_neutral(prompt: str) -> str:
    api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.com",
        "X-Title": "Neutral Prompt Generator",
    }

    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You rewrite user prompts into a single, neutral, general knowledge-seeking question.\n"
                    "Rules:\n"
                    "- Do NOT mention AI, chatbots, language models, tools, apps, products, companies, or brands.\n"
                    "- Remove all comparisons (better than, more than, vs, difference, compare).\n"
                    "- Remove emotional framing and personal pronouns (I, me, you, we, my, your).\n"
                    "- Remove persuasive or manipulative tone.\n"
                    "- Keep only the broad informational intent.\n"
                    "- The result must be about general concepts or factors, not about specific systems.\n"
                    "- Return ONLY the rewritten question, no explanation."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 200
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Rewrite error: {e}")
        return None

# -------------------------
# 6) API endpoints
# -------------------------
@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    text = req.text
    result = hybrid_predict([text])[0]
    return {"label": result["final_label"], "confidence": result["final_confidence"]}

@app.post("/api/analyze-and-rewrite")
def analyze_and_rewrite(req: AnalyzeRequest):
    prompt = req.text
    orig = hybrid_predict([prompt])[0]
    
    response = {
        "original_prompt": prompt,
        "original_label": orig["final_label"],
        "original_confidence": float(orig["final_confidence"]),
        "rewritten_prompt": None,
        "rewritten_label": None,
        "rewritten_confidence": None,
        "bias_removed": None
    }
    
    # Only rewrite if bias detected
    if orig["final_label"] != "neutral":
        rewritten = rewrite_prompt_neutral(prompt)
        if rewritten:
            new = hybrid_predict([rewritten])[0]
            response["rewritten_prompt"] = rewritten
            response["rewritten_label"] = new["final_label"]
            response["rewritten_confidence"] = float(new["final_confidence"])
            response["bias_removed"] = new["final_label"] == "neutral"
    
    return response

# Test that app exists
if __name__ == "__main__":
    print(f"App created successfully: {app}")