from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Setup CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load Models
sentiment_model = joblib.load('models/sentiment_model.pkl')
urgent_prototypes = joblib.load('models/urgency_model.pkl')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

@app.on_event("startup")
async def startup_event():
    # Clear file only on server restart
    with open("urgent_triage.txt", "w") as f:
        f.write("--- Session Started ---\n")

class ReviewRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze(request: ReviewRequest):
    text = request.text
    
    # Logic
    s_probs = sentiment_model.predict_proba([text])[0]
    s_winner = sentiment_model.classes_[np.argmax(s_probs)]
    
    review_emb = embedder.encode([text], show_progress_bar=False)
    scores = util.cos_sim(review_emb, urgent_prototypes)
    max_score = float(scores.max())
    
    # Save if urgent
    if max_score > 0.5:
        with open("urgent_triage.txt", "a") as f:
            f.write(f"URGENT: {text}\n")
            
    return {
        "sentiment": s_winner,
        "confidence": {label: round(prob, 2) for label, prob in zip(sentiment_model.classes_, s_probs)},
        "urgency_score": round(max_score, 2),
        "is_urgent": max_score > 0.55
    }

# Serve static frontend files
app.mount("/", StaticFiles(directory="static", html=True), name="static")