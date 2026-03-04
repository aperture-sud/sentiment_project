import os
import sys
import contextlib
import warnings
import logging
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, util

# --- NUCLEAR SILENT ZONE ---
# This block traps all library initialization noise
print("Loading... please wait.")
warnings.filterwarnings('ignore')
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

# HANDLING WARNINGS TILL NOW

def run_prediction():
    sentiment_path = 'models/sentiment_model.pkl'
    urgency_path = 'models/urgency_model.pkl'

    if not (os.path.exists(sentiment_path) and os.path.exists(urgency_path)):
        print("Error: Models not found! Run python3 src/train.py first.")
        return

    sentiment_model = joblib.load(sentiment_path)
    urgent_prototypes = joblib.load(urgency_path)

    print("--- System Ready: Sentiment + Semantic Urgency Analysis ---")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter review: ")
        if user_input.lower() == 'exit': break
        
        # Sentiment Analysis 
        s_probs = sentiment_model.predict_proba([user_input])[0]
        s_classes = sentiment_model.classes_
        s_winner = s_classes[np.argmax(s_probs)]
        
        # Semantic Urgency Analysis
        review_emb = embedder.encode([user_input], show_progress_bar=False)
        scores = util.cos_sim(review_emb, urgent_prototypes)
        max_score = float(scores.max())
        
        # Display Results
        print(f"\n--- Sentiment: {s_winner} ---")
        print("Confidence Breakdown:")
        for label, prob in zip(s_classes, s_probs):
            print(f"  {label}: {prob*100:.2f}%")
        
        print(f"\nUrgency Score: {max_score:.2f}")
        
        # Action & Logging
        if max_score > 0.5:
            print("ALERT: This review is semantically similar to an urgent request!")
            with open('urgent_triage.txt', 'a') as f:
                f.write(f"URGENT: {user_input}\n")
            print(">> Saved to urgent_triage.txt")

if __name__ == "__main__":
    run_prediction()