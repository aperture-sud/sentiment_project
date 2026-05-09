# Customer Sentiment Analysis & Urgency Alert System

A professional-grade NLP pipeline that performs dual-stream analysis on customer reviews: **Supervised Sentiment Classification** and **Semantic Urgency Detection**.

This system processes reviews using two distinct methods to optimize for both accuracy and flexibility:

* **Sentiment Module (Supervised):** Uses `TF-IDF` vectorisation and `LogisticRegression`. It requires `reviews.csv` training data to learn patterns (Positive, Neutral, Negative).
* **Urgency Module (Semantic):** An unsupervised system using `SentenceTransformers` (`all-MiniLM-L6-v2`). Instead of error-prone keyword matching, it calculates the **Cosine Similarity** between incoming text and predefined "Urgent Prototypes." This ensures the system understands context (e.g., "I need my money back" is flagged as urgent even without the word "urgent").



## Features
* **Semantic Urgency Triage:** Flag critical reviews based on meaning, not just keywords. No manual labeling required.
* **"Silent Mode" CLI:** Sophisticated suppression of library telemetry, progress bars, and Hugging Face warnings for a distraction-free, professional terminal experience.
* **Auto-Triage:** Automatically logs high-urgency reviews (Urgency Score > 0.5) to `urgent_triage.txt` for timely human follow-up.
* **Confidence Breakdown:** Provides respective confidence scores for Sentiment, helping to identify "borderline" reviews.

## Project Structure
```text
sentiment_project/
├── data/
│   └── reviews.csv           # Required for Sentiment training
├── models/                   # Auto-generated (sentiment_model.pkl & urgency_model.pkl)
├── src/
│   ├── train.py              # Supervised training & Urgency prototype initialization
│   └── predict.py            # Live CLI interface with silent loading & triage logging
├── urgent_triage.txt         # Auto-generated logs for urgent reviews
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation

```

## Setup, Installation and Run
```text

# 1. Clone Repository
git clone <your-repo-link>
cd sentiment_project

# 2. Install Dependencies
pip install -r requirements.txt

# If requirements.txt is missing:
# pip install pandas numpy scikit-learn joblib sentence-transformers torch fastapi uvicorn

# 3. Verify Structure
ls
# Expected: data/ models/ src/ main.py requirements.txt


# 4. Add Dataset
# Place at: data/reviews.csv

# 5. Train Models
python3 src/train.py

# 6. Run API Server
uvicorn main:app --reload

# Open in browser:
# http://127.0.0.1:8000
# http://127.0.0.1:8000/docs

# 7. Run CLI (Optional)
python3 src/predict.py

```
