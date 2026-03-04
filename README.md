<<<<<<< HEAD
# projects
Contains my work
=======
# Customer Sentiment Analysis & Urgency Alert System

A professional-grade NLP pipeline that performs dual-stream analysis on customer reviews: **Supervised Sentiment Classification** and **Semantic Urgency Detection**.

This system processes reviews using two distinct methods to optimize for both accuracy and flexibility:

* **Sentiment Module (Supervised):** Uses `TF-IDF` vectorisation and `LogisticRegression`. It requires `reviews.csv` training data to learn patterns (Positive, Neutral, Negative).
* **Urgency Module (Semantic):** An unsupervised system using `SentenceTransformers` (`all-MiniLM-L6-v2`). Instead of error-prone keyword matching, it calculates the **Cosine Similarity** between incoming text and predefined "Urgent Prototypes." This ensures the system understands context (e.g., "I need my money back" is flagged as urgent even without the word "refund").



## Features
* **Semantic Urgency Triage:** Flag critical reviews based on meaning, not just keywords. No manual labeling required.
* **"Silent Mode" CLI:** Sophisticated suppression of library telemetry, progress bars, and Hugging Face warnings for a distraction-free, professional terminal experience.
* **Auto-Triage:** Automatically logs high-urgency reviews (Urgency Score > 0.5) to `urgent_triage.txt` for timely human follow-up.
* **Confidence Breakdown:** Provides respective confidence scores for Sentiment, helping to identify "borderline" reviews.

## Project Structure
sentiment_project/
├── data/
│   └── reviews.csv           # Required for Sentiment training
├── models/                   # Auto-generated (sentiment_model.pkl & urgency_model.pkl)
├── src/
│   ├── train.py              # Supervised training & Urgency prototype initialization
│   └── predict.py            # Live CLI interface with silent loading & triage logging
├── urgent_triage.txt         # Auto-generated logs for urgent reviews
├── requirements.txt          # Python dependencies
└── README.md
>>>>>>> 5d65d13 (Complete project)
