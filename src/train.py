import os
import logging
import pandas as pd
import joblib
import contextlib
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from s import SentenceTransformer


# --- NUCLEAR SILENT ZONE ---
# This block traps all library initialization noise
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

def train_sentiment_model(df):
    print("Training Sentiment Model...")
    # Preprocessing
    df = df.dropna(subset=['Score', 'Text'])
    
    def map_score(s):
        if s <= 2: return 'Negative'
        if s == 3: return 'Neutral'
        return 'Positive'

    labels = df['Score'].apply(map_score)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000))
    ])

    pipeline.fit(df['Text'], labels)
    joblib.dump(pipeline, 'models/sentiment_model.pkl')
    print("Sentiment model saved!")

def init_urgency_prototypes():
    print("Initializing Urgency Prototypes (Semantic Similarity)...")
    
    # Load the lightweight model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Define urgent phrases
    # This is unsupervised, so CSV data not needed
    urgent_prototypes = [
        "I need a refund immediately",
        "My order is broken or damaged",
        "The product is a scam",
        "I need human support right now",
        "My package never arrived",
        "This is urgent please help"
    ]
    
    # Encode prototypes into vectors
    embeddings = model.encode(urgent_prototypes)
    
    # Save the embeddings
    joblib.dump(embeddings, 'models/urgency_model.pkl')
    print("Urgency model initialized!")

if __name__ == "__main__":
    if not os.path.exists('models'): os.makedirs('models')
    
    # Train Sentiment (Requires CSV)
    full_df = pd.read_csv('data/reviews.csv', usecols=['Score', 'Text'])
    train_sentiment_model(full_df)
    
    # Initialize Urgency (Does NOT require CSV)
    init_urgency_prototypes()
    
    print("\nSystem ready: Sentiment model trained and Urgency prototypes initialized!")