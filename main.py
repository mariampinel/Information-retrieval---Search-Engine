import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import time

start_time = time.time()
df = pd.read_csv('/Users/jamesshortland/Desktop/genius_lyrics_reduced.csv')

def preprocess(text):
    text = str(text).lower()
    text = text.replace("\n", " ")  # Remove newlines
    return ' '.join([word for word in text.split() if len(word) > 2])  # Simple tokenization


df['processed_lyrics'] = df['lyrics'].apply(preprocess)

# BM25 Setup
tokenized_corpus = [doc.split() for doc in df['processed_lyrics']]
bm25 = BM25Okapi(tokenized_corpus)

# BERT Setup (small model for quick testing)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
lyrics_embeddings = bert_model.encode(df['processed_lyrics'])


# Hybrid Search Function
def hybrid_search(query, bm25_weight=0.6, bert_weight=0.4, n_results=3):
    print("starting search...")
    # Preprocess query
    processed_query = preprocess(query)

    # BM25 scores
    tokenized_query = processed_query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # BERT scores
    query_embedding = bert_model.encode([processed_query])[0]
    bert_scores = lyrics_embeddings @ query_embedding.T  # Cosine similarity

    # Normalize scores
    bm25_scores_norm = MinMaxScaler().fit_transform(bm25_scores.reshape(-1, 1)).flatten()
    bert_scores_norm = MinMaxScaler().fit_transform(bert_scores.reshape(-1, 1)).flatten()

    # Combine scores
    combined_scores = (bm25_weight * bm25_scores_norm) + (bert_weight * bert_scores_norm)

    # Get top results
    top_indices = np.argsort(combined_scores)[-n_results:][::-1]
    results = df.iloc[top_indices].copy()
    results['score'] = combined_scores[top_indices]
    end_time = time.time()
    print(end_time - start_time)

    return results[['artist', 'title', 'views', 'score', 'lyrics']]


# Try it out!
print(hybrid_search("sad song"))
print("\n---\n")
print(hybrid_search("plan from god"))
print("\n---\n")
print(hybrid_search("calling you"))