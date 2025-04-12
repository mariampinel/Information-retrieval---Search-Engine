from query_processing import bm25_query, bert_query
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import numpy as np
import torch

bert_embeddings = np.load('/Users/jamesshortland/Desktop/bert_embeddings.npy')

df = pd.read_csv('/Users/jamesshortland/Desktop/preprocessed_genius_lyrics.csv')
lyrics = df['preprocessed_lyrics'].dropna().tolist()
tokenized_lyrics = [word_tokenize(song) for song in lyrics]
bm25 = BM25Okapi(tokenized_lyrics)

while True:
    query = input('Enter lyrics to find a song: \n')

    final_query_bm25 = bm25_query(query)
    final_query_bert = bert_query(query)

    scores = bm25.get_scores(final_query_bm25)
    df['bm25_score'] = scores

    top_n = 100  # or whatever range you want to pass into BERT
    bm25_top_df = df.sort_values(by='bm25_score', ascending=False).head(top_n)
    bm25_top_indices = bm25_top_df.index.tolist()

    bm25_top_embeddings = bert_embeddings[bm25_top_indices]

    query_tensor = torch.tensor(final_query_bert)
    bm25_top_tensor = torch.tensor(bm25_top_embeddings)

    # Cosine similarity = dot product since vectors are normalized
    similarities = torch.matmul(bm25_top_tensor, query_tensor)

    # Get top K matches
    top_k = 5
    top_scores, top_indices = torch.topk(similarities, k=top_k)

    print("\n🎧 Top Results (BERT re-ranked from BM25):")

    for score, idx in zip(top_scores, top_indices):
        real_idx = bm25_top_df.iloc[int(idx)].name
        row = df.loc[real_idx]
        print(f"\nScore: {score.item():.4f}")
        print(f"Title: {row['title']}")
        print(f'Artist: {row['artist']}')

