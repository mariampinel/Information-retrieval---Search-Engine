import pandas as pd
import numpy as np
import json
from query_processing import bm25_query, bert_query
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import torch

df = pd.read_csv('/Users/jamesshortland/Desktop/preprocessed_genius_lyrics.csv')
bert_embeddings = np.load('/Users/jamesshortland/Desktop/bert_embeddings.npy')
with open('/Users/jamesshortland/Desktop/bert_ids.json') as f:
    berts_ids = json.load(f)

id_to_embedding_row = {song_id: i for i, song_id in enumerate(berts_ids)}

def search_results(df, query):
    final_query_bm25 = bm25_query(query)
    final_query_bert = bert_query(query)

    lyrics = df['preprocessed_lyrics'].dropna().tolist()
    tokenized_lyrics = [word_tokenize(song) for song in lyrics]
    bm25 = BM25Okapi(tokenized_lyrics)

    scores = bm25.get_scores(final_query_bm25)
    df['bm25_score'] = scores

    top_n = 100  # or whatever range you want to pass into BERT
    bm25_top_df = df.sort_values(by='bm25_score', ascending=False).head(top_n)
    bm25_top_id = bm25_top_df['id'].tolist()

    embedding_indices = [id_to_embedding_row[song_id] for song_id in bm25_top_id if
                         song_id in id_to_embedding_row]

    bm25_top_embeddings = bert_embeddings[embedding_indices]

    query_tensor = torch.tensor(final_query_bert)
    bm25_top_tensor = torch.tensor(bm25_top_embeddings)

    similarities = torch.matmul(bm25_top_tensor, query_tensor)

    top_k = 5
    top_scores, top_indices = torch.topk(similarities, k=top_k)

    print("\nTop Results (BERT re-ranked from BM25):")

    for score, idx in zip(top_scores, top_indices):
        real_idx = bm25_top_df.iloc[int(idx)].name
        row = df.loc[real_idx]
        print(f"\nScore: {score.item():.4f}")
        print(f"Title: {row['title']}")
        print(f'Artist: {row['artist']}')


