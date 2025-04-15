import pandas as pd
import numpy as np
import json
from query_processing import bm25_query, bert_query
from rank_bm25 import BM25Okapi
import torch
import inquirer
from inquirer.themes import BlueComposure

df = pd.read_csv('/Users/jamesshortland/Desktop/preprocessed_genius_lyrics.csv')
bert_embeddings = np.load('/Users/jamesshortland/Desktop/bert_embeddings.npy')
with open('/Users/jamesshortland/Desktop/bert_ids.json') as f:
    berts_ids = json.load(f)

id_to_embedding_row = {song_id: i for i, song_id in enumerate(berts_ids)}

def search_results(df, query):
    final_query_bm25 = bm25_query(query)
    final_query_bert = bert_query(query)

    tokenized_lyrics = df['preprocessed_lyrics'].dropna().tolist()
    bm25 = BM25Okapi(tokenized_lyrics)

    scores = bm25.get_scores(final_query_bm25)
    df['bm25_score'] = scores

    top_n = 100  
    bm25_top_df = df.sort_values(by='bm25_score', ascending=False).head(top_n)
    bm25_top_id = bm25_top_df['id'].tolist()

    embedding_indices = [id_to_embedding_row[song_id] for song_id in bm25_top_id if
                         song_id in id_to_embedding_row]

    bm25_top_embeddings = bert_embeddings[embedding_indices]

    query_tensor = torch.tensor(final_query_bert)
    bm25_top_tensor = torch.tensor(bm25_top_embeddings)

    similarities = torch.matmul(bm25_top_tensor, query_tensor)

    top_k = 10
    top_scores, top_indices = torch.topk(similarities, k=top_k)

    results = []

    for score, idx in zip(top_scores, top_indices):
        real_idx = bm25_top_df.iloc[int(idx)].name
        row = df.loc[real_idx]
        results.append({
            "Score": score.item(),
            "Title": row["title"],
            "Artist": row["artist"],
            "Lyrics": row['lyrics']  
        })

    top_results_df = pd.DataFrame(results)
    top_results_df = top_results_df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    title_artist_list = [f"{row['Title']} - {row['Artist']}" for _, row in top_results_df.iterrows()]

    results = [
        inquirer.List(name='song_results',
                      message='Top 10 results, pick a song to see the full lyrics',
                      choices=title_artist_list)]

    results_answer = inquirer.prompt(results, theme=BlueComposure())
    results_answer = results_answer['song_results']
    selected_title, selected_artist = results_answer.split(" - ", 1)
    selected_row = \
        top_results_df[(top_results_df['Title'] == selected_title)].iloc[0]
    lyrics = selected_row['Lyrics']
    print(lyrics)


