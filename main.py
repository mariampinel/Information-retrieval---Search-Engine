#!/usr/bin/env python3
import inquirer
from inquirer.themes import BlueComposure
import pandas as pd
from query_processing import bm25_query, bert_query
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import numpy as np
import json
import torch

print('Welcome to our lyric search, please select an option below:')

df = pd.read_csv('/Users/jamesshortland/Desktop/preprocessed_genius_lyrics.csv')
bert_embeddings = np.load('/Users/jamesshortland/Desktop/bert_embeddings.npy')
with open('/Users/jamesshortland/Desktop/bert_ids.json') as f:
    berts_ids = json.load(f)

id_to_embedding_row = {song_id: i for i, song_id in enumerate(berts_ids)}

on_opening = [
    inquirer.List(name='Search_options',
                  message='Know what you want to look for? Select an option below to narrow your search, or select '
                          '"search everything" to search the whole database',
                  choices=['Artist', 'Genre', 'Release Year', 'Search Everything'])]

while True:
    opening_answer = inquirer.prompt(on_opening, theme=BlueComposure())
    opening_answer = opening_answer['Search_options']
    if opening_answer == 'Artist':
        while True:
            artist = input("Search by artist: please type an artist's name below and we'll see if they're "
                           "in the database. type 'break' to return to the menu. \n")
            artist = artist.strip().lower()
            df['artist_normalized'] = df['artist'].str.strip().str.lower()

            if artist in df['artist_normalized'].values:
                print(f"We've got them!")
                df_artist = df[df['artist_normalized'] == artist].copy()
                df_artist.drop(columns='artist_normalized', inplace=True)
                lyrics = df_artist['preprocessed_lyrics'].dropna().tolist()
                tokenized_lyrics = [word_tokenize(song) for song in lyrics]
                bm25 = BM25Okapi(tokenized_lyrics)

                query = input(f"Enter lyrics to search for songs by {artist}:\n")

                final_query_bm25 = bm25_query(query)
                final_query_bert = bert_query(query)

                scores = bm25.get_scores(final_query_bm25)
                df_artist['bm25_score'] = scores

                top_n = 100  # or whatever range you want to pass into BERT
                bm25_top_df = df_artist.sort_values(by='bm25_score', ascending=False).head(top_n)
                bm25_top_id = bm25_top_df['id'].tolist()

                embedding_indices = [id_to_embedding_row[song_id] for song_id in bm25_top_id if
                                     song_id in id_to_embedding_row]

                bm25_top_embeddings = bert_embeddings[embedding_indices]

                query_tensor = torch.tensor(final_query_bert)
                bm25_top_tensor = torch.tensor(bm25_top_embeddings)

                similarities = torch.matmul(bm25_top_tensor, query_tensor)

                top_k = 5
                top_scores, top_indices = torch.topk(similarities, k=top_k)

                print("\n🎧 Top Results (BERT re-ranked from BM25):")

                for score, idx in zip(top_scores, top_indices):
                    real_idx = bm25_top_df.iloc[int(idx)].name
                    row = df.loc[real_idx]
                    print(f"\nScore: {score.item():.4f}")
                    print(f"Title: {row['title']}")
                    print(f'Artist: {row['artist']}')

            elif artist == 'break':
                break
            else:
                print("Sorry, we couldn't find that artist in the database, try again?")
