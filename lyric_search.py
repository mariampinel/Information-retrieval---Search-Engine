#!/usr/bin/env python3
import inquirer
from inquirer.themes import BlueComposure
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi
import json
import torch
from torch.nn.functional import cosine_similarity


base_stopwords = set(stopwords.words("english"))
lyrical_keep_words = {
    'i', 'you', 'me', 'we', 'my', 'your', 'she', 'her', 'his',
    'not', 'no', 'never',
    "don't", "can't", "won't", "didn't", "isn't", "aren't",
    'oh', 'hey', 'yeah'
}

# due to the difference in lyrical meaning, some traditional stop words are kept to preserve semantic meaning
custom_stopwords = base_stopwords - lyrical_keep_words
stemmer = PorterStemmer()


# bm25 query preprocessing needs to go through the same steps as the documents
def bm25_query(query):
    words = re.findall(r"[\w'\-\[\]]+", query.lower())
    filtered = [
        word for word in words
        if (word in lyrical_keep_words) or ("'" in word) or (word not in base_stopwords)
    ]
    stemmed = [stemmer.stem(word) for word in filtered]
    return stemmed


# bert query preprocessing is done using the same model as the dataset
model = SentenceTransformer('all-MiniLM-L6-V2')


def bert_query(query):
    return model.encode(query, normalize_embeddings=True)


print('-------------------------------------------------------------')
print('Welcome to our lyric search, please select an option below:')
print('-------------------------------------------------------------')

df = pd.read_csv('/Users/jamesshortland/Desktop/preprocessed_genius_lyrics.csv')
bert_embeddings = np.load('/Users/jamesshortland/Desktop/bert_embeddings.npy')
with open('/Users/jamesshortland/Desktop/bert_ids.json') as f:
    berts_ids = json.load(f)

id_to_embedding_row = {song_id: i for i, song_id in enumerate(berts_ids)}

def search_results(df, query):
    final_query_bm25 = bm25_query(query)
    final_query_bert = bert_query(query)

    tokenized_lyrics = df['preprocessed_lyrics'].dropna().tolist()
    bm25 = BM25Okapi(tokenized_lyrics, k1=0.5, b=0.5)

    scores = bm25.get_scores(final_query_bm25)
    df['bm25_score'] = scores

    top_n = 25000
    bm25_top_df = df.sort_values(by='bm25_score', ascending=False).head(top_n)
    bm25_top_id = bm25_top_df['id'].tolist()

    embedding_indices = [id_to_embedding_row[song_id] for song_id in bm25_top_id if
                         song_id in id_to_embedding_row]

    bm25_top_embeddings = bert_embeddings[embedding_indices]

    query_tensor = torch.tensor(final_query_bert)
    bm25_top_tensor = torch.tensor(bm25_top_embeddings)

    similarities = cosine_similarity(query_tensor.unsqueeze(0), bm25_top_tensor)

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

section_stems = {
    'intro': ['intro'],
    'verse': ['verse', 'vers'],
    'chorus': ['chorus', 'choru'],
    'outro': ['outro']
}

# Process and filter lyrics based on section (if needed)
def process_lyrics(lyrics, sections):
    # Normalize headers in brackets
    lyrics = re.sub(r'\[([^\]]+)\]', lambda m: f"[{m.group(1).lower()}]", lyrics)

    # Map stemmed terms to section
    stem_to_section = {stem: section for section, stems in section_stems.items() for stem in stems}

    # Match sections to lyrics
    pattern = r'\[([^\]]+)\](.*?)(?=\[[^\]]+\]|$)'
    matches = re.findall(pattern, lyrics, re.DOTALL)

    output = []

    for raw_section, content in matches:
        section_label = raw_section.strip().split()[0]
        canonical_section = stem_to_section.get(section_label)

        if canonical_section in sections:
            output.append(content.strip())

    return '\n'.join(output)


# Only apply section filter if user selected something
def filter_lyrics_by_section(df, sections):
    if sections:
        df['preprocessed_lyrics'] = df['preprocessed_lyrics'].fillna('').apply(lambda x: process_lyrics(x, sections))
    else:
        df['preprocessed_lyrics'] = df['preprocessed_lyrics'].fillna('')
    return df


# Prompts
section_prompt = [
    inquirer.Checkbox(name='sections',
                      message='Limit search to specific lyric sections?',
                      choices=['intro', 'verse', 'chorus', 'outro', 'full song'])
]


on_opening = [
    inquirer.List(name='Search_options',
                  message='Know what you want to look for? Select an option below to narrow your search, or select '
                          '"search everything" to search the whole database',
                  choices=['Artist', 'Genre', 'Release Year', 'Search Everything'])]

genres = [
    inquirer.List(name='genres',
                  message='Which genre would you like to search in?',
                  choices=['Rap', 'RB', 'Rock', 'Pop', 'Country', 'go back'])]

decades = [
    inquirer.List(name='decades',
                  message='Select the decade you want to search from',
                  choices=[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 'go back'])]

while True:
    response = inquirer.prompt(on_opening, theme=BlueComposure())
    if not response:
        print("Something went wrong with the prompt. Exiting.")
        break

    opening = response['Search_options']

    if opening == 'Artist':
        while True:
            artist = input("Enter artist name (or type 'break' to go back):\n").strip().lower()
            df['artist_normalized'] = df['artist'].str.strip().str.lower()
            if artist == 'break':
                break
            if artist in df['artist_normalized'].values:
                print("We've got them!")
                filtered_df = df[df['artist_normalized'] == artist].copy()
                sections = inquirer.prompt(section_prompt, theme=BlueComposure())['sections']
                if "Full song" in sections:
                    sections = []
                filtered_df = filter_lyrics_by_section(filtered_df, sections)
                query = input(f"Enter lyrics to search for songs by {artist}:\n")
                search_results(filtered_df, query)
            else:
                print("Sorry, we couldn't find that artist.")

    elif opening == 'Genre':
        while True:
            genre = inquirer.prompt(genres, theme=BlueComposure())['genres']
            if genre == 'go back':
                break
            filtered_df = df[df['tag'].str.lower() == genre.lower()].copy()
            sections = inquirer.prompt(section_prompt, theme=BlueComposure())['sections']
            if "Full song" in sections:
                sections = []
            filtered_df = filter_lyrics_by_section(filtered_df, sections)
            query = input(f"Enter lyrics to search for songs in {genre}:\n")
            search_results(filtered_df, query)

    elif opening == 'Release Year':
        df['decade'] = (df['year'] // 10) * 10
        while True:
            decade = inquirer.prompt(decades, theme=BlueComposure())['decades']
            if decade == 'go back':
                break
            try:
                decade = int(decade)
                filtered_df = df[df['decade'] == decade].copy()
                if filtered_df.empty:
                    print(f"No songs found for the {decade}s.")
                    continue
                sections = inquirer.prompt(section_prompt, theme=BlueComposure())['sections']
                if "Full song" in sections:
                    sections = []
                filtered_df = filter_lyrics_by_section(filtered_df, sections)
                query = input(f"Enter lyrics to search for songs in the {decade}s:\n")
                search_results(filtered_df, query)
            except ValueError:
                print("Invalid input.")

    elif opening == 'Search Everything':
        while True:
            sections = inquirer.prompt(section_prompt, theme=BlueComposure())['sections']
            if "Full song" in sections:
                sections = []
            query = input("Enter lyrics to search the entire database (or press Enter to go back):\n")
            if query == '':
                break
            filtered_df = df.copy()
            filtered_df = filter_lyrics_by_section(filtered_df, sections)
            search_results(filtered_df, query)
