#!/usr/bin/env python3
import inquirer
from inquirer.themes import BlueComposure
import pandas as pd
import re
from search_function import search_results

print('Welcome to our lyric search, please select an option below:')

df= pd.read_csv('preprocessed_genius_lyrics.csv')
# Mapping stemmed versions to sections
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
        section_label = raw_section.strip().split()[0] # Take first word of section header only
        canonical_section = stem_to_section.get(section_label)

        if canonical_section in sections:
            output.append(content.strip())

    return '\n'.join(output)

# Only apply section filter only if user selected something
def filter_lyrics_by_section(df, sections):
    if sections:
        df['preprocessed_lyrics'] = df['preprocessed_lyrics'].fillna('').apply(lambda x: process_lyrics(x, sections))
    else:
        df['preprocessed_lyrics'] = df['preprocessed_lyrics'].fillna('')
    return df

# Sections prompt
section_prompt = [
    inquirer.Checkbox(name='sections',
                      message='Limit search to specific lyric sections?',
                      choices=['intro', 'verse', 'chorus', 'outro'])
]

on_opening = [
    inquirer.List(name='Search_options',
                  message='Know what you want to look for? Select an option below:',
                  choices=['Artist', 'Genre', 'Release Year', 'Search Everything'])
]

genres = [
    inquirer.List(name='genres',
                  message='Which genre would you like to search in?',
                  choices=['Rap', 'RB', 'Rock', 'Pop', 'Country', 'go back'])
]

decades = [
    inquirer.List(name='decades',
                  message='Select the decade you want to search from',
                  choices=[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 'go back'])
]


# while True:
#     opening = inquirer.prompt(on_opening, theme=BlueComposure())['Search_options']
while True:
    response = inquirer.prompt(on_opening, theme=BlueComposure())
    print("Prompt response:", response)  # <-- Add this line for debugging

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
                filtered_df = filter_lyrics_by_section(filtered_df, sections)
                query = input(f"Enter lyrics to search for songs in the {decade}s:\n")
                search_results(filtered_df, query)
            except ValueError:
                print("Invalid input.")

    elif opening == 'Search Everything':
        while True:
            query = input("Enter lyrics to search the entire database (or press Enter to go back):\n")
            if query == '':
                break
            sections = inquirer.prompt(section_prompt, theme=BlueComposure())['sections']
            filtered_df = df.copy()
            filtered_df = filter_lyrics_by_section(filtered_df, sections)
            search_results(filtered_df, query)
