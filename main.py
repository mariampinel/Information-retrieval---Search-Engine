#!/usr/bin/env python3
import inquirer
from inquirer.themes import BlueComposure
import pandas as pd
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from search_function import search_results

print('---------------------------------')
print('Welcome to our lyric search, please select an option below:')
print('---------------------------------')

df = pd.read_csv('/Users/jamesshortland/Desktop/preprocessed_genius_lyrics.csv')

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

                query = input(f"Enter lyrics to search for songs by {artist}:\n")
                search_results(df_artist, query)

            elif artist == 'break':
                break
            else:
                print("Sorry, we couldn't find that artist in the database, try again?")
    if opening_answer == 'Genre':
        while True:
            print('Select from the options below')
            genre_answer = inquirer.prompt(genres, theme=BlueComposure())
            genre_answer = genre_answer['genres']

            if genre_answer == 'go back':
                break

            else:
                genre_df = df[df['tag'].str.lower() == genre_answer.lower()].copy()

                query = input(f"Enter lyrics to search for songs in {genre_answer}:\n")
                search_results(genre_df, query)

    if opening_answer == 'Release Year':
        while True:
            print('Select from the decades below to narrow your search')
            decade_answer = inquirer.prompt(decades, theme=BlueComposure())
            decade_answer = decade_answer['decades']

            df['decade'] = (df['year'] // 10) * 10

            if decade_answer == 'go back':
                break
            else:
                decade_int = int(decade_answer)
                decade_df = df[df['decade'] == decade_int].copy()

                if decade_df.empty:
                    print(f"No songs found for the {decade_int}s.")
                    continue

                query = input(f"Enter lyrics to search for songs in the {decade_int}'s:\n")
                search_results(decade_df, query)

    if opening_answer == 'Search Everything':
        while True:
            query = input(f"Enter lyrics to search for songs in the whole database or just press enter to go back\n")
            if query == '':
                break
            else:
                search_results(df, query)
