import pandas as pd

df = pd.read_csv('/Users/jamesshortland/Desktop/song_lyrics.csv')
df_cleaned = df[df['language_cld3'] == 'en']
df_cleaned = df_cleaned[df_cleaned['language'] == 'en']
df_cleaned = df_cleaned[df_cleaned['language_ft'] == 'en']
df_cleaned = df_cleaned.drop(['language', 'language_ft', 'language_cld3'], axis=1)
df_cleaned = df_cleaned[df_cleaned['tag'] != 'misc']
df_cleaned = df_cleaned[df_cleaned['lyrics'].str.contains(r'\[', na=False)]
df_cleaned = df_cleaned.drop_duplicates(subset=['lyrics'])

df_cleaned = df_cleaned[df_cleaned['views'] > 10000]

df_cleaned.to_csv('genius_lyrics_reduced.csv', index=False)