from collections import Counter
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_csv('/Users/jamesshortland/Desktop/genius_lyrics_reduced.csv')

nltk_stopwords = set(stopwords.words("english"))

print(nltk_stopwords)

stopword_counts = Counter()
for lyric in df["lyrics"].dropna().astype(str):
    words = word_tokenize(lyric.lower())
    # Track stopwords that appear in your lyrics
    for word in words:
        if word in nltk_stopwords:
            stopword_counts[word] += 1

print("Top stopwords in lyrics:")
print(stopword_counts.most_common(50))

lyrical_overrides = {
    'i', 'you', 'me', 'we', 'us', 'my', 'in', 'that', 'on', 'your', 'of', 'with', 'for', 'she', 'he'
    'o', 'hey', 'yeah', 'now', 'can', 'her', 'his', 'them'             # Vocal expressions
    'don\'t', 'can\'t', 'not'         # Negations
}

auto_keep = {
    word for word, count in stopword_counts.most_common(200)
    if word in lyrical_overrides
}

print("Auto-selected keep-words:")
print(auto_keep)
