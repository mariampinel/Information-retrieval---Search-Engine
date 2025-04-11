from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re

df = pd.read_csv('/Users/jamesshortland/Desktop/genius_lyrics_reduced.csv')

# 1. Base NLTK stopwords
base_stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

lyrical_keep_words = {
    # Pronouns
    'i', 'you', 'me', 'we', 'my', 'your', 'she', 'her', 'his',
    # Negations (full words + contractions)
    'not', 'no', 'never',
    "don't", "can't", "won't", "didn't", "isn't", "aren't",
    # Emotional/vocal
    'oh', 'hey', 'yeah'
}
def preprocess_lyrics(text):
    words = re.findall(r"[\w'\-\[\]]+", text.lower())  # preserves brackets
    filtered = [
        word for word in words
        if (word in lyrical_keep_words) or ("'" in word) or (word not in base_stopwords)
    ]
    stemmed = [stemmer.stem(word) for word in filtered]
    return ' '.join(stemmed)

df['preprocessed_lyrics'] = df['lyrics'].apply(preprocess_lyrics)

df.to_csv('preprocessed_genius_lyrics.csv')
