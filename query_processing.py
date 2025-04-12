from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from sentence_transformers import SentenceTransformer

base_stopwords = set(stopwords.words("english"))
lyrical_keep_words = {
    # Pronouns
    'i', 'you', 'me', 'we', 'my', 'your', 'she', 'her', 'his',
    # Negations (full words + contractions)
    'not', 'no', 'never',
    "don't", "can't", "won't", "didn't", "isn't", "aren't",
    # Emotional/vocal
    'oh', 'hey', 'yeah'
}

custom_stopwords = base_stopwords - lyrical_keep_words
stemmer = PorterStemmer()
def bm25_query(query):
    # Tokenize using the same regex: keep words, apostrophes, dashes, and brackets
    words = re.findall(r"[\w'\-\[\]]+", query.lower())

    # Keep if:
    # - in lyrical keep words
    # - contains an apostrophe (e.g., don't)
    # - not in base stopwords
    filtered = [
        word for word in words
        if (word in lyrical_keep_words) or ("'" in word) or (word not in base_stopwords)
    ]

    # Stem
    stemmed = [stemmer.stem(word) for word in filtered]

    return stemmed

model = SentenceTransformer('all-MiniLM-L6-V2')
def bert_query(query):
    return model.encode(query, normalize_embeddings=True)