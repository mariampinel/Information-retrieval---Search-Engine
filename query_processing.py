from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from sentence_transformers import SentenceTransformer

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