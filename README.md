# 🎵 Lyric Search Engine

Welcome to the Lyric Search Engine! This tool lets you search through the lyrics of over 100,000 popular songs sourced from Genius.com.

## 🔧 Getting Started

To run the engine, make sure the following files are in the **same directory**:

- `lyric_search.py`
- `bert_ids.json`
- `bert_embeddings.npy`
- `preprocessed_genius_lyrics.csv`

## ▶️ How to Run

- **Preferred**: Open `lyric_search.py` in an IDE (such as PyCharm or VS Code) and run it in the **integrated terminal**.
- **Alternative**: You can also run the script in your system’s terminal, but make sure you’ve installed all the required Python libraries beforehand. (This option is recommended for more advanced users.)

## ⚙️ Functions

Once the program is running, you'll be prompted to narrow your search using one of the following options:

- **Artist** – Search for lyrics by a specific artist.
- **Genre** – Search within a specific genre (e.g., Pop, Rap, Rock).
- **Release Year** – Filter songs by decade to search lyrics from a particular era.
- **Search Everything** – Search the full database without filters.

After entering your query, the engine uses a two-step semantic search process to find the most relevant results:

1. **BM25** is used to retrieve the top matching candidates based on keyword relevance.
2. **BERT** embeddings are then used to re-rank these results based on semantic similarity to your query, improving the quality and context of the matches.

This hybrid approach allows the system to capture both lexical and semantic relevance for more accurate lyric discovery.

