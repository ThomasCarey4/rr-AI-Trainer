from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import torch

# Upgraded to one of the best open-source models available
MODEL_NAME = 'sentence-transformers/gtr-t5-large'  # 335M parameter model
# Alternative options (uncomment one):
# MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # 110M params (good balance)
# MODEL_NAME = 'sentence-transformers/sentence-t5-xxl'    # 4.8B params (massive)

def preprocess_book(entry):
    """Enhanced preprocessing with more context"""
    tags = entry.tags.get_text(strip=True)
    blurb = entry.blurb.get_text(strip=True)
    return {
        'search_text': f"Genre/Tags: {tags}. Synopsis: {blurb}",
        'tags': tags,
        'blurb': blurb,
        'url': entry.url.get_text(strip=True),
        'id': int(entry.get('id', 0))
    }

# Load data
with open('scraped_output.txt') as f:
    books = [preprocess_book(entry) for entry in BeautifulSoup(f.read(), 'html.parser').find_all('entry')]

# Initialize model with automatic device detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(MODEL_NAME, device=device)

# Batch processing for memory efficiency
batch_size = 32 if 'large' in MODEL_NAME else 128
embeddings = model.encode(
    [book['search_text'] for book in books],
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_tensor=False,
    normalize_embeddings=True
)

# Create and populate the index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype('float32'))  # FAISS requires float32

# Save index and metadata
faiss.write_index(index, 'book_index.faiss')
with open('book_metadata.json', 'w') as f:
    json.dump(books, f)