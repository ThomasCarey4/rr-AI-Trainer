from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import torch
import re

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # Base model

def preprocess_book(entry):
    """Enhanced preprocessing with tag normalization"""
    tags = entry.tags.get_text(strip=True)
    return {
        'search_text': f"{tags}: {entry.blurb.get_text(strip=True)}",
        'title': entry.title.get_text(strip=True),
        'tags': [t.strip().lower() for t in tags.split(',')],
        'blurb': entry.blurb.get_text(strip=True),
        'url': entry.url.get_text(strip=True),
        'id': int(entry.get('id', 0))
    }

def process_data(input_file='../scraped_output2.txt', output_index='book_index2.faiss'):
    # Load and parse data
    with open(input_file) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        books = [preprocess_book(entry) for entry in soup.find_all('entry')[:2048]]
    # Initialize model
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == 'cuda':
        print("GPU Memory Allocated:", torch.cuda.memory_allocated(0))
        print("GPU Memory Cached:", torch.cuda.memory_reserved(0))
    # Generate embeddings in batches
    embeddings = model.encode(
        [b['search_text'] for b in books],
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True
    ).astype('float32')  # Move to CPU for FAISS
    
    # Create and save FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, output_index)
    
    # Save metadata
    with open('book_metadata2.json', 'w') as f:
        json.dump(books, f)

if __name__ == "__main__":
    process_data()