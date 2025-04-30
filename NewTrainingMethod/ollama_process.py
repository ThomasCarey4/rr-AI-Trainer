from bs4 import BeautifulSoup
import numpy as np
import faiss
import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "all-minilm-l6-v2"  # More efficient model for 6GB VRAM

def preprocess_book(entry):
    """Structured preprocessing to emphasize blurb 'vibe'"""
    tags = entry.tags.get_text(strip=True)
    blurb = entry.blurb.get_text(strip=True)
    return {
        # Structure text to separate blurb "vibe" and tags
        'search_text': f"Blurb: {blurb}\nTags: {tags}",
        'tags': [t.strip().lower() for t in tags.split(',')],
        'blurb': blurb,
        'url': entry.url.get_text(strip=True),
        'id': int(entry.get('id', 0))
    }

def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            'model': EMBED_MODEL,
            'prompt': text,
            'options': {'temperature': 0}  # Not needed for embeddings but required by API
        }
    )
    return np.array(response.json()['embedding'], dtype='float32')

def process_data(input_file='../scraped_output.txt', output_index='book_index.faiss'):
    # Load and parse data
    with open(input_file) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        books = [preprocess_book(entry) for entry in soup.find_all('entry')[:1000]]
    
    # Generate embeddings using Ollama
    embeddings = []
    for book in books:
        embedding = get_embedding(book['search_text'])
        embeddings.append(embedding)
    embeddings = np.stack(embeddings)
    
    # Create and save FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, output_index)
    
    # Save metadata
    with open('book_metadata.json', 'w') as f:
        json.dump(books, f)

if __name__ == "__main__":
    process_data()