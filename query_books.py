import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import torch

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

def recommend(query, top_k=5, min_score=1.0):  # min_score on 0-10 scale
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # Load data
    index = faiss.read_index('book_index.faiss')
    with open('book_metadata.json') as f:
        books = json.load(f)
    
    if not books:
        return []

    # Get query embedding
    query_embedding = model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
    
    # Search with more candidates
    distances, indices = index.search(query_embedding, min(1000, len(books)))
    
    seen_urls = set()
    results = []
    max_id = max(b['id'] for b in books) if books else 1
    
    for i, idx in enumerate(indices[0]):
        if idx >= len(books):
            continue
            
        book = books[idx]
        if book['url'] in seen_urls:
            continue
            
        # Convert cosine similarity to 0-1 scale properly
        cosine_sim = (1 + distances[0][i]) / 2  # FAISS returns cosine distances
        
        # Apply ID boost (1.0-1.1 range)
        id_boost = 1.0 + (1.0 - (book['id'] / max_id)) * 0.1
        
        # Calculate final score (0-10 scale)
        final_score = cosine_sim * 10 * (0.95 ** (max_id - book['id']))
        
        if final_score >= min_score:
            seen_urls.add(book['url'])
            results.append({
                'url': book['url'],
                'score': round(final_score, 2),
                'tags': book['tags'],
                'blurb': book['blurb'][:200] + ("..." if len(book['blurb']) > 200 else ""),
                'id': book['id']
            })
    
    # Sort results by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    test_queries = [
        "funny xianxia story",
    ]
    
    for query in test_queries:
        print(f"\n{'='*40}\nResults for: '{query}'\n{'='*40}")
        results = recommend(query, min_score=0.1)  # Start with very low threshold
        
        if not results:
            print("No matches found (try lowering min_score below 0.1)")
        else:
            for i, res in enumerate(results):
                print(f"\n{i+1}. Score: {res['score']:.2f}/10 (ID: {res['id']})")
                print(f"URL: {res['url']}")
                print(f"Tags: {res['tags']}")
                print(f"Blurb: {res['blurb']}")

        # Debug output
        if results:
            print("\nDebug Info:")
            print(f"Highest score: {results[0]['score']:.2f}")
            print(f"Lowest score: {results[-1]['score']:.2f}")
            print(f"Number of results: {len(results)}")