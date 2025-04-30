import faiss
import numpy as np
import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "all-minilm-l6-v2"  # Match the model used in processing

class BookRecommender:
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        self.index = faiss.read_index('book_index.faiss')
        with open('book_metadata.json') as f:
            self.books = json.load(f)
        self.max_id = max(b['id'] for b in self.books) if self.books else 1
    
    def parse_query(self, query):
        tags = set(t.lower().strip() for t in re.findall(r'\[(.*?)\]', query))
        text = re.sub(r'\[.*?\]', '', query).strip()
        # Frame the query to emphasize "vibe"
        return {'tags': tags, 'text': f"Vibe: {text}"}

    def recommend(self, query, top_k=5, tag_weight=1.0):  # Reduced tag_weight
        parsed = self.parse_query(query)
        query_embedding = self.get_embedding(parsed['text'])
        
        # FAISS returns similarity scores (higher = better)
        similarities, indices = self.index.search(query_embedding, len(self.books))
        
        results = []
        seen_urls = set()
        
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.books):
                continue
                
            book = self.books[idx]
            if book['url'] in seen_urls:
                continue
                
            # Use raw similarity score (already normalized for inner product)
            text_score = similarities[0][i] * 10  # Scale to match tag weight
            tag_score = len(parsed['tags'] & set(book['tags'])) * tag_weight
            id_boost = self.calculate_id_boost(book['id'])
            
            total_score = (text_score + tag_score) * id_boost
            
            results.append({
                'score': total_score,
                'book': book,
                'components': {
                    'text': text_score,
                    'tags': tag_score,
                    'id_boost': id_boost
                }
            })
            seen_urls.add(book['url'])
        
        return sorted(results, key=lambda x: -x['score'])[:top_k]