import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import re
import torch

torch.cuda.is_available = lambda: False

class BookRecommender:
    def __init__(self, model_path='fine-tuned-model2'):
        self.device = 'cpu' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_path, device=self.device)
        self.load_data()
    
    def load_data(self):
        self.index = faiss.read_index('book_index2.faiss')
        with open('book_metadata2.json') as f:
            self.books = json.load(f)
        self.max_id = max(b['id'] for b in self.books) if self.books else 1
    
    def parse_query(self, query):
        # Extract all tags (both regular and exclusion tags)
        all_tags = re.findall(r'\[(.*?)\]', query)
        text = re.sub(r'\[.*?\]', '', query).strip()
        
        # Separate inclusion and exclusion tags
        include_tags = set()
        exclude_tags = set()
        
        for tag in all_tags:
            tag = tag.lower().strip()
            if tag.startswith('!'):
                exclude_tags.add(tag[1:])  # Remove the ! prefix
            else:
                include_tags.add(tag)
        
        return {
            'include_tags': include_tags,
            'exclude_tags': exclude_tags,
            'text': text
        }
    
    def calculate_id_boost(self, book_id, strength=5.0):
        """Boost lower IDs with exponential decay"""
        return 1.0 + strength * (1 - book_id / self.max_id)
    
    def recommend(self, query, top_k=5, tag_weight=2.0):
        parsed = self.parse_query(query)
        query_embedding = self.model.encode(
            parsed['text'], 
            convert_to_tensor=False,
            normalize_embeddings=True
        ).astype('float32')
        
        # Reshape the embedding to 2D if needed
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search index
        distances, indices = self.index.search(query_embedding, len(self.books))
        
        results = []
        seen_urls = set()
        
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.books):
                continue
                
            book = self.books[idx]
            if book['url'] in seen_urls:
                continue
                
            # Check for excluded tags
            book_tags = set(book['tags'])
            if parsed['exclude_tags'] and parsed['exclude_tags'].intersection(book_tags):
                continue  # Skip books with excluded tags
                
            # Calculate components
            text_score = (1 + distances[0][i]) / 2  # Convert to 0-1 scale
            tag_score = len(parsed['include_tags'] & book_tags) * tag_weight
            id_boost = self.calculate_id_boost(book['id'])
            
            total_score = (text_score * 10 + tag_score) * id_boost
            
            results.append({
                'score': total_score,
                'book': book,
                'components': {
                    'text': text_score * 10,
                    'tags': tag_score,
                    'id_boost': id_boost
                }
            })
            seen_urls.add(book['url'])
        
        # Sort and return top results
        return sorted(results, key=lambda x: -x['score'])[:top_k]
