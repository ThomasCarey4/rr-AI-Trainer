from recommend import BookRecommender

def test_system():
    recommender = BookRecommender()
    
    test_queries = [
        "[comedy] [xianxia] [!harem] story",
    ]
    
    for query in test_queries:
        print(f"\n{'-'*40}\nQuery: {query}\n{'-'*40}")
        results = recommender.recommend(query, top_k=3)
        
        if not results:
            print("No results found")
            continue
            
        for i, res in enumerate(results):
            book = res['book']
            print(f"\n{i+1}. Score: {res['score']:.2f}")
            print(f"ID: {book['id']} | URL: {book['url']}")
            print(f"Tags: {', '.join(book['tags'])}")
            print(f"Blurb: {book['blurb'][:150]}...")
            print(f"Score Breakdown: {res['components']}")

if __name__ == "__main__":
    test_system()
