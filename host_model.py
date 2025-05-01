from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from recommend import BookRecommender
from pydantic import BaseModel

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = BookRecommender()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/recommend")
async def recommend_books(request: QueryRequest):
    results = recommender.recommend(request.query, top_k=request.top_k)
    
    # Convert numpy floats to Python floats for JSON serialization
    for result in results:
        result['score'] = float(result['score'])
        for key in result['components']:
            result['components'][key] = float(result['components'][key])
    
    return {"results": results}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}