from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
import torch

# Force CPU usage
torch.cuda.is_available = lambda: False
device = 'cpu'

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # Base model
NUMBER_OF_BOOKS = 96 # Number of books to process

def load_training_data(metadata_file='book_metadata2.json'):
    with open(metadata_file) as f:
        books = json.load(f)
    return [
        (book['blurb'], book['tags']) 
        for book in books
        if len(book['tags']) > 0
    ]

def fine_tune_model():
    # Load base model
    device = 'cpu'
    print(device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # Prepare training data
    train_examples = []
    count = 0
    for blurb, tags in load_training_data():
        count += 1
        if count >= NUMBER_OF_BOOKS:
            break
        # Create positive pairs
        for tag in tags:
            train_examples.append(InputExample(
                texts=[blurb, tag],
                label=1.0
            ))
        # Create negative examples (optional)
        # ...
    
    # DataLoader and loss
    loader = DataLoader(train_examples, shuffle=True, batch_size=32)
    loss = losses.MultipleNegativesRankingLoss(model)
    
    # Fine-tune
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=3,
        warmup_steps=100,
        output_path='fine-tuned-model2',
        show_progress_bar=True
    )

if __name__ == "__main__":
    fine_tune_model()
