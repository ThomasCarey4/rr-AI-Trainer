from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import faiss, torch

# — 1. Install sentencepiece (Nix): python3Packages.sentencepiece

# — 2. Load & quantize Flan-T5-small
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
model.eval()

# — 3. Embedder + chunker
embedder = SentenceTransformer('all-MiniLM-L6-v2')
def chunk(txt, max_tokens=200):
    toks = tokenizer.tokenize(txt)
    for i in range(0, len(toks), max_tokens):
        yield tokenizer.convert_tokens_to_string(toks[i:i+max_tokens])

# — 4. Load & chunk blurbs
blurbs = []
with open('scraped_output.txt','r',encoding='utf-8') as f:
    entry=''
    for line in f:
        if line.strip()=='</ENTRY>':
            b=entry.split('<BLURB>')[1].split('</BLURB>')[0].strip()
            blurbs += list(chunk(b))
            entry=''
        else:
            entry+=line

# — 5. Build cosine-FAISS index
embs = embedder.encode(blurbs, convert_to_tensor=True)
embs = torch.nn.functional.normalize(embs, dim=1).cpu().numpy()
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

# — 6. Query fn with threshold + prompt
def ask(q, k=3, sim_thresh=0.2):
    q_emb = embedder.encode([q], convert_to_tensor=True)
    q_emb = torch.nn.functional.normalize(q_emb, dim=1).cpu().numpy()
    sims, I = index.search(q_emb, k)
    if sims[0][0] < sim_thresh:
        print("No story found.")
        return

    ctx = "\n".join(f"{i+1}. {blurbs[idx]}" for i,idx in enumerate(I[0]))
    prompt = (
        "tell me a joke”\n\n"
        f"Facts:\n{ctx}\n\nQ: {q}\nA:"
    )

    inp = tokenizer(
        prompt, return_tensors='pt',
        truncation=True, max_length=512, padding='longest'
    )
    out = model.generate(
        **inp,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        max_length=128,
        early_stopping=True,
        do_sample=False
    )
    ans = tokenizer.decode(out[0], skip_special_tokens=True).split("A:")[-1].strip()
    print(ans)

# Example
ask("Is there a story about time loops?")
