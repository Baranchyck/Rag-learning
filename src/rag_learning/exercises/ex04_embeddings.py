from sentence_transformers import SentenceTransformer
import json 

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# texts =[
#     "сумний гітарний альбом",
#     "melancholic guitar record",
#     "веселий танцювальний поп",
#     "Шахтар виграв у Динамо",
# ]

# emb = model.encode(texts, normalize_embeddings=True)
# print('Embedding Shape ====')

# print(emb.shape)

# scores = emb @ emb.T
# print('Embedding Scores ====')

# print(scores.round(2))

# ==========================

reviews = []
with open("data/processed/guardian_reviews.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 4000:
            break
        reviews.append(json.loads(line))

docs = [r["headline"] + ". " + (r["trail"] or "") for r in reviews]
doc_emb = model.encode(docs, normalize_embeddings=True)


def search(query: str, k: int = 5) -> None:
    q = model.encode(query, normalize_embeddings=True)
    scores = doc_emb @ q
    top = scores.argsort()[::-1][:k]
    print(f"\n=== {query}")
    for i in top:
        print(f"{scores[i]:.2f}  {reviews[i]['headline']}")


search("Rock")
search("Rap")
search("Jazz")
search("heavy blues rock like 70s British bands")