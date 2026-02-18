import os
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding)
def chunk_text(text , size=1200 , overlap=200):

    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks
def build_index(chunks):

    vectors = []

    for ch in chunks:
        vec = get_embedding(ch)
        vectors.append(vec)

    return np.array(vectors)
def retrieve(query , chunks , vectors , k=4):

    q_vec = get_embedding(query)

    scores = vectors @ q_vec / (
        np.linalg.norm(vectors , axis=1) * np.linalg.norm(q_vec)
    )

    top_idx = np.argsort(scores)[-k:][::-1]

    return [chunks[i] for i in top_idx]
