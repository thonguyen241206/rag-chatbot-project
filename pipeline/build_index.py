from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embedder import embed_text

import faiss
import numpy as np
import pickle

text = load_pdf("data/data_sample.pdf")

chunks = chunk_text(text)

embeddings = [embed_text(chunk) for chunk in chunks]

dimension = len(embeddings[0])

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "vector.index")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Build xong vector DB")
