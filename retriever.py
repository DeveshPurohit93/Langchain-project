import os, pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleRetriever:
    def __init__(self, store_dir='store'):
        self.store_dir = store_dir
        vfile = os.path.join(store_dir, 'vectorizer.pkl')
        mfile = os.path.join(store_dir, 'matrix.pkl')
        cfile = os.path.join(store_dir, 'chunks.pkl')
        sfile = os.path.join(store_dir, 'sources.pkl')
        if not all(os.path.exists(p) for p in [vfile, mfile, cfile, sfile]):
            raise FileNotFoundError('Store not found. Run ingest first: python app.py ingest')
        with open(vfile, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(mfile, 'rb') as f:
            self.matrix = pickle.load(f)  # sparse matrix
        with open(cfile, 'rb') as f:
            self.chunks = pickle.load(f)
        with open(sfile, 'rb') as f:
            self.sources = pickle.load(f)

    def retrieve(self, query, k=3):
        qv = self.vectorizer.transform([query])  # 1 x D
        sims = cosine_similarity(qv, self.matrix)[0]  # (N,)
        idxs = np.argsort(sims)[::-1][:k]
        results = []
        for i in idxs:
            results.append({
                'score': float(sims[i]),
                'text': self.chunks[i],
                'source': self.sources[i]
            })
        return results
