import os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        chunk = text[i:i+chunk_size]
        chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks

def ingest_dir(data_dir='data', store_dir='store'):
    os.makedirs(store_dir, exist_ok=True)
    texts = []
    filenames = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.txt', '.md')):
                path = os.path.join(root, f)
                with open(path, 'r', encoding='utf-8') as fh:
                    texts.append(fh.read())
                    filenames.append(f)
    if not texts:
        return 'No documents found in data/, add a .txt file.'

    chunks = []
    chunk_sources = []
    for idx, t in enumerate(texts):
        cks = chunk_text(t)
        for j, c in enumerate(cks):
            chunks.append(c)
            chunk_sources.append({'file': filenames[idx], 'chunk': j})

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(chunks)

    with open(os.path.join(store_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(store_dir, 'matrix.pkl'), 'wb') as f:
        pickle.dump(X, f)
    with open(os.path.join(store_dir, 'chunks.pkl'), 'wb') as f:
        pickle.dump(chunks, f)
    with open(os.path.join(store_dir, 'sources.pkl'), 'wb') as f:
        pickle.dump(chunk_sources, f)

    return f'Ingested {len(chunks)} chunks from {len(texts)} documents into {store_dir}'
