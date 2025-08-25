import re
class MockLLM:
    """A tiny mock LLM that 'summarizes' retrieved chunks by taking their first sentences
    and returning them as a short answer referencing sources.""" 
    def generate(self, question: str, retrieved: list):
        # retrieved: list of dicts with 'text' and 'source'
        pieces = []
        for i, r in enumerate(retrieved, start=1):
            text = r.get('text', '').strip()
            # extract first sentence
            first = re.split(r'(?<=[.!?])\\s+', text)[0]
            pieces.append(f"{first} [{i}]")
        if pieces:
            summary = ' '.join(pieces)
            return f"Question: {question}\n\nAnswer (mock): {summary}\n\nSources:\n" + '\n'.join([f"[{i}] {r.get('source')}" for i,r in enumerate(retrieved, start=1)])
        else:
            return 'Sorry — no retrieved context to answer the question.' 
