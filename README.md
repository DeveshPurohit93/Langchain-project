# Simple LangChain Tutorial (Mini RAG)

This is a tiny, **self-contained** project to *learn the key ideas behind LangChain*:
- Document loading
- Text splitting (chunks)
- Vectorization (like embeddings)
- A simple vector store + retriever
- A (mock) LLM that produces a concise answer using retrieved chunks

> This project *does not require* OpenAI API keys or heavy LLMs. It's intentionally minimal so you can run it quickly and learn concepts. Later in the README you'll find instructions on how to replace the mock LLM with a real LangChain + HuggingFace/OpenAI model.

## Files
- `app.py` - CLI entrypoint (ingest & ask)
- `ingest.py` - Load text files, chunk them, build TF-IDF vectors and save store
- `retriever.py` - Simple retriever using TF-IDF + cosine similarity
- `mock_llm.py` - A tiny "LLM" that creates a summary from retrieved chunks
- `data/sample.txt` - Example document
- `requirements.txt` - Python packages (lightweight)

## Quick start (Windows / macOS / Linux)
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Ingest the `data/` folder:
   ```bash
   python app.py ingest
   ```
4. Ask a question (examples provided):
   ```bash
   python app.py ask "what does the sample document say about langchain?"
   ```

## What this demonstrates (mapping to LangChain)
- **Document loaders** -> `ingest.py` reading files from `data/`
- **Text splitters** -> `ingest.py` `chunk_text()` splits long docs into chunks
- **Embeddings / vectorization** -> TF-IDF vectorizer (a lightweight stand-in for real embeddings)
- **Vector store** -> pickled TF-IDF matrix + chunks in `store/`
- **Retriever** -> `retriever.py` performs similarity search
- **LLM / Chain** -> `mock_llm.py` simulates an LLM step that consumes retrieved chunks and a prompt to produce an answer

## How to upgrade to real LangChain LLMs
In `app.py`, the mock LLM is used via `mock_llm.MockLLM`. Replace `MockLLM` with a LangChain-compatible LLM:

- **OpenAI (requires key):**
  ```python
  from langchain.chat_models import ChatOpenAI
  llm = ChatOpenAI(openai_api_key="YOUR_KEY", model="gpt-4o-mini")
  ```
- **HuggingFace (local or hub):**
  ```python
  from langchain import HuggingFaceHub
  llm = HuggingFaceHub(repo_id="google/flan-t5-small", huggingfacehub_api_token="YOUR_TOKEN")
  ```

Then build a prompt with `langchain.prompts.PromptTemplate` and call `llm` in place of `MockLLM.generate(...)`.

## Notes
- This tutorial is intentionally small and focused on concepts. It avoids heavy downloads and API keys so you can experiment quickly.
