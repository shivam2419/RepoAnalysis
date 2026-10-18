# RepoAnalysis

RepoAnalysis is an MVP for an AI-powered GitHub repository analysis assistant. It clones a public GitHub repository, extracts readable source files, chunks them, creates local embeddings, stores them in a FAISS index, and answers questions using the Groq API.

## Features

- Public GitHub HTTPS validation
- Safe repository cloning with temporary directories
- File filtering for source and config files
- Chunking with metadata for file path and line ranges
- Local sentence-transformers embeddings and FAISS retrieval
- Groq chat completions using its OpenAI-compatible API
- FastAPI backend for indexing and chatting
- Streamlit frontend for repository indexing and Q&A
- Beginner-friendly modular project structure

## Python version

This project targets Python 3.11+.

## Setup

1. Create a virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Copy the sample environment file and set your Groq API key in `.env`:

   ```bash
   copy .env.example .env
   ```

   - `AI_API_KEY`: your Groq API key
   - `AI_MODEL`: `openai/gpt-oss-120b`
   - `AI_ENDPOINT`: `https://api.groq.com/openai/v1` (plain URL, not Markdown link syntax)

   Groq handles answer generation. The `sentence-transformers/all-MiniLM-L6-v2` model embeds code and questions on your computer, so no separate embedding API key is needed. Its model files download the first time you index a repository; it returns 384-dimensional vectors, matching `VECTOR_DIMENSION`.

   `MAX_CONTEXT_CHARS` limits how much retrieved code is sent per question, and `MAX_COMPLETION_TOKENS` limits answer length. Lower these if your Groq account has strict token-per-minute limits.

   Never commit `.env` or share your Groq API key. If a real key has been exposed, revoke it and create a replacement.

4. Start the API backend:

   ```bash
   uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
   ```

5. Start the Streamlit app in a second terminal:

   ```bash
   streamlit run app/streamlit_app.py
   ```

## Security notes

- Only GitHub HTTPS repository URLs are accepted.
- The app avoids executing repository code and does not install dependencies from indexed repositories.
- Secret-like files such as `.env` files are ignored.
- Temporary directories are cleaned up when repositories are reset.

## Example questions

- How does authentication work in this project?
- Which file configures the database connection?
- How does the frontend communicate with the backend?
- Where is user registration implemented?
