# RAG Learning

Hands-on learning project: exploring LangChain and building a Retrieval-Augmented Generation (RAG) pipeline step by step, using Groq-hosted LLMs.

## Getting started

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Copy `.env.example` to `.env` and set your API keys:

   ```env
   GROQ_API_KEY=your_groq_api_key
   GUARDIAN_API_KEY=your_guardian_api_key
   ```

3. Run the project:

   ```bash
   uv run rag-learning
   ```

## Current functionality

Download album reviews from the Guardian API:

```bash
uv run python -m rag_learning.ingest.guardian
```

## Tests

```bash
uv run pytest
```
