# RAG-as-YouTube-Assistant

An AI-powered assistant that answers questions about a YouTube video using **Retrieval-Augmented Generation (RAG)** with **long-term + short-term memory**.

## Features
- Hybrid retrieval: Transcript + Memory
- Smart fallback to entire transcript
- Persistent long-term memory in FAISS
- Multi-retriever evaluation with Ragas
- Context-aware interactive chat
- `.env` configuration (no hardcoded keys)

## 🛠 Requirements
```bash
pip install -U python-dotenv youtube-transcript-api langchain langchain-community langchain-openai faiss-cpu datasets ragas pandas
