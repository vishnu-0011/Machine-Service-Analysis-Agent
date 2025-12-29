# Machine Service RAG System

This project provides a Retrieval-Augmented Generation (RAG) system for answering questions about machine service records using local LLMs and embeddings via Ollama, with LangChain and ChromaDB.

## Features
- Loads and indexes 40,000+ machine service records from a CSV file
- Uses Ollama for both embeddings and LLM (Qwen3:4B)
- Supports natural language Q&A with fallback logic for statistics and record count
- Handles large datasets by batching vector store inserts

## Requirements
- Python 3.9+
- Ollama (https://ollama.com/download) running locally
- The following Python packages (see requirements.txt):
  - langchain
  - langchain-ollama
  - langchain-chroma
  - pandas

## Setup
1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download and start Ollama:
   - Download from https://ollama.com/download
   - Start the server: `ollama serve`
   - Pull required models:
     ```
     ollama pull mxbai-embed-large
     ollama pull qwen3:4b
     ```
3. Place your `machine_service.csv` in the project root.

## Usage
Run the main script:
```
python main.py
```
Ask questions about your machine service records interactively.

## Notes
- The first run will build the vector store from the CSV. This may take several minutes for large datasets.
- If you want to rebuild the vector store, delete the `chrome_langchain_db` folder and rerun.

## Example Questions
- How many machine service records are in the database?
- What types of problems have occurred most frequently?
- What is the sum of all service costs in the records?
- Which machines had a "Sensor Error" in 2023?

---

MIT License
