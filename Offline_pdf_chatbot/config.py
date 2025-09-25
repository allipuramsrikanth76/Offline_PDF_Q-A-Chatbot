import os

#path to save vectorsstore/index
PERSIST_PATH = os.getenv("PERSIST_PATH", "./vectorstore/faiss_index")
os.makedirs(PERSIST_PATH, exist_ok=True)

#sentence - transformers model for embeddings (offline)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

#ollama model name (Must be available in your local system)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

#chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

TOP_K = int(os.getenv("TOP_K", 4))









