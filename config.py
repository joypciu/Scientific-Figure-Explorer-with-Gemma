# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration for deployment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_TYPE = "faiss"
MAX_FIGURES = 50