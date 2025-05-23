# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration for deployment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("SITE_URL", "https://localhost:8501")  # Your site URL
SITE_NAME = os.getenv("SITE_NAME", "Scientific Figure Explorer")  # Your site name
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_TYPE = "faiss"
MAX_FIGURES = 50