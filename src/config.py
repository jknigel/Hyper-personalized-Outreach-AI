# src/config.py

import os

# --- Paths ---
# Use absolute paths or paths relative to the project root for clarity
# This makes it easier when running scripts from different directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KNOWLEDGE_BASE_DOCS_PATH = os.path.join(DATA_DIR, "knowledge_base_docs")
# PERSONA_TEMPLATES_PATH = os.path.join(DATA_DIR, "persona_templates") # Uncomment if you use this

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
HUGGINGFACE_CACHE_DIR = os.path.join(MODELS_DIR, "huggingface_models")
CHROMA_DB_DIR = os.path.join(MODELS_DIR, "chroma_db")

# Ensure these directories exist
os.makedirs(KNOWLEDGE_BASE_DOCS_PATH, exist_ok=True)
os.makedirs(HUGGINGFACE_CACHE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)


# --- MCP Server Configuration ---
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:5000")
MCP_SCRAPE_ENDPOINT = os.getenv("MCP_SCRAPE_ENDPOINT", "/scrape") # Default endpoint, adjust if needed


# --- Embedding Model Configuration (Hugging Face Sentence Transformers) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --- Vector Store Configuration (ChromaDB) ---
CHROMA_COLLECTION_NAME = "linkedin_outreach_knowledge"
RAG_NUM_RESULTS = 3 # Number of relevant chunks to retrieve from the vector DB


# --- Large Language Model (LLM) Configuration (Hugging Face Local) ---
# Choose a model that can run on your local machine.
# Options:
# "microsoft/phi-2" (good balance, ~4GB RAM)
# "gpt2" (very small, less capable)
# "distilgpt2" (even smaller, faster, less capable)
# "HuggingFaceH4/zephyr-7b-beta" (7B params, better quality, requires ~16GB RAM for CPU)
# You might need specific hardware/setup for larger models (e.g., GPU for 7B+).
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "microsoft/phi-2")

# LLM Generation Parameters
LLM_MAX_NEW_TOKENS = 250   # Max length of the generated message
LLM_TEMPERATURE = 0.7    # Controls creativity (0.0-1.0), higher means more creative
LLM_TOP_P = 0.9          # Controls diversity, typically 0.9 for good balance
# For CPU inference, use torch.float32. For GPU, consider torch.float16 or torch.bfloat16.
LLM_TORCH_DTYPE = os.getenv("LLM_TORCH_DTYPE", "torch.float32") # Will convert string to torch.dtype in llm_model.py


# --- Prompt Configuration ---
# You can define common prompt elements or parameters here, though the full prompt
# template will likely live in prompt_manager.py for better readability.
MESSAGE_LENGTH_GUIDELINE = "2-4 sentences maximum"
CALL_TO_ACTION_GUIDELINE = "polite, non-demanding call to action (e.g., 'I'd love to connect' or 'Would you be open to a quick chat?')."

# --- Other Configurations ---
# Add any other global constants or settings here