# src/embeddings.py

from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from . import config # Import config from the same package (src)

# Set up logging for this module
logger = logging.getLogger(__name__)

# Cache the loaded embedding model
_embedding_model = None

def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads and returns the HuggingFaceEmbeddings model.
    This function ensures the model is loaded only once (singleton pattern).

    Returns:
        HuggingFaceEmbeddings: The loaded embedding model.
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        try:
            _embedding_model = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME,
                cache_folder=config.HUGGINGFACE_CACHE_DIR, # Specify cache directory
                model_kwargs={'device': 'cpu'}, # Force CPU for local setup; adjust if you have GPU
                encode_kwargs={'normalize_embeddings': True} # Often good for vector search
            )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model {config.EMBEDDING_MODEL_NAME}: {e}")
            raise # Re-raise the exception as the application cannot proceed without it

    return _embedding_model

# You can add a small test block here for direct testing during development
if __name__ == "__main__":
    print("--- Testing Embeddings Module ---")
    print(f"Embedding Model: {config.EMBEDDING_MODEL_NAME}")
    print(f"Cache Directory: {config.HUGGINGFACE_CACHE_DIR}")

    try:
        # Load the model
        model = load_embedding_model()
        print("\nEmbedding model loaded (or retrieved from cache).")

        # Test encoding a simple text
        test_text = "Hello, world! This is a test sentence."
        print(f"Encoding test text: '{test_text}'")
        vector = model.embed_query(test_text)
        print(f"Generated vector of length: {len(vector)}")
        print(f"First 5 dimensions: {vector[:5]}")

        # Test loading again to ensure singleton pattern works
        print("\nAttempting to load model again (should be fast, using cached instance)...")
        model_again = load_embedding_model()
        print("Model loaded again (from cache).")
        assert model is model_again, "Models are not the same instance, singleton failed!"
        print("Singleton pattern confirmed.")

    except Exception as e:
        print(f"An error occurred during embedding model test: {e}")