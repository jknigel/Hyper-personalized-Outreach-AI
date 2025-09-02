# src/llm_model.py

import os
import logging
from typing import Optional

# Import necessary Hugging Face and LangChain components
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch

from . import config # Import config from the same package (src)

# Set up logging for this module
logger = logging.getLogger(__name__)

# Cache the loaded LLM model and pipeline
_llm_model_pipeline: Optional[HuggingFacePipeline] = None

def load_llm_model() -> HuggingFacePipeline:
    """
    Loads and returns the Hugging Face Large Language Model wrapped in a LangChain pipeline.
    This function ensures the model is loaded only once (singleton pattern).

    Returns:
        HuggingFacePipeline: The loaded and wrapped LLM model.
    """
    global _llm_model_pipeline

    if _llm_model_pipeline is None:
        logger.info(f"Loading LLM model: {config.LLM_MODEL_ID}...")
        logger.info(f"Model will be cached in: {config.HUGGINGFACE_CACHE_DIR}")
        logger.info(f"Using torch_dtype: {config.LLM_TORCH_DTYPE}")

        try:
            # Dynamically get the torch dtype
            torch_dtype = getattr(torch, config.LLM_TORCH_DTYPE, torch.float32)
            if not isinstance(torch_dtype, torch.dtype):
                logger.warning(f"Invalid torch_dtype specified: {config.LLM_TORCH_DTYPE}. Defaulting to torch.float32.")
                torch_dtype = torch.float32

            # 1. Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL_ID,
                cache_dir=config.HUGGINGFACE_CACHE_DIR,
                trust_remote_code=True
            )
            # Some models (e.g., GPT2) don't have a pad token by default, which can cause issues with batching
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Tokenizer pad token set to EOS token.")

            # 2. Load Model
            model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_ID,
                torch_dtype=torch_dtype,
                cache_dir=config.HUGGINGFACE_CACHE_DIR,
                trust_remote_code=True,
                # For GPU usage, you might add:
                # device_map="auto",
                # load_in_8bit=True, # For 8-bit quantization
                # load_in_4bit=True, # For 4-bit quantization
            )
            model.eval() # Set model to evaluation mode
            logger.info("LLM model and tokenizer loaded successfully.")

            # 3. Create HuggingFace Pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                do_sample=True, # Enable sampling for creative responses
                top_p=config.LLM_TOP_P,
                # repetition_penalty=1.1, # Optional: to reduce repetitive phrases
                # Set device if not using device_map="auto"
                # device=0 if torch.cuda.is_available() else -1 # 0 for first GPU, -1 for CPU
            )

            # 4. Wrap in LangChain HuggingFacePipeline
            _llm_model_pipeline = HuggingFacePipeline(pipeline=pipe)
            logger.info("LangChain HuggingFacePipeline created.")

        except Exception as e:
            logger.error(f"Failed to load LLM model {config.LLM_MODEL_ID}: {e}", exc_info=True)
            logger.warning("Please ensure you have enough RAM/GPU memory and the model ID is correct.")
            raise # Re-raise the exception as the application cannot proceed without it

    return _llm_model_pipeline

# You can add a small test block here for direct testing during development
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Ensure logging is active
    print("--- Testing LLM Model Module ---")
    print(f"LLM Model ID: {config.LLM_MODEL_ID}")
    print(f"Cache Directory: {config.HUGGINGFACE_CACHE_DIR}")
    print(f"Max New Tokens: {config.LLM_MAX_NEW_TOKENS}, Temperature: {config.LLM_TEMPERATURE}")

    try:
        # Load the model
        llm = load_llm_model()
        print("\nLLM model loaded (or retrieved from cache).")

        # Test generation with a simple prompt
        test_prompt = "Hello, I am a software engineer. Write a short, friendly introduction:"
        print(f"\nGenerating response for prompt: '{test_prompt}'")
        response = llm.invoke(test_prompt)
        print("\n--- Generated Response ---")
        print(response.strip())
        print("---")

        # Test loading again to ensure singleton pattern works
        print("\nAttempting to load model again (should be fast, using cached instance)...")
        llm_again = load_llm_model()
        print("Model loaded again (from cache).")
        assert llm is llm_again, "LLM models are not the same instance, singleton failed!"
        print("Singleton pattern confirmed.")

    except Exception as e:
        print(f"An error occurred during LLM model test: {e}")
        print("Please check your LLM_MODEL_ID in config.py and ensure system resources (RAM/GPU) are sufficient.")