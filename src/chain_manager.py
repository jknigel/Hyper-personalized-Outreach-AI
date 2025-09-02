# src/chain_manager.py

import logging
from typing import Optional

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.vectorstores.base import VectorStoreRetriever

from . import config
from .llm_model import load_llm_model
from .vector_store import get_retriever
from .prompt_manager import get_outreach_prompt_template

logger = logging.getLogger(__name__)

# Cache the initialized LangChain RAG chain
_outreach_rag_chain = None

def create_outreach_chain() -> Optional[RunnablePassthrough]:
    """
    Assembles and returns the LangChain RAG chain for generating personalized
    LinkedIn outreach messages. This chain combines retrieval from the
    vector store with LLM generation based on a prompt.
    """
    global _outreach_rag_chain

    if _outreach_rag_chain is not None:
        logger.info("LangChain RAG chain already created. Returning existing instance.")
        return _outreach_rag_chain

    logger.info("Initializing LangChain RAG chain components...")

    # Load components
    try:
        llm: HuggingFacePipeline = load_llm_model()
        retriever: Optional[VectorStoreRetriever] = get_retriever()
        prompt: ChatPromptTemplate = get_outreach_prompt_template()
    except Exception as e:
        logger.error(f"Failed to load one or more chain components: {e}")
        return None

    if llm is None:
        logger.error("LLM model not loaded. Cannot create outreach chain.")
        return None
    # Retriever can be None if knowledge base setup failed, we handle that gracefully

    # Define how to get the 'context' for the prompt
    # This uses the retriever to fetch relevant documents based on the input query
    def format_docs(docs):
        """Formats a list of Document objects into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    # --- Construct the RAG Chain ---
    # The chain expects a dictionary with all the placeholders required by the prompt
    # and a 'query' key for the retriever.
    # We use RunnablePassthrough to pass all input variables directly to the prompt,
    # and RunnableLambda to process the 'query' for the retriever.
    
    if retriever:
        # Full RAG chain with retrieval
        _outreach_rag_chain = (
            {
                "context": RunnableLambda(lambda x: x["query"]) | retriever | format_docs,
                "recipient_name": RunnablePassthrough(),
                "job_title": RunnablePassthrough(),
                "company_name": RunnablePassthrough(),
                "about_section": RunnablePassthrough(),
                "recent_activities": RunnablePassthrough(),
                "skills": RunnablePassthrough(),
                "query": RunnablePassthrough() # Pass query through for context processing
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("LangChain RAG chain with retrieval created.")
    else:
        # Fallback chain if retriever is not available (e.g., KB setup failed)
        logger.warning("Retriever not available. Creating LLM-only chain (without RAG).")
        _outreach_rag_chain = (
            {
                "context": RunnableLambda(lambda x: "No external knowledge base context available."), # Provide fallback context
                "recipient_name": RunnablePassthrough(),
                "job_title": RunnablePassthrough(),
                "company_name": RunnablePassthrough(),
                "about_section": RunnablePassthrough(),
                "recent_activities": RunnablePassthrough(),
                "skills": RunnablePassthrough(),
                "query": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.warning("Using fallback LLM-only chain.")

    return _outreach_rag_chain

# You can add a small test block here for direct testing during development
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Ensure logging is active
    print("--- Testing Chain Manager Module ---")

    # This test will try to load all underlying components (LLM, embeddings, ChromaDB)
    # So ensure all previous modules are configured correctly and your KB files exist.
    print("Attempting to create outreach chain...")
    chain = create_outreach_chain()

    if chain:
        print("\nOutreach chain created successfully.")

        # Prepare dummy input for testing the chain
        # These values will simulate the output from your scraper
        dummy_profile_data = {
            "recipient_name": "Jane Doe",
            "job_title": "Head of AI Research",
            "company_name": "Innovate AI Labs",
            "about_section": "Experienced AI leader focused on machine learning ethics and scalable solutions. Passionate about natural language processing and responsible AI deployment.",
            "recent_activities": "Recently posted about new breakthroughs in LLM efficiency and edge computing.",
            "skills": "Machine Learning, Deep Learning, Natural Language Processing, Python, TensorFlow, Ethics in AI",
            "query": "AI research leader interested in LLM efficiency and ethical deployment at Innovate AI Labs." # This is what the RAG will use
        }

        print("\nInvoking chain with dummy data...")
        try:
            generated_message = chain.invoke(dummy_profile_data)
            print("\n--- Generated Message ---")
            print(generated_message.strip())
            print("---")
        except Exception as e:
            print(f"Error invoking the chain: {e}")
            print("Please ensure your LLM model is loaded correctly and all components are functional.")

        # Test calling create_outreach_chain again (should return same instance)
        print("\nAttempting to get chain again (should be fast, using cached instance)...")
        chain_again = create_outreach_chain()
        assert chain is chain_again, "Chain instances are not the same, singleton failed!"
        print("Singleton pattern confirmed for chain.")

    else:
        print("\nFailed to create outreach chain. Check logs for errors in component loading.")