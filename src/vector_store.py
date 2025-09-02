# src/vector_store.py

import os
import logging
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever

from . import config # Import config
from .embeddings import load_embedding_model # Import the embedding model loader

# Set up logging for this module
logger = logging.getLogger(__name__)

_vectorstore_retriever: Optional[VectorStoreRetriever] = None

def setup_knowledge_base() -> Optional[VectorStoreRetriever]:
    """
    Loads text documents from a specified directory, splits them into chunks,
    and stores their embeddings in ChromaDB. Returns a retriever for the vector store.
    If the vector store already exists, it loads it.
    """
    global _vectorstore_retriever

    if _vectorstore_retriever is not None:
        logger.info("Knowledge base (ChromaDB) already set up. Returning existing retriever.")
        return _vectorstore_retriever

    logger.info(f"Setting up knowledge base in ChromaDB at: {config.CHROMA_DB_DIR}")
    logger.info(f"Loading documents from: {config.KNOWLEDGE_BASE_DOCS_PATH}")

    # Load the embedding model
    try:
        embeddings = load_embedding_model()
    except Exception as e:
        logger.error(f"Failed to load embedding model, cannot set up knowledge base: {e}")
        return None

    # Try to load existing ChromaDB
    try:
        # Check if the persist directory already contains data for the collection
        # This is a heuristic, Chroma.from_existing() or Chroma(persist_directory=...)
        # will handle the actual loading, but we want to avoid unnecessary document processing.
        if os.path.exists(config.CHROMA_DB_DIR) and any(f.endswith('.bin') or f.endswith('.pkl') for f in os.listdir(config.CHROMA_DB_DIR)):
            logger.info(f"Attempting to load existing ChromaDB from {config.CHROMA_DB_DIR}")
            vectordb = Chroma(
                persist_directory=config.CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_name=config.CHROMA_COLLECTION_NAME
            )
            # A simple check to see if the collection actually has data
            if vectordb._collection.count() > 0:
                logger.info(f"Successfully loaded existing ChromaDB with {vectordb._collection.count()} documents.")
                _vectorstore_retriever = vectordb.as_retriever(search_kwargs={"k": config.RAG_NUM_RESULTS})
                return _vectorstore_retriever
            else:
                logger.warning("Existing ChromaDB found but collection is empty. Rebuilding knowledge base.")
        else:
            logger.info("No existing ChromaDB found or it's empty. Proceeding to build new knowledge base.")

    except Exception as e:
        logger.warning(f"Error loading existing ChromaDB: {e}. Will attempt to rebuild the knowledge base.")

    # If no existing DB was loaded or it was empty, proceed to load documents and build new DB
    documents: List[Document] = []
    if not os.path.exists(config.KNOWLEDGE_BASE_DOCS_PATH) or not os.listdir(config.KNOWLEDGE_BASE_DOCS_PATH):
        logger.warning(f"No documents found in '{config.KNOWLEDGE_BASE_DOCS_PATH}'. Please add .txt files to populate the knowledge base.")
        # Create dummy files for initial setup if directory is empty
        _create_dummy_knowledge_base_files()

    for file_name in os.listdir(config.KNOWLEDGE_BASE_DOCS_PATH):
        if file_name.endswith(".txt"):
            file_path = os.path.join(config.KNOWLEDGE_BASE_DOCS_PATH, file_name)
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading document '{file_name}': {e}")

    if not documents:
        logger.error("No valid text documents were loaded. Cannot build a knowledge base.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Max size of each chunk
        chunk_overlap=50, # Overlap between chunks to maintain context
        length_function=len # Use character length for splitting
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"Loaded {len(documents)} raw documents, split into {len(texts)} chunks for embedding.")

    # Create and persist the new vector store
    try:
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=config.CHROMA_DB_DIR,
            collection_name=config.CHROMA_COLLECTION_NAME
        )
        vectordb.persist() # Explicitly persist the changes to disk
        logger.info("Knowledge base setup complete and persisted successfully.")
        _vectorstore_retriever = vectordb.as_retriever(search_kwargs={"k": config.RAG_NUM_RESULTS})
        return _vectorstore_retriever
    except Exception as e:
        logger.error(f"Failed to create or persist ChromaDB: {e}")
        return None

def get_retriever() -> Optional[VectorStoreRetriever]:
    """
    Returns the initialized ChromaDB retriever. If not yet set up, it calls setup_knowledge_base.
    """
    if _vectorstore_retriever is None:
        return setup_knowledge_base()
    return _vectorstore_retriever

def _create_dummy_knowledge_base_files():
    """Helper function to create dummy files if the knowledge base directory is empty."""
    logger.info("Creating dummy knowledge base files for initial setup...")
    os.makedirs(config.KNOWLEDGE_BASE_DOCS_PATH, exist_ok=True)
    with open(os.path.join(config.KNOWLEDGE_BASE_DOCS_PATH, "value_prop.txt"), "w") as f:
        f.write("Our cutting-edge platform boosts B2B sales efficiency by automating lead qualification and providing predictive analytics. We help companies reduce their sales cycle by up to 25% and increase conversion rates by leveraging AI.")
    with open(os.path.join(config.KNOWLEDGE_BASE_DOCS_PATH, "cto_benefits.txt"), "w") as f:
        f.write("For Chief Technology Officers, our solution offers robust, scalable architecture built on cloud-native principles. It features end-to-end encryption, seamless API integration, and adheres to industry-leading security standards (ISO 27001, SOC 2 Type II). We prioritize developer experience and minimize technical debt.")
    with open(os.path.join(config.KNOWLEDGE_BASE_DOCS_PATH, "marketing_manager_pain_points.txt"), "w") as f:
        f.write("Marketing Managers frequently struggle with accurately measuring campaign ROI and generating high-quality marketing qualified leads (MQLs). Our system provides real-time, granular campaign performance insights and employs advanced segmentation to deliver a consistent flow of high-intent MQLs, directly impacting revenue growth.")
    logger.info("Dummy files created. Please replace these with your actual company knowledge.")


# You can add a small test block here for direct testing during development
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Ensure logging is active
    print("--- Testing Vector Store Module ---")
    print(f"Chroma DB Directory: {config.CHROMA_DB_DIR}")
    print(f"Knowledge Base Docs Path: {config.KNOWLEDGE_BASE_DOCS_PATH}")

    # Ensure dummy files exist if the directory is empty
    if not os.path.exists(config.KNOWLEDGE_BASE_DOCS_PATH) or not os.listdir(config.KNOWLEDGE_BASE_DOCS_PATH):
        _create_dummy_knowledge_base_files()

    # Get the retriever (will set up if not already done)
    retriever_instance = get_retriever()

    if retriever_instance:
        print("\nRetriever obtained successfully. Testing retrieval...")

        test_query = "How can a Marketing Manager improve lead quality?"
        print(f"Querying for: '{test_query}'")
        retrieved_docs = retriever_instance.invoke(test_query)

        print(f"\n--- Retrieved {len(retrieved_docs)} Documents ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"Document {i+1} (Score: {doc.metadata.get('score', 'N/A')}):") # Chroma doesn't always provide score directly from .invoke
            print(doc.page_content)
            print("---")

        print("\nTesting retrieval with another query...")
        test_query_cto = "What are the security features for a CTO?"
        print(f"Querying for: '{test_query_cto}'")
        retrieved_docs_cto = retriever_instance.invoke(test_query_cto)
        print(f"\n--- Retrieved {len(retrieved_docs_cto)} Documents ---")
        for i, doc in enumerate(retrieved_docs_cto):
            print(f"Document {i+1}:\n{doc.page_content}\n---")

        # Test calling get_retriever again (should return same instance)
        print("\nCalling get_retriever again (should be fast, using cached instance)...")
        another_retriever_instance = get_retriever()
        assert retriever_instance is another_retriever_instance, "Retriever instances are not the same, singleton failed!"
        print("Singleton pattern confirmed for retriever.")

    else:
        print("\nFailed to obtain retriever. Check logs for errors during knowledge base setup.")