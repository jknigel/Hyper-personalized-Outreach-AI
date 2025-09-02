# Hyper-personalized-Outreach-AI
Hyper-personalize your outreach messages. Get a good first impression!

Structuring your application files properly is crucial for maintainability, scalability, and collaboration. For an application like this, a modular approach will work best.

Here's a recommended file structure, broken down by logical components:

```
linkedin_outreach_ai/
├── data/
│   ├── knowledge_base_docs/
│   │   ├── value_prop.txt
│   │   ├── cto_benefits.txt
│   │   └── marketing_manager_pain_points.txt
│   └── persona_templates/
│       ├── sales_leader.json
│       └── recruiter.json
│
├── models/
│   ├── huggingface_models/  # Cached Hugging Face models (downloaded here)
│   └── chroma_db/           # Persistent ChromaDB data
│
├── src/
│   ├── __init__.py          # Makes 'src' a Python package
│   ├── config.py            # Global configurations (API keys, paths, model names)
│   ├── scraper.py           # Logic for interacting with the MCP server
│   ├── embeddings.py        # Embedding model initialization
│   ├── vector_store.py      # ChromaDB setup and retrieval logic
│   ├── llm_model.py         # LLM model initialization
│   ├── prompt_manager.py    # Manages prompt templates
│   ├── chain_manager.py     # Assembles LangChain components into a chain
│   └── main.py              # Main application logic and Gradio interface
│
├── utils/
│   ├── __init__.py
│   └── data_preprocessing.py # Utility functions for cleaning scraped data (optional for now)
│
├── .env                     # Environment variables (e.g., OPENAI_API_KEY, if you switch back)
├── .gitignore               # Files/directories to ignore in Git
├── requirements.txt         # Python dependencies
├── run_mcp_server.sh        # Script to easily start the MCP server (optional)
├── README.md                # Project description and setup instructions
```

Let's break down each part:

### `linkedin_outreach_ai/` (Root Directory)

This is the main project folder.

### `data/`

*   **`knowledge_base_docs/`**: This directory holds your raw text files that form your custom knowledge base for RAG. Each `.txt` file could represent a different aspect of your product, service, or target persona insights.
*   **`persona_templates/` (Optional but good for future growth)**: If you evolve to have very specific message structures or tone guidelines for different personas that go beyond what the LLM can infer, you might store JSON or YAML files here.

### `models/`

*   **`huggingface_models/`**: This is where Hugging Face will cache the downloaded embedding and LLM models. It's good to keep this organized.
*   **`chroma_db/`**: This is where ChromaDB will store its persistent vector store data.

### `src/` (Source Code)

This is where all your Python application logic lives. Making it a package (`__init__.py`) is good practice.

*   **`config.py`**:
    *   Store all your configurable parameters here:
        *   `MCP_SERVER_URL = "http://localhost:5000"`
        *   `EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"`
        *   `CHROMA_DB_DIR = "../models/chroma_db"` (Paths relative to `src/`)
        *   `LLM_MODEL_ID = "microsoft/phi-2"`
        *   `KNOWLEDGE_BASE_PATH = "../data/knowledge_base_docs/"`
        *   `MAX_NEW_TOKENS = 200`, `TEMPERATURE = 0.7`, etc.
    *   This makes it easy to change settings without digging through code.

*   **`scraper.py`**:
    *   Contains the `get_linkedin_profile_data` function.
    *   Handles all direct interaction with the `linkedin-mcp-server`.
    *   Might include error handling specific to the scraper.

*   **`embeddings.py`**:
    *   Initializes and returns the `HuggingFaceEmbeddings` object.
    *   `load_embedding_model()` function.

*   **`vector_store.py`**:
    *   Contains the `setup_knowledge_base` function.
    *   Handles loading documents, splitting them, and creating/persisting the `Chroma` vector store.
    *   Provides a function to get the `retriever` object.

*   **`llm_model.py`**:
    *   Initializes and returns the `HuggingFacePipeline` wrapped `HuggingFaceLLM` object.
    *   `load_llm_model()` function.
    *   Manages model loading, tokenizer, pipeline setup.

*   **`prompt_manager.py`**:
    *   Defines your `ChatPromptTemplate`.
    *   `get_outreach_prompt()` function.
    *   Centralizes all prompt engineering.

*   **`chain_manager.py`**:
    *   Assembles the LangChain components: prompt, LLM, retriever.
    *   `create_outreach_chain(llm, retriever)` function.
    *   This is where your RAG chain is defined.

*   **`main.py`**:
    *   This is the entry point for your application.
    *   Imports functions/objects from other `src` modules.
    *   Contains the `generate_message_from_profile` function (or similar orchestrator).
    *   Sets up and launches the Gradio interface.
    *   Includes the main `if __name__ == "__main__":` block for execution.

### `utils/`

*   **`data_preprocessing.py`**:
    *   (Optional for now, but useful) Functions to clean and standardize data received from the MCP server.
    *   E.g., `normalize_job_title(title)`, `extract_key_phrases(text)`.

### Root Level Files

*   **`.env`**: (Highly recommended) Stores sensitive information like API keys (if you used OpenAI or a paid scraper). *Never commit this to Git.*
*   **`.gitignore`**: Specifies files and directories that Git should ignore (e.g., `__pycache__/`, `*.pyc`, `.env`, `models/huggingface_models/`, `models/chroma_db/`, virtual environment folders).
*   **`requirements.txt`**: Lists all Python packages your project depends on. Generate it with `pip freeze > requirements.txt`.
*   **`run_mcp_server.sh` (or `.bat` for Windows)**: A simple script to remind you/others how to start the MCP server.
    ```bash
    #!/bin/bash
    cd linkedin-mcp-server
    python main.py
    ```
*   **`README.md`**: Essential for explaining what your project is, how to set it up, how to run it, and any dependencies (like the MCP server).

### Benefits of this Structure:

*   **Modularity:** Each component has its own file, making it easier to understand, test, and debug.
*   **Separation of Concerns:** Each file has a clear responsibility (e.g., `scraper.py` only scrapes, `vector_store.py` only deals with Chroma).
*   **Readability:** Easier for new developers to onboard and understand the project flow.
*   **Maintainability:** Changes to the LLM don't require changes to the scraper, for example.
*   **Scalability:** As your application grows, you can add new modules or features without cluttering existing files.
*   **Testability:** Individual modules can be unit-tested more easily.

This structure will set you up for success as you build and potentially expand your AI-powered outreach system!