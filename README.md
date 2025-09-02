# ✨ Hyper-Personalized LinkedIn Outreach AI ✨

## Overview

This project develops an AI-powered system that generates hyper-personalized LinkedIn outreach messages. By leveraging a local LinkedIn profile scraper, a custom knowledge base (RAG), and a locally-runnable Large Language Model (LLM), it crafts unique messages tailored to the recipient's profile and your strategic messaging. A Gradio interface provides a simple way to use the system.

**Key Features:**

1.  **LinkedIn Profile Scraping:** Connects to a local `linkedin-mcp-server` instance to extract detailed profile information (job title, company, 'About' section, recent activities, skills).
2.  **Retrieval Augmented Generation (RAG):** Integrates with a custom knowledge base (stored in ChromaDB) to retrieve relevant company-specific value propositions, pain points, and solutions.
3.  **Local LLM Generation:** Utilizes a Hugging Face Large Language Model (configurable, e.g., `microsoft/phi-2`) to generate messages locally, ensuring privacy and control.
4.  **Hyper-Personalization:** Crafts messages that are highly relevant to the recipient's persona (job title, company) and augmented with insights from your knowledge base.
5.  **Gradio User Interface:** Provides an intuitive web-based interface for easy interaction.
6.  **LangChain Integration:** Orchestrates the entire pipeline using the LangChain framework.

## Project Structure
linkedin_outreach_ai/
├── data/
│ ├── knowledge_base_docs/ # Your custom knowledge base (.txt files)
│ └── persona_templates/ # (Optional) Future persona-specific message structures
│
├── models/
│ ├── huggingface_models/ # Cache for Hugging Face LLMs and embedding models
│ └── chroma_db/ # Persistent storage for ChromaDB vector store
│
├── src/
│ ├── config.py # Global configurations and settings
│ ├── scraper.py # Interfaces with the MCP server for LinkedIn data
│ ├── embeddings.py # Loads and manages the Hugging Face embedding model
│ ├── vector_store.py # Sets up ChromaDB and provides retrieval functions
│ ├── llm_model.py # Loads and manages the local Hugging Face LLM
│ ├── prompt_manager.py # Defines the LangChain ChatPromptTemplate
│ ├── chain_manager.py # Assembles the LangChain RAG pipeline
│ └── main.py # Main application logic and Gradio interface
│
├── utils/
│ └── data_preprocessing.py # (Optional) Utility functions for data cleaning
│
├── .env # Environment variables (e.g., custom MCP URL)
├── .gitignore # Files/directories to ignore in Git
├── requirements.txt # Python dependencies
├── run_mcp_server.sh # Convenience script to start the MCP server
└── README.md # This file
code
Code
## Setup and Installation

### Prerequisites

1.  **Python 3.9+:** Ensure you have a compatible Python version.
2.  **`linkedin-mcp-server`:** This project relies on a separate local server to scrape LinkedIn profiles.
    *   **Clone the repository:**
        ```bash
        git clone https://github.com/stickerdaniel/linkedin-mcp-server.git
        ```
    *   **Follow its setup instructions:** This usually involves installing its `requirements.txt` and `playwright`.
        ```bash
        cd linkedin-mcp-server
        pip install -r requirements.txt
        playwright install
        ```
    *   **Keep this repository separate from `linkedin_outreach_ai`**.

### Project Installation

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/your-username/linkedin-outreach-ai.git # Replace with your repo URL
    cd linkedin-outreach-ai
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configure Your Knowledge Base

1.  Navigate to the `data/knowledge_base_docs/` directory within this project.
2.  Add your own `.txt` files containing information relevant to your outreach:
    *   Company value propositions
    *   Customer success stories
    *   Common pain points for different job titles/industries
    *   Product features and benefits
    *   Any other text you want the AI to "know" when generating messages.
    *   *Note: Dummy files will be created automatically if this directory is empty on first run.*

### Configuration Settings

Review and modify `src/config.py` as needed. Key parameters include:

*   `MCP_SERVER_URL`: The address where your `linkedin-mcp-server` is running (default: `http://localhost:5000`).
*   `LLM_MODEL_ID`: The Hugging Face model to use. `microsoft/phi-2` is a good default for local CPU. Consider `HuggingFaceH4/zephyr-7b-beta` for better quality if you have enough RAM (16GB+) or a GPU.
*   `LLM_MAX_NEW_TOKENS`, `LLM_TEMPERATURE`, `LLM_TOP_P`: Parameters to control LLM generation.
*   `CHROMA_DB_DIR`, `HUGGINGFACE_CACHE_DIR`: Paths for persistent data and model caching.

You can also override `MCP_SERVER_URL` or `LLM_MODEL_ID` using environment variables if you prefer.

## How to Run

There are two main components that need to be running: the MCP server and this application.

### 1. Start the LinkedIn MCP Server

Open a **separate terminal** window.

```bash
# Assuming you are in the linkedin_outreach_ai/ directory, navigate to the MCP server's directory
cd ../linkedin-mcp-server # Adjust path if needed
python main.py
Leave this terminal open and the server running. You should see messages indicating the server has started.
2. Start the Hyper-Personalized Outreach AI Application
Open a new terminal window, navigate back to the linkedin_outreach_ai/ project root, and ensure your virtual environment is active.
code
Bash
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
python -m src.main
The application will first initialize all its components (downloading models, setting up ChromaDB – this might take a few minutes on the first run). Once initialized, it will launch the Gradio interface and provide a local URL (e.g., http://127.0.0.1:7860).
3. Use the Gradio Interface
Open your web browser and navigate to the URL provided by the application.
Paste a LinkedIn profile URL into the input box.
Click "Submit" to generate a personalized outreach message.
Important Notes & Troubleshooting
MCP Server: This is a crucial dependency. If scraping fails, first check if your linkedin-mcp-server is running correctly and accessible at the configured MCP_SERVER_URL. The MCP server might require you to manually log into LinkedIn through the browser it controls on its first run.
LLM Memory: Large Language Models require significant RAM (e.g., phi-2 needs ~4GB, zephyr-7b-beta needs ~16GB). If the application fails to load the LLM, you might see "CUDA out of memory" (if using GPU) or system slowdowns. Consider using a smaller LLM_MODEL_ID in src/config.py if you have limited RAM.
First Run: The first time you run the application, it will download the embedding model and the LLM from Hugging Face. This can take some time depending on your internet connection. Models are cached in models/huggingface_models/.
Data Accuracy: The quality of the generated messages heavily depends on two factors:
MCP Server Output: The accuracy and completeness of the data scraped by the linkedin-mcp-server.
Your Knowledge Base: The quality and relevance of the .txt files in data/knowledge_base_docs/.
LangChain Expression Language (LCEL): The LangChain pipeline is built using LCEL, which provides a powerful and flexible way to compose chains.
Logging: Check the terminal where you run python -m src.main for detailed logs (INFO, WARNING, ERROR, CRITICAL) which can help in troubleshooting.
Future Enhancements (Ideas)
More Robust Scraper: Implement retry logic or alternative scraping methods.
Persona-Specific RAG: Integrate different knowledge bases or retrieval strategies based on the identified persona.
Message Templates: Allow users to define their own high-level message templates.
Fine-tuning: Fine-tune a smaller LLM for even more domain-specific message generation.
CRM Integration: Connect to CRM systems to automatically log outreach.
Feedback Loop: Implement a mechanism to collect user feedback on message quality to improve the system.
Rate Limiting: Add safeguards to prevent over-scraping LinkedIn or hitting LLM API rate limits (if using external LLMs).