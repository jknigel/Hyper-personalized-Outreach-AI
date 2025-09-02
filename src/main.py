# src/main.py

import gradio as gr
import logging
from typing import Optional

# Import all necessary components from our src modules
from . import config
from .scraper import get_linkedin_profile_data
from .chain_manager import create_outreach_chain
from .embeddings import load_embedding_model # Imported to ensure early loading/caching
from .vector_store import setup_knowledge_base # Imported to ensure early loading/caching
from .llm_model import load_llm_model # Imported to ensure early loading/caching

# Configure logging for the entire application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Application State ---
# These will hold the initialized chain and components
# We explicitly load them early to give feedback during startup
_outreach_chain = None

def initialize_application_components():
    """
    Initializes all necessary application components (LLM, Embeddings, ChromaDB, LangChain chain).
    This function should be called once at startup.
    """
    global _outreach_chain

    logger.info("--- Initializing Application Components ---")

    # Load Embedding Model (will also cache it)
    try:
        load_embedding_model()
        logger.info("Embedding model initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        # Application might still run without RAG if LLM is strong enough, but warn heavily.

    # Setup Knowledge Base (will load/create ChromaDB and its retriever)
    try:
        setup_knowledge_base() # This implicitly calls load_embedding_model again, but it's cached
        logger.info("Knowledge base setup complete.")
    except Exception as e:
        logger.error(f"Failed to setup knowledge base: {e}")
        # The chain_manager will handle retriever being None

    # Load LLM Model (will also cache it)
    try:
        load_llm_model()
        logger.info("LLM model initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize LLM model: {e}")
        logger.critical("Application cannot run without a functional LLM. Exiting.")
        # In a real app, you might exit here, or provide a minimal UI.
        # For Gradio, we'll return None and handle it in the UI startup.
        return False

    # Create LangChain Outreach Chain
    _outreach_chain = create_outreach_chain()
    if _outreach_chain is None:
        logger.critical("Failed to create LangChain outreach chain. Application cannot proceed.")
        return False

    logger.info("--- All components initialized successfully ---")
    return True

def generate_personalized_message_logic(linkedin_url: str) -> str:
    """
    Main logic function to generate a personalized outreach message
    given a LinkedIn profile URL. This function interacts with the scraper,
    preprocesses data, and invokes the LangChain RAG chain.

    Args:
        linkedin_url (str): The URL of the LinkedIn profile.

    Returns:
        str: The generated personalized message or an error message.
    """
    if not _outreach_chain:
        return "Application not fully initialized. Please check server logs for errors."

    logger.info(f"Received request to generate message for: {linkedin_url}")

    # 1. Scrape LinkedIn Profile Data
    profile_data = get_linkedin_profile_data(linkedin_url)

    if "error" in profile_data:
        logger.error(f"Scraping failed for {linkedin_url}: {profile_data['error']}")
        return f"Error scraping LinkedIn profile: {profile_data['error']}\nEnsure MCP server is running and accessible."

    # 2. Extract and Preprocess Key Profile Information
    # Adjust these keys based on the actual output of your MCP server
    recipient_name = profile_data.get("name", "there")
    job_title = profile_data.get("headline", "a professional") # 'headline' often contains job title
    company_name = "their company"
    if profile_data.get("experience"):
        # Assuming the first experience entry is the current one
        current_experience = profile_data["experience"][0]
        company_name = current_experience.get("companyName", "their company")

    about_section = profile_data.get("summary", "no specific 'About' section.") # 'summary' is common for the About section

    # Extract recent activities/posts - this will be highly dependent on MCP server output
    recent_activities_list = []
    if profile_data.get('posts'):
        recent_activities_list.extend([p.get('title', p.get('text', '')) for p in profile_data['posts'] if p.get('title') or p.get('text')])
    if profile_data.get('articles'):
        recent_activities_list.extend([a.get('title', a.get('text', '')) for a in profile_data['articles'] if a.get('title') or a.get('text')])
    recent_activities = ", ".join(filter(None, recent_activities_list[:3])) # Take top 3, filter empty strings
    if not recent_activities:
        recent_activities = "None mentioned."

    # Extract skills
    skills_list = [skill.get('name') for skill in profile_data.get('skills', []) if skill.get('name')]
    skills = ", ".join(skills_list[:5]) # Take top 5 skills
    if not skills:
        skills = "None listed."

    logger.info(f"Extracted info for {recipient_name} at {company_name} ({job_title})")

    # 3. Prepare Input for LangChain RAG Chain
    # The 'query' field is specifically for the RAG retriever
    rag_query = f"LinkedIn profile for {recipient_name}, {job_title} at {company_name}. Interested in {about_section}. Recent activities: {recent_activities}. Key skills: {skills}."

    chain_input = {
        "query": rag_query,
        "recipient_name": recipient_name,
        "job_title": job_title,
        "company_name": company_name,
        "about_section": about_section,
        "recent_activities": recent_activities,
        "skills": skills
    }

    # 4. Invoke the LangChain RAG Chain
    try:
        logger.info("Invoking LangChain outreach chain...")
        generated_message = _outreach_chain.invoke(chain_input)
        logger.info("Message generation complete.")
        return generated_message
    except Exception as e:
        logger.exception(f"Error during message generation for {linkedin_url}")
        return f"An error occurred during message generation: {e}"

# --- Gradio User Interface Setup ---
def launch_gradio_interface():
    """
    Sets up and launches the Gradio web interface for the application.
    """
    logger.info("Setting up Gradio interface.")

    iface = gr.Interface(
        fn=generate_personalized_message_logic,
        inputs=gr.Textbox(
            label="LinkedIn Profile URL",
            placeholder="e.g., https://www.linkedin.com/in/johndoe/",
            lines=1
        ),
        outputs=gr.Textbox(
            label="Generated Outreach Message",
            lines=10,
            show_copy_button=True
        ),
        title="✨ Hyper-Personalized LinkedIn Outreach AI ✨",
        description=(
            "Enter a LinkedIn profile URL and let the AI generate a personalized connection request message. "
            "**Important:** Ensure your `linkedin-mcp-server` is running in a separate terminal before starting this application!"
        ),
        allow_flagging="never", # Disable Gradio's default data flagging feature
    )

    logger.info("Launching Gradio interface...")
    iface.launch(
        server_name="0.0.0.0", # Make accessible from network if needed (use 127.0.0.1 for local only)
        server_port=7860,
        share=False # Set to True to get a public link (for demos, be cautious with sensitive data)
    )
    logger.info("Gradio interface shut down.")

# --- Main Application Entry Point ---
if __name__ == "__main__":
    if initialize_application_components():
        launch_gradio_interface()
    else:
        logger.critical("Application failed to initialize. Gradio interface will not be launched.")
        print("\nFATAL ERROR: Application components could not be initialized. Please check logs above.")
        print("Common issues: MCP server not running, insufficient RAM for LLM, or incorrect config.")