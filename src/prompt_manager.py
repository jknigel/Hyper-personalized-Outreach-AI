# src/prompt_manager.py

import logging
from langchain.prompts import ChatPromptTemplate
from . import config # Import config from the same package (src)

logger = logging.getLogger(__name__)

# Cache the prompt template
_outreach_prompt_template: ChatPromptTemplate = None

def get_outreach_prompt_template() -> ChatPromptTemplate:
    """
    Returns the LangChain ChatPromptTemplate for generating LinkedIn outreach messages.
    Ensures the template is created only once (singleton pattern).
    """
    global _outreach_prompt_template

    if _outreach_prompt_template is None:
        logger.info("Creating LinkedIn outreach prompt template.")

        # Define the system message, providing context and instructions to the LLM
        system_message = f"""You are an expert LinkedIn outreach specialist. Your goal is to write a highly personalized, concise, and value-driven connection request message for a professional on LinkedIn.

Use the provided LinkedIn profile information and relevant knowledge base context to craft the message.

Instructions:
1.  **Personalization:** Address the recipient by name. Mention something specific from their profile (job title, company, 'About' section, a recent post/activity, or a key skill) to show you've done your research.
2.  **Value Proposition:** Briefly and subtly connect their role/company to a potential value proposition or solution from the provided knowledge base context. Focus on their pain points, goals, or industry challenges if the context allows.
3.  **Conciseness:** Keep the message to {config.MESSAGE_LENGTH_GUIDELINE}.
4.  **Call to Action:** End with a {config.CALL_TO_ACTION_GUIDELINE}.
5.  **Tone:** Professional, friendly, and respectful. Avoid overly salesy or aggressive language. Focus on genuine connection and potential mutual interest/value.
6.  **Fallback:** If no specific profile details for personalization are found or no relevant knowledge is retrieved, write a more general but still professional and friendly message.
"""

        # Define the user message, which will contain the dynamic data
        user_message = """LinkedIn Profile Information:
Name: {recipient_name}
Job Title: {job_title}
Company: {company_name}
About Section: {about_section}
Recent Posts/Activities: {recent_activities}
Skills: {skills}

Relevant Knowledge Base Information:
{context}

Please generate the personalized LinkedIn connection message:"""

        _outreach_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", user_message)
            ]
        )
        logger.info("Prompt template created successfully.")

    return _outreach_prompt_template

# You can add a small test block here for direct testing during development
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Ensure logging is active
    print("--- Testing Prompt Manager Module ---")

    # Get the prompt template
    prompt_template = get_outreach_prompt_template()
    print("\nPrompt template loaded (or retrieved from cache).")

    # Verify a few key instructions are present
    system_content = prompt_template.messages[0].content
    user_content = prompt_template.messages[1].content

    print("\n--- System Message (partial) ---")
    print(system_content[:500] + "...") # Print first 500 chars
    assert config.MESSAGE_LENGTH_GUIDELINE in system_content
    assert config.CALL_TO_ACTION_GUIDELINE in system_content
    print("System message contains length and CTA guidelines.")

    print("\n--- User Message Placeholders ---")
    expected_placeholders = ["recipient_name", "job_title", "company_name",
                             "about_section", "recent_activities", "skills", "context"]
    for placeholder in expected_placeholders:
        assert f"{{{placeholder}}}" in user_content, f"Placeholder '{placeholder}' missing from user message."
        print(f"- Found: {{{placeholder}}}")

    # Test calling get_outreach_prompt_template again (should return same instance)
    print("\nAttempting to get prompt template again (should be fast, using cached instance)...")
    prompt_template_again = get_outreach_prompt_template()
    print("Prompt template retrieved again (from cache).")
    assert prompt_template is prompt_template_again, "Prompt templates are not the same instance, singleton failed!"
    print("Singleton pattern confirmed.")

    print("\nPrompt Manager Module Test Passed.")