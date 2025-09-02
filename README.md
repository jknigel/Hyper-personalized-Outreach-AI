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