# src/scraper.py

import requests
import json
import logging
from . import config # Import config from the same package (src)

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_linkedin_profile_data(profile_url: str) -> dict:
    """
    Retrieves LinkedIn profile data from the local MCP server.

    Args:
        profile_url (str): The URL of the LinkedIn profile to scrape.

    Returns:
        dict: A dictionary containing the scraped profile data, or an error message
              if the scraping fails.
    """
    mcp_server_full_url = f"{config.MCP_SERVER_URL}{config.MCP_SCRAPE_ENDPOINT}"
    payload = {"url": profile_url}
    timeout_seconds = 180 # Generous timeout, scraping can take a while

    logger.info(f"Attempting to scrape profile: {profile_url} using MCP server at {mcp_server_full_url}")

    try:
        response = requests.post(
            mcp_server_full_url,
            json=payload,
            timeout=timeout_seconds
        )
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        profile_data = response.json()
        logger.info(f"Successfully received data for {profile_url}")

        # Basic validation: Check if essential fields are present
        if not profile_data or "name" not in profile_data:
            logger.warning(f"Scraped data for {profile_url} seems incomplete: {profile_data.keys() if profile_data else 'empty'}")
            return {"error": "Scraped data is incomplete or empty.", "raw_response": profile_data}

        return profile_data

    except requests.exceptions.Timeout:
        logger.error(f"MCP server request timed out after {timeout_seconds} seconds for {profile_url}")
        return {"error": f"Scraping timed out after {timeout_seconds} seconds. Is the MCP server running and responsive?", "profile_url": profile_url}
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to MCP server at {config.MCP_SERVER_URL}. Is the server running?")
        return {"error": f"Could not connect to MCP server at {config.MCP_SERVER_URL}. Please ensure it is running.", "profile_url": profile_url}
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request error with MCP server for {profile_url}: {e}")
        return {"error": f"Failed to scrape profile due to HTTP error: {e}", "profile_url": profile_url}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from MCP server for {profile_url}: {e}. Response text: {response.text}")
        return {"error": f"Invalid JSON response from MCP server: {e}", "profile_url": profile_url, "raw_response": response.text}
    except Exception as e:
        logger.exception(f"An unexpected error occurred during scraping for {profile_url}: {e}")
        return {"error": f"An unexpected error occurred: {e}", "profile_url": profile_url}

# You can add a small test block here for direct testing during development
if __name__ == "__main__":
    print("--- Testing Scraper Module ---")
    print(f"MCP Server URL: {config.MCP_SERVER_URL}{config.MCP_SCRAPE_ENDPOINT}")

    # Ensure your MCP server is running before running this test!
    test_profile_url = "https://www.linkedin.com/in/williamhgates/" # Replace with a test profile

    if config.MCP_SERVER_URL == "http://localhost:5000":
        print("\nNOTE: Ensure the 'linkedin-mcp-server' is running in a separate terminal.")
        print(f"It should be accessible at {config.MCP_SERVER_URL}")
        print("To run the MCP server, navigate to its directory and execute 'python main.py'.")
        input("Press Enter to continue with scraping test...")

    profile_info = get_linkedin_profile_data(test_profile_url)

    if "error" in profile_info:
        print(f"\nError: {profile_info['error']}")
        if "raw_response" in profile_info:
            print(f"Raw Response Part: {profile_info['raw_response']}")
    else:
        print("\n--- Scraped Profile Data (Partial) ---")
        print(f"Name: {profile_info.get('name', 'N/A')}")
        print(f"Headline: {profile_info.get('headline', 'N/A')}")
        print(f"Company: {profile_info.get('experience', [{}])[0].get('companyName', 'N/A') if profile_info.get('experience') else 'N/A'}")
        print(f"Summary (About): {profile_info.get('summary', 'N/A')[:200]}...")
        print(f"Skills (first 5): {profile_info.get('skills', [])[:5]}")
        print(f"Posts count: {len(profile_info.get('posts', []))}")
        # You'll need to inspect the MCP server's actual JSON output to know all keys.
        # This example assumes common keys based on typical LinkedIn scraping.