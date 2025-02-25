import streamlit as st
import pandas as pd
import requests
from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List
from composio_phidata import Action, ComposioToolSet
import json
import os

# Define Schema for Extracted Data
class QuoraUserInteractionSchema(BaseModel):
    username: str = Field(description="Username of the person who posted")
    bio: str = Field(description="User's bio")
    post_type: str = Field(description="Type of post ('question' or 'answer')")
    timestamp: str = Field(description="Time of posting")
    upvotes: int = Field(default=0, description="Number of upvotes")
    links: List[str] = Field(default_factory=list, description="Any links in the post")

class QuoraPageSchema(BaseModel):
    interactions: List[QuoraUserInteractionSchema] = Field(description="List of user interactions on Quora")

# Function to Search for Relevant Quora Links
def search_for_urls(company_description: str, firecrawl_api_key: str, num_links: int) -> List[str]:
    url = "https://api.firecrawl.dev/v1/search"
    headers = {"Authorization": f"Bearer {firecrawl_api_key}", "Content-Type": "application/json"}
    query = f"Quora discussions about {company_description}"
    payload = {"query": query, "limit": num_links, "lang": "en", "location": "United States", "timeout": 60000}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200 and response.json().get("success"):
        return [result["url"] for result in response.json().get("data", [])]
    return []

# Function to Extract User Information
def extract_user_info_from_urls(urls: List[str], firecrawl_api_key: str) -> List[dict]:
    firecrawl_app = FirecrawlApp(api_key=firecrawl_api_key)
    extracted_data = []

    for url in urls:
        response = firecrawl_app.extract(
            [url],
            {'prompt': 'Extract usernames, bios, post types, timestamps, upvotes, and links.', 
             'schema': QuoraPageSchema.model_json_schema()}
        )
        if response.get('success') and response.get('status') == 'completed':
            extracted_data.append({"website_url": url, "user_info": response.get('data', {}).get('interactions', [])})
    
    return extracted_data

# Function to Flatten Data for CSV and Google Sheets
def format_user_info_to_flattened_json(extracted_data: List[dict]) -> List[dict]:
    flattened_data = []
    for data in extracted_data:
        website_url = data["website_url"]
        for interaction in data["user_info"]:
            flattened_data.append({
                "Website URL": website_url,
                "Username": interaction.get("username", ""),
                "Bio": interaction.get("bio", ""),
                "Post Type": interaction.get("post_type", ""),
                "Timestamp": interaction.get("timestamp", ""),
                "Upvotes": interaction.get("upvotes", 0),
                "Links": ", ".join(interaction.get("links", [])),
            })
    return flattened_data

# Function to Save Extracted Data to CSV
def save_to_csv(flattened_data: List[dict]) -> str:
    df = pd.DataFrame(flattened_data)
    csv_filename = "quora_leads.csv"
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    return csv_filename

# Function to Write Data to Google Sheets
def write_to_google_sheets(flattened_data: List[dict], composio_api_key: str, openai_api_key: str) -> str:
    composio_toolset = ComposioToolSet(api_key=composio_api_key)
    google_sheets_tool = composio_toolset.get_tools(actions=[Action.GOOGLESHEETS_SHEET_FROM_JSON])[0]
    
    google_sheets_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key),
        tools=[google_sheets_tool],
        show_tool_calls=True
    )
    
    message = "Create a Google Sheet with these columns: Website URL, Username, Bio, Post Type, Timestamp, Upvotes, Links.\n\n"
    message += json.dumps(flattened_data, indent=2)
    
    try:
        response = google_sheets_agent.run(message)
        if "https://docs.google.com/spreadsheets/d/" in response.content:
            return response.content.strip()
    except Exception:
        pass
    return None

# Streamlit App
def main():
    st.title("🎯 AI Lead Generation Agent")
    st.info("This agent finds Quora leads related to your business and saves them in a CSV or Google Sheet.")

    # Sidebar for API Keys
    with st.sidebar:
        st.header("🔑 API Keys")
        firecrawl_api_key = st.text_input("Firecrawl API Key", type="password")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        composio_api_key = st.text_input("Composio API Key", type="password")
        num_links = st.number_input("Number of links to search", min_value=1, max_value=10, value=3)
        if st.button("Reset"):
            st.session_state.clear()
            st.experimental_rerun()

    # User Input for Search
    user_query = st.text_area("Describe the leads you need:", 
                              placeholder="e.g., AI chatbots for customer support",
                              help="Be specific for better results.")

    if st.button("Generate Leads"):
        if not all([firecrawl_api_key, openai_api_key, composio_api_key, user_query]):
            st.error("Please enter all API keys and a search query.")
        else:
            with st.spinner("Transforming query..."):
                transformed_query = f"Find Quora discussions about {user_query}"
                st.write("🎯 Searching for:", transformed_query)

            with st.spinner("Searching for Quora links..."):
                urls = search_for_urls(transformed_query, firecrawl_api_key, num_links)

            if urls:
                st.subheader("🔗 Quora Links Used:")
                for url in urls:
                    st.write(url)

                with st.spinner("Extracting user data..."):
                    extracted_data = extract_user_info_from_urls(urls, firecrawl_api_key)

                with st.spinner("Formatting data..."):
                    flattened_data = format_user_info_to_flattened_json(extracted_data)

                with st.spinner("Saving to CSV..."):
                    csv_filename = save_to_csv(flattened_data)

                st.success("Data saved successfully!")

                # Provide Download Option for CSV
                st.subheader("📂 Download Extracted Data")
                st.download_button(
                    label="Download CSV",
                    data=open(csv_filename, "rb"),
                    file_name=csv_filename,
                    mime="text/csv"
                )

                with st.spinner("Uploading to Google Sheets..."):
                    google_sheets_link = write_to_google_sheets(flattened_data, composio_api_key, openai_api_key)

                if google_sheets_link:
                    st.success("Lead data successfully uploaded to Google Sheets!")
                    st.subheader("🔗 Google Sheets Link:")
                    st.markdown(f"[View Google Sheet]({google_sheets_link})")
                else:
                    st.error("❌ Failed to create Google Sheet.")
            else:
                st.warning("No relevant Quora links found.")

if __name__ == "__main__":
    main()
