import streamlit as st
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai.types.generation_types import GenerationConfig
from typing import Union, List
import os

# Configure page settings
st.set_page_config(
    page_title="Architecture Diagram Generator",
    layout="wide"
)

# Initialize Gemini API
def initialize_gemini():
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    return model

def get_gemini_response(
    model,
    contents: Union[str, List],
    generation_config: GenerationConfig = GenerationConfig(
        temperature=0.1,
        max_output_tokens=2048
    ),
    stream: bool = True
) -> str:
    """Generate a response from the Gemini model."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    
    try:
        responses = model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=stream,
        )
        
        if not stream:
            return responses.text
        
        final_response = []
        for r in responses:
            try:
                final_response.append(r.text)
            except IndexError:
                final_response.append("")
                continue
        return " ".join(final_response)
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return ""

def create_prompt(context: str) -> str:
    """Create the prompt for diagram generation."""
    return f"""Context: {context}
    You are an experienced Google Cloud Architect who specializes in creating architectural diagrams using Mermaid diagramming tool.
    Follow these steps to analyze and create the diagram:

    1. Initial Analysis:
       - Identify the starting components and entry points
       - Note the overall flow direction (TD = top-down, LR = left-right)
       - Identify all sections or service categories
       - Document all relationships and dependencies

    2. Component Identification:
       - List every service mentioned in the specifications
       - Categorize each service by its type:
         * Compute (VMs, Functions, Containers)
         * Storage (Databases, Object Storage)
         * Networking (Load Balancers, CDN)
         * Security (IAM, Firewalls)
         * Analytics & ML
         * Other specialized services
       - Map services to their appropriate GCP products

    3. Connection Analysis:
       - Document all service interactions
       - Note data flow directions
       - Identify any specific protocols or methods
       - Mark high-availability or redundancy patterns

    Based on this analysis:
    1. Create a Google Cloud Architecture Diagram using Mermaid.js syntax
    2. Use flowchart TD or LR based on complexity
    3. Enclose the diagram inside a box labeled GCP
    4. Group services by their categories (Network, Components, Databases)
    5. Show clear connections between services
    6. Do not use classDef, Style, or CSS tags in output
    7. For service categorization examples:
       - Compute Engine → VM
       - Cloud SQL, CloudSpanner → Database
       - Cloud CDN → Network
       - Cloud Functions → Serverless

    Ensure the connection order follows: Network → Components → Databases
    Make the diagram clear, logical, and easy to follow."""

def main():
    st.title("Architecture Diagram Generator")
    st.write("Convert your architecture specifications into Mermaid diagrams")
    
    # Initialize Gemini model
    model = initialize_gemini()
    
    # Input section
    st.header("Input")
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload"]
    )
    
    context = ""
    if input_method == "Text Input":
        context = st.text_area(
            "Enter your architecture specifications:",
            height=200
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload your specifications file",
            type=["txt", "md"]
        )
        if uploaded_file:
            context = uploaded_file.getvalue().decode("utf-8")
    
    # Generate button
    if st.button("Generate Diagram") and context:
        with st.spinner("Generating diagram..."):
            # Create prompt and get response
            prompt = create_prompt(context)
            response = get_gemini_response(
                model,
                prompt,
                GenerationConfig(temperature=0, max_output_tokens=2048)
            )
            
            # Display the Mermaid diagram
            if response:
                st.header("Generated Diagram")
                st.markdown(f"```mermaid\n{response}\n```")
                
                # Show the raw Mermaid code
                with st.expander("Show Mermaid Code"):
                    st.code(response, language="mermaid")

if __name__ == "__main__":
    main()