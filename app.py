import streamlit as st
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai.types.generation_types import GenerationConfig
from typing import Union, List
import os
import tempfile
from PIL import Image
import PyPDF2
import re
import base64

# Configure page settings
st.set_page_config(
    page_title="Architecture Diagram Generator",
    layout="wide"
)

# Initialize Gemini API
def initialize_gemini():
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    # Use gemini-pro for text and gemini-pro-vision for images
    models = {
        'text': genai.GenerativeModel('gemini-pro'),
        'vision': genai.GenerativeModel('gemini-pro-vision')
    }
    return models

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_youtube_id(url):
    """Extract YouTube video ID from URL."""
    youtube_regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(youtube_regex, url)
    return match.group(1) if match else None

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

def create_prompt(context: str, input_type: str) -> str:
    """Create the prompt based on input type."""
    base_prompt = f"""Analyze the following {input_type} and create a Mermaid diagram that represents the architecture:

    Follow these steps:
    1. Initial Analysis:
       - Identify key components and their relationships
       - Determine the flow direction (TD or LR)
       - Identify system boundaries and categories
    
    2. Component Identification:
       - List all services/components
       - Categorize them by type (Compute, Storage, Network, etc.)
       - Map to appropriate cloud services if applicable
    
    3. Connection Analysis:
       - Document all interactions
       - Note data flow directions
       - Identify protocols or methods
    
    Create a Mermaid diagram that:
    1. Uses clear, logical structure
    2. Shows all components and relationships
    3. Uses appropriate Mermaid syntax
    4. Groups related components
    5. Shows clear direction of flow
    
    Context for analysis: {context}
    
    Respond with ONLY the Mermaid code, no explanations."""
    return base_prompt

def display_mermaid_preview(mermaid_code: str):
    """Display a preview of the Mermaid diagram."""
    # Create HTML for Mermaid preview
    html = f"""
        <div class="mermaid">
        {mermaid_code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    """
    st.components.v1.html(html, height=600)

def main():
    st.title("Architecture Diagram Generator")
    st.write("Convert your architecture specifications into Mermaid diagrams")
    
    # Initialize Gemini models
    models = initialize_gemini()
    
    # Input section
    st.header("Input")
    input_type = st.selectbox(
        "Choose input type:",
        ["Text", "Image", "PDF", "YouTube URL"]
    )
    
    mermaid_code = None
    
    if input_type == "Text":
        context = st.text_area("Enter your architecture specifications:", height=200)
        if st.button("Generate Diagram") and context:
            with st.spinner("Generating diagram..."):
                prompt = create_prompt(context, "text")
                mermaid_code = get_gemini_response(models['text'], prompt)
                
    elif input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file and st.button("Generate Diagram"):
            with st.spinner("Analyzing image..."):
                image = Image.open(uploaded_file)
                prompt = create_prompt("", "image")
                mermaid_code = get_gemini_response(models['vision'], [prompt, image])
                
    elif input_type == "PDF":
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file and st.button("Generate Diagram"):
            with st.spinner("Analyzing PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                prompt = create_prompt(text, "PDF document")
                mermaid_code = get_gemini_response(models['text'], prompt)
                
    elif input_type == "YouTube URL":
        url = st.text_input("Enter YouTube URL")
        if url and st.button("Generate Diagram"):
            with st.spinner("Processing video..."):
                video_id = extract_youtube_id(url)
                if video_id:
                    prompt = create_prompt(f"YouTube video: {url}", "video")
                    mermaid_code = get_gemini_response(models['text'], prompt)
                else:
                    st.error("Invalid YouTube URL")
    
    # Display results
    if mermaid_code:
        st.header("Generated Diagram")
        
        # Show the Mermaid preview
        try:
            display_mermaid_preview(mermaid_code)
        except Exception as e:
            st.error(f"Error displaying preview: {str(e)}")
        
        # Show the raw Mermaid code
        with st.expander("Show Mermaid Code"):
            st.code(mermaid_code, language="mermaid")
            
        # Add download button for the Mermaid code
        st.download_button(
            label="Download Mermaid Code",
            data=mermaid_code,
            file_name="diagram.mmd",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()