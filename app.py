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

def initialize_gemini():
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    models = {
        'text': genai.GenerativeModel('gemini-1.5-flash'),
        'vision': genai.GenerativeModel('gemini-1.5-flash')
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

def handle_youtube_url(url: str, model) -> str:
    """Process YouTube URL and generate diagram."""
    # Extract video ID
    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
        
    # Create specific prompt for video content
    video_prompt = f"""Analyze this YouTube video and create a clear flowchart diagram.
    URL: {url}
    
    Create a Mermaid flowchart following these exact rules:
    1. Use ONLY 'flowchart TD' as the first line
    2. Each node should be concise and clear
    3. Use proper Mermaid syntax:
       - Nodes: A["Text"]
       - Decisions: A{{"Question"}}
       - Connections: A --> B
       - Conditions: A -- "Yes" --> B
    
    Example format:
    flowchart TD
        A["Start"] --> B{{"Decision"}}
        B -- "Yes" --> C["Action"]
        B -- "No" --> D["Alternative"]
        C --> E["End"]
        D --> E
    
    Convert the video content into this format, focusing on the main flow and key decision points."""
    
    # Get response from model
    response = get_gemini_response(model, video_prompt, 
                                 GenerationConfig(temperature=0.2, max_output_tokens=2048))
    
    # Clean response
    cleaned_response = clean_mermaid_code(response)
    
    return cleaned_response

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
    base_prompt = f"""You are a Mermaid diagram expert. Analyze the following {input_type} and create a Mermaid flowchart that EXACTLY represents the content, following these strict rules:

    1. Always start with ONLY: flowchart TD
    2. For line breaks in node text, use ` (backtick) NOT \n or _n
    3. Follow these syntax rules:
       - Node text must be in quotes: A["Text here"]
       - Decision nodes use double curly braces: A{{"Decision"}}
       - Use & for multiple connections: A --> B & C
       - Conditions use quotes: A -- "condition" --> B
       - Subgraphs must have quoted names: subgraph "Name"

    Example:
    flowchart TD
        subgraph "Main Process"
            A["First line`Second line`Third line"] --> B{{"Decision"}}
            B -- "Yes" --> C["Process"] & D["Next Step"]
            B -- "No" --> E["End"]
        end

    Rules for text formatting:
    - Replace any \n or _n with ` for line breaks
    - Keep original text but remove special characters
    - Maintain exact spacing in node text
    - Use quotes for all node text and subgraph names

    Here's the content to convert: {context}

    Generate ONLY the flowchart code, no markdown blocks or explanations."""
    return base_prompt

def clean_mermaid_code(code: str) -> str:
    """Clean and validate Mermaid code."""
    # Remove any content before flowchart TD
    if 'flowchart TD' in code:
        code = code[code.find('flowchart TD'):]
    
    # Remove markdown and mermaid tags
    code = re.sub(r'```.*?```', '', code, flags=re.DOTALL)
    code = code.replace('```mermaid', '').replace('```', '')
    code = code.replace('mermaid', '')
    
    # Ensure single flowchart TD
    code = re.sub(r'flowchart TD\s+flowchart TD', 'flowchart TD', code)
    code = code.strip()
    
    if not code.startswith('flowchart TD'):
        code = 'flowchart TD\n' + code
    
    # Process line by line
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    cleaned_lines = []
    
    for line in lines:
        # Skip duplicate headers
        if line == 'flowchart TD' and cleaned_lines and cleaned_lines[0] == 'flowchart TD':
            continue
            
        # Clean up node syntax
        line = re.sub(r'\s+\[', '[', line)
        line = re.sub(r'\[([^\]"]+)\]', lambda m: f'["{m.group(1).strip()}"]', line)
        line = re.sub(r'\{\{([^}]+)\}\}', lambda m: f'{{{{{m.group(1).strip()}}}}}', line)
        
        # Fix arrows and conditions
        line = re.sub(r'\s*-->\s*', ' --> ', line)
        line = re.sub(r'\s*--\s*"([^"]+)"\s*-->\s*', ' -- "\\1" --> ', line)
        
        # Remove semicolons at line ends
        line = line.rstrip(';')
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def display_mermaid_preview(mermaid_code: str):
    """Display a preview of the Mermaid diagram."""
    html = f"""
        <div class="mermaid">
        {mermaid_code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                flowchart: {{
                    curve: 'linear',
                    nodeSpacing: 50,
                    rankSpacing: 50,
                    padding: 10,
                    htmlLabels: true
                }},
                securityLevel: 'loose'
            }});
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
                try:
                    mermaid_code = handle_youtube_url(url, models['text'])
                    if not mermaid_code:
                        st.error("Could not generate diagram from video")
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
    
    # Display results
    if mermaid_code:
        st.header("Generated Diagram")
        
        # Clean and validate the code
        cleaned_code = clean_mermaid_code(mermaid_code)
        
        # Show the Mermaid preview
        try:
            display_mermaid_preview(cleaned_code)
        except Exception as e:
            st.error(f"Error displaying preview: {str(e)}")
            st.error("Raw preview error - check syntax:")
            st.code(cleaned_code)
        
        # Show the raw Mermaid code
        with st.expander("Show Mermaid Code"):
            st.code(cleaned_code, language="mermaid")
            
        # Add download button for the Mermaid code
        st.download_button(
            label="Download Mermaid Code",
            data=cleaned_code,
            file_name="diagram.mmd",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()