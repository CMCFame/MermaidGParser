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
        'text': genai.GenerativeModel('gemini-pro'),
        'vision': genai.GenerativeModel('gemini-pro-vision')
    }
    return models

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_youtube_id(url):
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
    base_prompt = f"""You are a Mermaid diagram expert. Analyze the following {input_type} and create a Mermaid flowchart, following these strict rules:

    1. ALWAYS start with: flowchart TD

    2. Syntax Rules:
       - Use unique IDs for nodes (A1, B2, etc.)
       - Regular nodes: A["Text here"]
       - Decision nodes: A{{"Decision text"}}
       - Multiple arrows: A --> B & C & D
       - Conditional paths: A -- "condition" --> B
       - Replace newlines with spaces in node text
       - Quotes around all node text
       - Clean node text: remove special chars, replace _ with space

    3. Subgraph Syntax:
       subgraph name["Display Name"]
           content
       end

    4. Here's a complete example:
       ```mermaid
       flowchart TD
           subgraph process["Main Process"]
               A["Start Here"] --> B{{"Make Decision"}}
               B -- "Yes" --> C["Process A"] & D["Process B"]
               B -- "No" --> E["End Process"]
           end
           C --> F["Final Step"]
           D --> F
           E --> F
       ```

    5. Important Rules:
       - Replace _n or \n with spaces in text
       - Always use quotes around node text ["text"] and {{"text"}}
       - No special characters in node IDs
       - Multiple connections use & symbol
       - Keep node text readable and clean

    Convert this content: {context}

    Respond with ONLY the Mermaid code."""
    return base_prompt

def clean_mermaid_code(code: str) -> str:
    # Remove markdown blocks
    code = re.sub(r'^```mermaid\s*\n', '', code)
    code = re.sub(r'\n```$', '', code)
    
    # Ensure correct start
    if not code.strip().startswith('flowchart TD'):
        code = 'flowchart TD\n' + code
    
    # Fix common syntax issues
    code = re.sub(r'_n|\\n', ' ', code)  # Replace newline indicators with spaces
    code = re.sub(r'\s+', ' ', code)  # Normalize spaces
    
    # Fix node syntax
    code = re.sub(r'\[([^\]]+)\]', lambda m: f'["{m.group(1).strip()}"]', code)  # Add quotes to node text
    code = re.sub(r'\{\{([^\}]+)\}\}', lambda m: f'{{""{m.group(1).strip()}""}}', code)  # Fix decision nodes
    
    # Fix subgraph syntax
    code = re.sub(r'subgraph\s+([^\n\[]+)(?!\[)', lambda m: f'subgraph {m.group(1)}[""{m.group(1)}""]', code)
    
    # Clean up arrows and connections
    code = re.sub(r'\s*-->\s*', ' --> ', code)
    code = re.sub(r'\s*--\s*"([^"]+)"\s*-->\s*', ' -- "\\1" --> ', code)
    
    # Remove any remaining invalid characters
    code = re.sub(r'[^\w\s\{\}\[\]"\'_\-/>&;=,.]', ' ', code)
    
    return code

def display_mermaid_preview(mermaid_code: str):
    html = f"""
        <div class="mermaid">
        {mermaid_code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                flowchart: {{
                    curve: 'linear',
                    defaultRenderer: 'dagre-d3'
                }},
                securityLevel: 'loose'
            }});
        </script>
    """
    st.components.v1.html(html, height=600)

def main():
    st.title("Architecture Diagram Generator")
    st.write("Convert your architecture specifications into Mermaid diagrams")
    
    models = initialize_gemini()
    
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
    
    if mermaid_code:
        st.header("Generated Diagram")
        
        cleaned_code = clean_mermaid_code(mermaid_code)
        
        try:
            display_mermaid_preview(cleaned_code)
        except Exception as e:
            st.error(f"Error displaying preview: {str(e)}")
            st.error("Raw preview error - check syntax:")
            st.code(cleaned_code)
        
        with st.expander("Show Mermaid Code"):
            st.code(cleaned_code, language="mermaid")
            
        st.download_button(
            label="Download Mermaid Code",
            data=cleaned_code,
            file_name="diagram.mmd",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()