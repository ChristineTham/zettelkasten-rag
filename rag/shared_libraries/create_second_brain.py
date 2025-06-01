# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.auth import default
import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Part
import os
from dotenv import load_dotenv, set_key
import requests
import tempfile
import PyPDF2 # For PDF text extraction
import uuid
import re # For parsing and naming

# Load environment variables from .env file
load_dotenv()

# --- Please fill in your configurations ---
# Retrieve the PROJECT_ID from the environmental variables.
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError(
        "GOOGLE_CLOUD_PROJECT environment variable not set. Please set it in your .env file."
    )
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
if not LOCATION:
    raise ValueError(
        "GOOGLE_CLOUD_LOCATION environment variable not set. Please set it in your .env file."
    )
CORPUS_DISPLAY_NAME = "Alphabet_10K_2024_corpus"
CORPUS_DESCRIPTION = "Corpus containing Alphabet's 10-K 2024 document"
PDF_URL = "https://abc.xyz/assets/77/51/9841ad5c4fbe85b4440c47a4df8d/goog-10-k-2024.pdf"
PDF_FILENAME = "goog-10-k-2024.pdf"
ENV_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Global variable for the Gemini model
GEMINI_MODEL = None
MODEL="gemini-2.5-pro-preview-05-06"

# --- Start of the script ---
def initialize_vertex_ai_and_gemini():
  """Initializes Vertex AI and the Gemini Pro model."""
  global GEMINI_MODEL
  credentials, _ = default()
  vertexai.init(
      project=PROJECT_ID, location=LOCATION, credentials=credentials
  )
  print("Vertex AI initialized.")
  try:
    GEMINI_MODEL = GenerativeModel(MODEL)
    print(f"Gemini model ({MODEL}) initialized successfully.")
  except Exception as e:
    print(f"Failed to initialize Gemini model: {e}")
    raise


def create_or_get_corpus():
  """Creates a new corpus or retrieves an existing one."""
  embedding_model_config = rag.EmbeddingModelConfig(
      publisher_model="publishers/google/models/text-embedding-004"
  )
  existing_corpora = rag.list_corpora()
  corpus = None
  for existing_corpus in existing_corpora:
    if existing_corpus.display_name == CORPUS_DISPLAY_NAME:
      corpus = existing_corpus
      print(f"Found existing corpus with display name '{CORPUS_DISPLAY_NAME}'")
      break
  if corpus is None:
    corpus = rag.create_corpus(
        display_name=CORPUS_DISPLAY_NAME,
        description=CORPUS_DESCRIPTION,
        embedding_model_config=embedding_model_config,
    )
    print(f"Created new corpus with display name '{CORPUS_DISPLAY_NAME}'")
  return corpus


def download_pdf_from_url(url, output_path):
  """Downloads a PDF file from the specified URL."""
  print(f"Downloading PDF from {url}...")
  response = requests.get(url, stream=True)
  response.raise_for_status()  # Raise an exception for HTTP errors
  
  with open(output_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
      f.write(chunk)
  
  print(f"PDF downloaded successfully to {output_path}")
  return output_path


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text content from a PDF file."""
    print(f"Extracting text from {pdf_path}...")
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        print(f"Text extracted successfully from {pdf_path} (approx {len(text)} characters).")
        return text
    except Exception as e:
        print(f"Could not read or parse PDF {pdf_path}: {e}")
        raise


def generate_zettelkasten_cards_from_text(text_content: str, source_pdf_name: str) -> list[str]:
    """Generates Zettelkasten cards from text using Gemini."""
    global GEMINI_MODEL
    if not GEMINI_MODEL:
        raise ValueError("Gemini model not initialized. Call initialize_vertex_ai_and_gemini() first.")

    print(f"Generating Zettelkasten cards for {source_pdf_name} using Gemini...")

    # Truncate text_content if it's too long to avoid hitting model limits.
    # gemini-1.5-flash-001 has a 1M token limit (input + output).
    # Roughly 4 chars/token. Let's cap input text to ~3MB chars to be safe.
    max_chars = 3000000 
    if len(text_content) > max_chars:
        print(f"Warning: Text content is very long ({len(text_content)} chars). Truncating to {max_chars} chars for Gemini.")
        text_content = text_content[:max_chars]

    prompt = f"""
You are an expert in knowledge extraction and summarization, tasked with creating Zettelkasten notes.
Given the following text from the document '{source_pdf_name}', please generate a set of Zettelkasten cards.

Each card MUST:
1. Capture a single, atomic idea, key point, or piece of information from the text.
2. Have a concise title as the first line, formatted as a Markdown H1 (e.g., '# Card Title').
3. Be written entirely in Markdown format.
4. Be self-contained and understandable on its own.
5. Focus on extracting factual information, key arguments, and important data.

Here is the text:
---
{text_content}
---

Please provide each Zettelkasten card as a separate Markdown block.
Use '---ZETTELKASTEN_CARD_SEPARATOR---' as a delimiter on its own line between individual cards.

Example of a single card:
# Example Card Title
This is the content of the Zettelkasten card, explaining a key concept or piece of information.
- Bullet point 1
- Bullet point 2

If the text is too short, lacks substance for Zettelkasten cards, or if you cannot process it, please return an empty response or a single message indicating this, instead of the separator.
Do not include any preamble or explanation before the first card or after the last card, other than the cards themselves and their separators.
"""

    try:
        response = GEMINI_MODEL.generate_content(
            [Part.from_text(prompt)],
            generation_config={"temperature": 0.2, "top_p": 0.95}
        )
        
        raw_response_text = response.text
        # print(f"Gemini raw response snippet: {raw_response_text[:500]}...")

        if "---ZETTELKASTEN_CARD_SEPARATOR---" in raw_response_text:
            cards = [card.strip() for card in raw_response_text.split("---ZETTELKASTEN_CARD_SEPARATOR---") if card.strip()]
        elif raw_response_text.strip().startswith("#"): # Assume single card if no separator but looks like a card
            cards = [raw_response_text.strip()]
        else:
            print(f"Gemini did not return cards in the expected format for {source_pdf_name}. Response: {raw_response_text}")
            cards = []
            
        print(f"Generated {len(cards)} Zettelkasten cards for {source_pdf_name}.")
        return cards
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Gemini API error details: {e.response}")
        raise


def upload_zettelkasten_notes_to_corpus(corpus_name: str, pdf_path: str, source_pdf_display_name: str):
  """
  Extracts text from the PDF, generates Zettelkasten cards using Gemini,
  and uploads these cards as Markdown files to the specified corpus.
  """
  print(f"Processing PDF {source_pdf_display_name} to generate and upload Zettelkasten cards...")

  try:
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text.strip():
        print(f"No text extracted from {source_pdf_display_name}. Skipping Zettelkasten generation.")
        return []
  except Exception as e:
    print(f"Failed to extract text from PDF {pdf_path}: {e}. Aborting Zettelkasten generation.")
    return []

  try:
    markdown_cards = generate_zettelkasten_cards_from_text(pdf_text, source_pdf_display_name)
    if not markdown_cards:
        print(f"No Zettelkasten cards generated by Gemini for {source_pdf_display_name}.")
        return []
  except Exception as e:
    print(f"Failed to generate Zettelkasten cards from Gemini: {e}. Aborting upload.")
    return []

  uploaded_rag_files = []
  with tempfile.TemporaryDirectory() as md_temp_dir:
    for i, card_content in enumerate(markdown_cards):
        if not card_content.strip():
            continue

        card_title_match = re.search(r"^#\s*(.*)", card_content, re.MULTILINE)
        if card_title_match:
            extracted_title = card_title_match.group(1).strip()
            # Sanitize title for filename and display name
            safe_title_segment = re.sub(r'[^\w\s-]', '', extracted_title).replace(' ', '_')[:50]
        else:
            extracted_title = f"Untitled_Card_{i+1}"
            safe_title_segment = extracted_title

        md_filename = f"Zettel_{safe_title_segment}_{uuid.uuid4().hex[:8]}.md"
        md_file_path = os.path.join(md_temp_dir, md_filename)

        try:
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(card_content)
            
            print(f"Uploading Markdown note: {md_filename} to corpus {corpus_name}...")
            rag_file = rag.upload_file(
                corpus_name=corpus_name,
                path=md_file_path,
                display_name=md_filename,
                description=f"Zettelkasten card: {extracted_title} (from {source_pdf_display_name})",
            )
            print(f"Successfully uploaded {md_filename} (RAG Name: {rag_file.name}) to corpus.")
            uploaded_rag_files.append(rag_file)
        except Exception as e:
            print(f"Error uploading Markdown file {md_filename} to corpus: {e}")

  if not uploaded_rag_files:
      print(f"No Markdown files were ultimately uploaded for {source_pdf_display_name}.")
  else:
      print(f"Successfully uploaded {len(uploaded_rag_files)} Zettelkasten cards from {source_pdf_display_name}.")
  return uploaded_rag_files


def update_env_file(corpus_name, env_file_path):
    """Updates the .env file with the corpus name."""
    try:
        set_key(env_file_path, "RAG_CORPUS", corpus_name)
        print(f"Updated RAG_CORPUS in {env_file_path} to {corpus_name}")
    except Exception as e:
        print(f"Error updating .env file: {e}")

def list_corpus_files(corpus_name):
  """Lists files in the specified corpus."""
  files = list(rag.list_files(corpus_name=corpus_name))
  print(f"Total files in corpus: {len(files)}")
  for file in files:
    print(f"File: {file.display_name} - {file.name}")


def main():
  initialize_vertex_ai_and_gemini()
  corpus = create_or_get_corpus()

  # Update the .env file with the corpus name
  update_env_file(corpus.name, ENV_FILE_PATH)
  
  # Create a temporary directory to store the downloaded PDF
  with tempfile.TemporaryDirectory() as temp_dir:
    pdf_path = os.path.join(temp_dir, PDF_FILENAME)
    
    # Download the PDF from the URL
    download_pdf_from_url(PDF_URL, pdf_path)
    
    # Process the PDF: generate Zettelkasten notes and upload them to the corpus
    upload_zettelkasten_notes_to_corpus(
        corpus_name=corpus.name,
        pdf_path=pdf_path,
        source_pdf_display_name=PDF_FILENAME
    )
  
  # List all files in the corpus
  list_corpus_files(corpus_name=corpus.name)

if __name__ == "__main__":
  main()
