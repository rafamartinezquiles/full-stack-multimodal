# Import PyMuPDF (also known as fitz) for reading and extracting text from PDF files
import fitz  

# Import pathlib's Path class for working with file paths 
from pathlib import Path

# Define a function to extract all text content from a PDF file
def extract_text_from_pdf(pdf_path: str) -> str:
    # Open the PDF file using PyMuPDF
    doc = fitz.open(pdf_path)

    # Initialize a string to accumulate text from all pages
    text = ""

    # Iterate through each page in the PDF and extract its text
    for page in doc:
        text += page.get_text()

    # Strip leading/trailing whitespace from the full extracted text and return
    return text.strip()
