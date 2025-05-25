from src.ingestion.pdf_loader import extract_text_from_pdf
from src.extraction.entity_extractor import extract_entities
import json
from dotenv import load_dotenv
import os

load_dotenv()  

def main():
    pdf_path = "data/employee_handbook.pdf"  
    text = extract_text_from_pdf(pdf_path)
    print("Extracted Text:\n", text[:500], "\n---")

    entities_json = extract_entities(text)
    print("Extracted Entities:\n", entities_json)

if __name__ == "__main__":
    main()
