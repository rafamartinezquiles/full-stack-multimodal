from PIL import Image
import pytesseract
import os

# Optional: set path to tesseract executable (only needed on Windows)
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = "your_path"

def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()
