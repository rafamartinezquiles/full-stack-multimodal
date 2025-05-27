# Import the Python Imaging Library to load and process images
from PIL import Image

# Import Tesseract OCR engine wrapper for text extraction
import pytesseract

# Import os for platform detection and path configuration
import os

# Optional: set the path to the Tesseract executable (required on Windows systems)
# Adjust this path if Tesseract is installed elsewhere on the user's system
if os.name == 'nt':  # Check if the OS is Windows
    pytesseract.pytesseract.tesseract_cmd = "your_path"

# Define a function to extract text content from an image file
def extract_text_from_image(image_path: str) -> str:
    # Open the image file using PIL
    image = Image.open(image_path)
    
    # Use Tesseract OCR to extract any visible text from the image
    text = pytesseract.image_to_string(image)
    
    # Strip any leading/trailing whitespace and return the result
    return text.strip()
