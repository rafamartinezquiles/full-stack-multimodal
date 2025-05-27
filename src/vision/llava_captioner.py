# Import the AutoProcessor and Llava model from Hugging Face's Transformers library
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Import PIL to handle image loading and processing
from PIL import Image

# Import PyTorch for tensor operations and model execution
import torch

# Define the model ID for LLaVA 1.5
model_id = "llava-hf/llava-1.5-7b-hf"

# Load the corresponding processor to handle both image and text input formatting
processor = AutoProcessor.from_pretrained(model_id)

# Load the LLaVA model with optional memory optimization for CPU use
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32,        
    low_cpu_mem_usage=True 
)

# Set the model to evaluation mode 
model.eval()

# Define a function that generates a detailed caption for a given image
def generate_caption(image_path: str) -> str:
    # Load the image and ensure it is in RGB format
    image = Image.open(image_path).convert("RGB")

    # Define the input prompt that instructs the model to describe the image
    prompt = "<image>\nDescribe this image in detail."

    # Preprocess the prompt and image, and move the resulting tensors to the model's device 
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)

    # Generate a response from the model with a token limit
    output = model.generate(**inputs, max_new_tokens=100)

    # Decode the output tokens to get the final caption string
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Return the generated caption
    return caption
