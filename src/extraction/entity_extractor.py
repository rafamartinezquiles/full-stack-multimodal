# Import the OpenAI-powered LLM wrapper from LangChain
from langchain_openai import ChatOpenAI

# Import tools for defining prompts and LLM chains
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Used to access environment variables 
import os

# Define a structured prompt template to instruct the LLM on the extraction task
# It asks to extract named entities and return them in a specific JSON format
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
Extract all named entities (people, organizations, places, dates, and concepts) from the following text and return them in JSON format with this structure:

{{"entities": [
  {{"name": "...", "type": "Person | Organization | Location | Date | Concept"}},
  ...
]}}

Text:
{text}
"""
)

# Define the entity extraction function using the OpenAI model
def extract_entities(text: str) -> str:
    # Retrieve OpenAI API key from environment variable 
    openai_api_key = os.getenv("OPENAI_API_KEY")  
    
    # Initialize the ChatOpenAI model (GPT-4) with no randomness (temperature=0)
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        openai_api_key=openai_api_key
    )
    
    # Wrap the LLM and prompt template into a callable chain
    entity_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Run the chain with the given text and return the output string 
    return entity_chain.run({"text": text})