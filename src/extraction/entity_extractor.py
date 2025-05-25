from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
Extract all named entities (people, organizations, places, dates, and concepts) from the following text and return them in JSON format with this structure:

{{
  "entities": [
    {{
      "name": "...",
      "type": "Person | Organization | Location | Date | Concept"
    }},
    ...
  ]
}}

Text:
{text}
"""
)

def extract_entities(text: str) -> str:
    openai_api_key = os.getenv("OPENAI_API_KEY")  
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4",
        openai_api_key=openai_api_key
    )
    entity_chain = LLMChain(llm=llm, prompt=prompt_template)
    return entity_chain.run(text=text)
