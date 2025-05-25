from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

llm = ChatOpenAI(temperature=0, model="gpt-4")  # or "gpt-3.5-turbo"

entity_chain = LLMChain(llm=llm, prompt=prompt_template)

def extract_entities(text: str) -> dict:
    result = entity_chain.run(text=text)
    return result
