from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

prompt = PromptTemplate(
    template='Generate 5 Intresting  facts  about {topic}',
    input_variables=['topic']
)

# Initialize the model
chat = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=api_key # Your API key
)

parser = StrOutputParser()

chain = prompt | chat | parser
result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()

