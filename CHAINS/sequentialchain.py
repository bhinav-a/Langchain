from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

chat = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=api_key # Your API key
)

prompt1 = PromptTemplate(
    template='Generate a summary on the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Summarize the following text in 5 points {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt1 | chat |parser| prompt2 | chat |parser

result = chain.invoke({'text' : 'Inflation'})

print(result)

chain.get_graph().print_ascii()