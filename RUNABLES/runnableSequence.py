from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
import os
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

chat = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=api_key # Your API key
)


prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)


parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, chat, parser, prompt2, chat, parser)

print(chain.invoke({'topic':'AI'}))