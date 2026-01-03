from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-pro",  # pick from your list
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

response = llm.invoke("Explain LangChain in simple terms")
print(response.content)
