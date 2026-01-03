from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage  
from dotenv import load_dotenv
from typing import TypedDict
import os
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Initialize the model
chat = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=api_key # Your API key
)

# Create a message
messages = [
    HumanMessage(content="""Software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")
]

class review(TypedDict):
    summary : str
    sentiment : str

structured_output = chat.with_structured_output(review)


# Generate a response
response = structured_output.invoke(messages)

# Print the answer
print(response)
print(response['summary'])
print(response['sentiment'])
