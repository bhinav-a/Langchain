from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.messages import HumanMessage

# HF API endpoint (NO local model download)
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    temperature=0.5,
    max_new_tokens=256
)

# Chat model (correct package)
chat = ChatHuggingFace(llm=endpoint)

response = chat.invoke([
    HumanMessage(content="Capital of India?")
])

print(response.content)
