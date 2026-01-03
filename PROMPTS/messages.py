#There are three type of messages :
# 1. system messages
# 2. Human messages
# 3. AI messages ##

from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_huggingface import(
    HuggingFaceEndpoint,
    ChatHuggingFace
)

endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    temperature=0.5
)
 
model = ChatHuggingFace(llm = endpoint)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me About langchain")
]
result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
