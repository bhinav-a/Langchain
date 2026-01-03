
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
from langchain_huggingface import(
    HuggingFaceEndpoint,
    ChatHuggingFace
)
model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    temperature=0.5
)

chat = ChatHuggingFace(llm=model)
chat_history = [
    SystemMessage(content="You are a helpful AI assistant")
]
while(True):
    user_input = input('You : ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = chat.invoke(chat_history)
    chat_history.append(AIMessage(content=result))
    print("Ai : " , result.content)

