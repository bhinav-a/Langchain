from langchain_huggingface import (ChatHuggingFace , HuggingFaceEndpoint)
from typing import TypedDict

model = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.7
)
chat = ChatHuggingFace(llm = model)

class review(TypedDict):
    summary : str
    sentiment : str

struct_out = chat.with_structured_output(review)
res = struct_out.invoke("""The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(res)