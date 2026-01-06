from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

chat = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=api_key # Your API key
)
chat2 = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=api_key # Your API key
)
prompt1 = PromptTemplate(
    template='Generate a short notes on the following text -> {text}',
    input_variables=['text']
)
promt2 = PromptTemplate(
    template='Generate 5 questions on the following notes -> {text}',
    input_variables=['text']
)
promt3 = PromptTemplate(
    template='Combine these both in one document , notes -> {notes} , quiz -> {quiz}',
    input_variables=['notes' , 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1|chat|parser,
    'quiz' : promt2|chat2|parser
})

merge_chain = promt3 | chat | parser

chain = parallel_chain | merge_chain

text = """
Models

Copy page

LLMs are powerful AI tools that can interpret and generate text like humans. They’re versatile enough to write content, translate languages, summarize, and answer questions without needing specialized training for each task.
In addition to text generation, many models support:
 Tool calling - calling external tools (like databases queries or API calls) and use results in their responses.
 Structured output - where the model’s response is constrained to follow a defined format.
 Multimodality - process and return data other than text, such as images, audio, and video.
 Reasoning - models perform multi-step reasoning to arrive at a conclusion.
Models are the reasoning engine of agents. They drive the agent’s decision-making process, determining which tools to call, how to interpret results, and when to provide a final answer.
The quality and capabilities of the model you choose directly impact your agent’s baseline reliability and performance. Different models excel at different tasks - some are better at following complex instructions, others at structured reasoning, and some support larger context windows for handling more information.
LangChain’s standard model interfaces give you access to many different provider integrations, which makes it easy to experiment with and switch between models to find the best fit for your use case.
For provider-specific integration information and capabilities, see the provider’s chat model page.
​
Basic usage
Models can be utilized in two ways:
With agents - Models can be dynamically specified when creating an agent.
Standalone - Models can be called directly (outside of the agent loop) for tasks like text generation, classification, or extraction without the need for an agent framework.
The same model interface works in both contexts, which gives you the flexibility to start simple and scale up to more complex agent-based workflows as needed.
​
Initialize a model
The easiest way to get started with a standalone model in LangChain is to use init_chat_model to initialize one from a chat model provider of your choice (examples below):
OpenAI
Anthropic
Azure
Google Gemini
AWS Bedrock
HuggingFace

"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()