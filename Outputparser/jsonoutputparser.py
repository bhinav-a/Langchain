
from langchain_huggingface import(
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    temperature=0.5
)

chat = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()
template1 = PromptTemplate(
    template="Give me the name  , age and city of a fictional person \n {format_ins}",
    input_variables=[],
    partial_variables={'format_ins':parser.get_format_instructions()}
)
chain = template1 | chat | parser
res = chain.invoke({})
print(res)

# disadvantage of jsonOutputParser is that we cannot determine the structure of the json.
# it does not enforce schema
