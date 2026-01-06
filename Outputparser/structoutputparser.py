
from langchain_huggingface import(
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    temperature=0.5
)

chat = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact1' , description = 'Fact 1 about the topic'),
    ResponseSchema(name='fact2' , description = 'Fact 2 about the topic'),
    ResponseSchema(name='fact3' , description = 'Fact 3 about the topic'),
]
parser = StructuredOutputParser.from_response_schemas(schema)
template = PromptTemplate(
    template='Give three fact about {topic} \n{fomat_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}

)
prompt = template.invoke({'topic' : 'Black Hole'})
res = chat.invoke(prompt)
print(res)