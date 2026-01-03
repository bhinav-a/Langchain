import streamlit as st
from langchain_huggingface import(
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate , load_prompt
 #static prompt


endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    temperature=0.5
)
chat = ChatHuggingFace(llm = endpoint)
st.header("Reseacrh Tool")
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')



if st.button("summarize"):
    chain = template | chat
    prompt = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})
    # result = chat.invoke(prompt)       # no need of this because we are using 'chain' in line 28 . chain help to combine invoking template and model together . therfore we only have to invoke once .
    st.write(prompt.content)






