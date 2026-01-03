from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage , HumanMessage

chat_temp = ChatPromptTemplate([
    ('system' , "You are a helpful {domain} expert" ),
    ('human' , "Explain in simple terms , what is {topic}")
    
    # we use above ^ method to make dymanic messages 
    
    
    
    # these methods will not work :

    # SystemMessage(content="You are a helpful {domain} expert"),
    # HumanMessage(content = "Explain in simple terms , what is {topic}")



])

prompt = chat_temp.invoke({'domain' : 'cricketer' , 'topic': 'Dusra'})

print(prompt)
