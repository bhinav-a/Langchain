from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

loader = DirectoryLoader(path='book',glob='*.pdf', loader_cls=PyPDFLoader)

docs = loader.load()

print(docs[200].page_content)