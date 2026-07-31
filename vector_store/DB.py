from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="chroma-db"
)

result = vectorstore.similarity_search("what is used for data analysis?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)
    
retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain Deep Learning")

for d in docs:
    print(d.page_content)