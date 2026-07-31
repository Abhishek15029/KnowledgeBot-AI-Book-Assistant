#load pdf
# split into chunks
# create the embeddings 
# store into chroma 
from langchain_community.document_loaders import PyPDFLoader #loads pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter #it creates chunking
from langchain_mistralai import MistralAIEmbeddings #it generates embeddings
from langchain_community.vectorstores import Chroma #store using chromadb 
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document_loaders/DL_book.pdf") 
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
  chunk_size = 1000,
  chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
)