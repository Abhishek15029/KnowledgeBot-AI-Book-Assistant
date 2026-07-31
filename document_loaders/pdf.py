from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document_loaders/GRU.pdf")

docs = data.load()

# splitter = TokenTextSplitter(
#         chunk_size=100,
#         chunk_overlap=10,
# )

splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10,
)

chunks = splitter.split_documents(docs)

print(docs[2]) # len - 5 --> in this list 1 list contains 5 list of documents no. of pages is number of list means 1 page in 1 list for seein page no. 2 only so docs[2]
# print(len(chunks))