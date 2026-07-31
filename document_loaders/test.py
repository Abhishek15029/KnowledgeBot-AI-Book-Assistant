# Langchain changes things regularly so you have to go to documents also  
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
        separator="",
        chunk_size=10,
        chunk_overlap=1
)
data = TextLoader("document_loaders/notes.txt") 

docs = data.load()
# how we want to split 
chunks = splitter.split_documents(docs)

# print(docs[0].page_content)
print(len(chunks))
# Every chunk's metadata also created

for i in chunks:
        print(i.page_content)
        print()
        print()
        print()