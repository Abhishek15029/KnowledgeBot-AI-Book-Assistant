# If we want to see any website's particular detail and we have to search it every time so we want AI to help us in this.for ex- apple website show core features at bottom of their page   
from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/macbook-pro/"
# If we want to use multiple url so put it into list or apply loop for every url 
 
data = WebBaseLoader(url)

docs = data.load()

# print(len(docs))
print(docs[0].page_content)