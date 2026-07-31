import tempfile
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="KnowledgeBot - AI Book Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Book Assistant")
st.write("Upload a PDF and ask questions about it.")

if "retriever" not in st.session_state:
    st.session_state.retriever = None

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Create Knowledge Base"):

        with st.spinner("Processing PDF..."):

            temp_dir = tempfile.mkdtemp()

            pdf_path = Path(temp_dir) / uploaded_file.name

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = MistralAIEmbeddings()

            db_path = "temp_chroma"

            shutil.rmtree(db_path, ignore_errors=True)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_path
            )

            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )

            st.success("Knowledge Base Created!")

if st.session_state.retriever:

    question = st.text_input(
        "Ask a question about the book"
    )

    if st.button("Ask") and question:

        retriever = st.session_state.retriever

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.

Use ONLY the provided context.

If the answer is not found,
reply:
'I could not find the answer in the document.'
"""
                ),
                (
                    "human",
                    """Context:
{context}

Question:
{question}
"""
                )
            ]
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        llm = ChatMistralAI(
            model="mistral-small-2603"
        )

        with st.spinner("Thinking..."):
            response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved Context"):
            st.write(context)