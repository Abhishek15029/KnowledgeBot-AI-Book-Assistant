import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


@st.cache_resource
def load_embedding_model():
    return MistralAIEmbeddings()


@st.cache_resource
def load_llm():
    return ChatMistralAI(model="mistral-small-2603")


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
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            MAX_CHUNKS = 1000

            if len(chunks) > MAX_CHUNKS:
                st.warning(
                    f"Large PDF detected ({len(chunks)} chunks).\n"
                    f"Only first {MAX_CHUNKS} chunks will be indexed."
                )
                chunks = chunks[:MAX_CHUNKS]

            st.info(f"Chunks Created: {len(chunks)}")

            embeddings = load_embedding_model()

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 20,
                    "lambda_mult": 0.7
                }
            )

            st.success("Knowledge Base Created Successfully!")

if st.session_state.retriever is not None:

    question = st.text_input(
        "Ask a question about the book"
    )

    if st.button("Ask"):

        if question.strip() == "":
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Searching..."):

            retrieved_docs = st.session_state.retriever.invoke(question)

        st.info(f"Retrieved Chunks: {len(retrieved_docs)}")

        if len(retrieved_docs) == 0:
            st.error("No relevant information found.")
            st.stop()

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is partially available,
answer with the available information.

If the answer is not present in the context,
reply exactly:

'I could not find the answer in the document.'
"""
                ),
                (
                    "human",
                    """
Context:
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

        llm = load_llm()

        with st.spinner("Generating Answer..."):
            response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved Context"):
            st.write(context)