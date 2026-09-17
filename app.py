import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="KnowledgeBot - AI Book Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 KnowledgeBot - AI Book Assistant")
st.write("Upload a PDF and ask questions about it.")


# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )

    return MistralAIEmbeddings(
        api_key=api_key
    )


# --------------------------------------------------
# Load LLM
# --------------------------------------------------

@st.cache_resource
def load_llm():

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY is not configured. "
            "Please add it to Streamlit Secrets."
        )

    return ChatMistralAI(
        model="mistral-small-2603",
        api_key=api_key,
        temperature=0.2,
        max_retries=2
    )


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# --------------------------------------------------
# Create Knowledge Base
# --------------------------------------------------

if uploaded_file is not None:

    if st.button("Create Knowledge Base"):

        with st.spinner("Processing PDF..."):

            try:

                # Temporary directory
                temp_dir = tempfile.mkdtemp()

                pdf_path = Path(temp_dir) / uploaded_file.name

                # Save uploaded PDF
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # -------------------------------
                # Load PDF
                # -------------------------------

                loader = PyPDFLoader(str(pdf_path))

                documents = loader.load()

                st.info(
                    f"Pages Loaded: {len(documents)}"
                )

                # -------------------------------
                # Split Documents
                # -------------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(
                    documents
                )

                # -------------------------------
                # Limit chunks
                # -------------------------------

                MAX_CHUNKS = 1000

                if len(chunks) > MAX_CHUNKS:

                    st.warning(
                        f"Large PDF detected "
                        f"({len(chunks)} chunks). "
                        f"Only the first {MAX_CHUNKS} "
                        f"chunks will be indexed."
                    )

                    chunks = chunks[:MAX_CHUNKS]

                st.info(
                    f"Chunks Created: {len(chunks)}"
                )

                # -------------------------------
                # Create Embeddings
                # -------------------------------

                embeddings = load_embedding_model()

                # -------------------------------
                # Create Chroma Vector Database
                # -------------------------------

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )

                # -------------------------------
                # Create Retriever
                # -------------------------------

                st.session_state.retriever = (
                    vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 6,
                            "fetch_k": 20,
                            "lambda_mult": 0.7
                        }
                    )
                )

                st.success(
                    "✅ Knowledge Base Created Successfully!"
                )

            except Exception as e:

                st.error(
                    "❌ Error while creating Knowledge Base."
                )

                st.code(str(e))

                st.stop()


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if st.session_state.retriever is not None:

    st.divider()

    question = st.text_input(
        "Ask a question about the book",
        placeholder="Example: What is the main idea of this book?"
    )

    if st.button("Ask"):

        # -------------------------------
        # Validate question
        # -------------------------------

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()

        # -------------------------------
        # Retrieve Documents
        # -------------------------------

        with st.spinner("🔍 Searching the book..."):

            try:

                retrieved_docs = (
                    st.session_state
                    .retriever
                    .invoke(question)
                )

            except Exception as e:

                st.error(
                    "❌ Error during document retrieval."
                )

                st.code(str(e))

                st.stop()

        st.info(
            f"Retrieved Chunks: {len(retrieved_docs)}"
        )

        # -------------------------------
        # No documents found
        # -------------------------------

        if len(retrieved_docs) == 0:

            st.error(
                "No relevant information found."
            )

            st.stop()

        # -------------------------------
        # Build Context
        # -------------------------------

        context_parts = []

        for i, doc in enumerate(
            retrieved_docs,
            start=1
        ):

            # Limit each chunk
            text = doc.page_content[:4000]

            context_parts.append(
                f"[Source {i}]\n{text}"
            )

        context = "\n\n".join(
            context_parts
        )

        # -------------------------------
        # Prompt
        # -------------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are KnowledgeBot, an AI assistant
that answers questions about an uploaded book.

Your task is to answer ONLY using the
provided context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the context contains the answer,
   provide a clear and concise answer.
4. If the context only partially answers
   the question, explain only what can be
   supported by the context.
5. If the answer is not present in the context,
   reply exactly:

'I could not find the answer in the document.'

6. When possible, mention the source number
   such as [Source 1] or [Source 2].
"""
                ),
                (
                    "human",
                    """
Context:

{context}

Question:

{question}

Answer:
"""
                )
            ]
        )

        # -------------------------------
        # Create Final Prompt
        # -------------------------------

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        # -------------------------------
        # Load LLM
        # -------------------------------

        try:

            llm = load_llm()

        except Exception as e:

            st.error(
                "❌ Could not initialize Mistral LLM."
            )

            st.code(str(e))

            st.stop()

        # -------------------------------
        # Generate Answer
        # -------------------------------

        with st.spinner(
            "🤖 Generating Answer..."
        ):

            try:

                response = llm.invoke(
                    final_prompt
                )

            except Exception as e:

                st.error(
                    "❌ Mistral API request failed."
                )

                st.warning(
                    "Check your MISTRAL_API_KEY, "
                    "Mistral account usage/billing, "
                    "model availability, and Streamlit logs."
                )

                st.code(str(e))

                st.stop()

        # -------------------------------
        # Display Answer
        # -------------------------------

        st.subheader("💡 Answer")

        st.write(
            response.content
        )

        # -------------------------------
        # Retrieved Context
        # -------------------------------

        with st.expander(
            "📖 Retrieved Context"
        ):

            st.write(context)

        # -------------------------------
        # Source Information
        # -------------------------------

        with st.expander(
            "📚 Source Information"
        ):

            for i, doc in enumerate(
                retrieved_docs,
                start=1
            ):

                page_number = (
                    doc.metadata.get(
                        "page",
                        "Unknown"
                    )
                )

                st.write(
                    f"**Source {i} — "
                    f"Page {page_number}**"
                )
