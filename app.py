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


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="KnowledgeBot - AI Book Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# Application Title
# ============================================================

st.title("📚 KnowledgeBot - AI Book Assistant")

st.write(
    "Upload a PDF and ask questions about the content."
)


# ============================================================
# Check Mistral API Key
# ============================================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.error(
        "MISTRAL_API_KEY is not configured."
    )

    st.info(
        "Please add MISTRAL_API_KEY to your "
        "Streamlit Secrets."
    )

    st.stop()


# ============================================================
# Load Embedding Model
# ============================================================

@st.cache_resource
def load_embedding_model():

    return MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY
    )


# ============================================================
# Load Mistral LLM
# ============================================================

@st.cache_resource
def load_llm():

    return ChatMistralAI(
        model="mistral-small-2603",
        api_key=MISTRAL_API_KEY,
        temperature=0.2,
        max_retries=0
    )


# ============================================================
# Session State
# ============================================================

if "retriever" not in st.session_state:

    st.session_state.retriever = None


# ============================================================
# PDF Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# ============================================================
# Create Knowledge Base
# ============================================================

if uploaded_file is not None:

    if st.button("Create Knowledge Base"):

        with st.spinner(
            "📚 Processing PDF..."
        ):

            try:

                # ------------------------------------------------
                # Create temporary directory
                # ------------------------------------------------

                temp_dir = tempfile.mkdtemp()

                pdf_path = (
                    Path(temp_dir)
                    / uploaded_file.name
                )

                # ------------------------------------------------
                # Save PDF
                # ------------------------------------------------

                with open(
                    pdf_path,
                    "wb"
                ) as f:

                    f.write(
                        uploaded_file.getbuffer()
                    )

                # ------------------------------------------------
                # Load PDF
                # ------------------------------------------------

                loader = PyPDFLoader(
                    str(pdf_path)
                )

                documents = loader.load()

                st.info(
                    f"📄 Pages Loaded: {len(documents)}"
                )

                # ------------------------------------------------
                # Split PDF into chunks
                # ------------------------------------------------

                splitter = (
                    RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                )

                chunks = splitter.split_documents(
                    documents
                )

                # ------------------------------------------------
                # Limit number of chunks
                # ------------------------------------------------

                MAX_CHUNKS = 1000

                if len(chunks) > MAX_CHUNKS:

                    st.warning(
                        f"Large PDF detected "
                        f"({len(chunks)} chunks). "
                        f"Only the first "
                        f"{MAX_CHUNKS} chunks "
                        f"will be indexed."
                    )

                    chunks = chunks[:MAX_CHUNKS]

                st.info(
                    f"🔹 Chunks Created: {len(chunks)}"
                )

                # ------------------------------------------------
                # Create embeddings
                # ------------------------------------------------

                with st.spinner(
                    "🔢 Creating embeddings..."
                ):

                    embeddings = (
                        load_embedding_model()
                    )

                # ------------------------------------------------
                # Create Chroma vector database
                # ------------------------------------------------

                with st.spinner(
                    "🗄️ Creating vector database..."
                ):

                    vectorstore = (
                        Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings
                        )
                    )

                # ------------------------------------------------
                # Create retriever
                # ------------------------------------------------

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
                    "❌ Error while creating "
                    "the Knowledge Base."
                )

                st.code(str(e))

                st.stop()


# ============================================================
# Question Answering
# ============================================================

if st.session_state.retriever is not None:

    st.divider()

    st.subheader(
        "💬 Ask the Book"
    )

    question = st.text_input(
        "Ask a question about the book",
        placeholder=(
            "Example: What is the main idea "
            "of the book?"
        )
    )

    if st.button("Ask"):

        # ------------------------------------------------
        # Validate question
        # ------------------------------------------------

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()

        # ------------------------------------------------
        # Retrieve relevant documents
        # ------------------------------------------------

        with st.spinner(
            "🔍 Searching the book..."
        ):

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
            f"📖 Retrieved Chunks: "
            f"{len(retrieved_docs)}"
        )

        # ------------------------------------------------
        # Check retrieval result
        # ------------------------------------------------

        if not retrieved_docs:

            st.error(
                "No relevant information "
                "was found in the document."
            )

            st.stop()

        # ------------------------------------------------
        # Build context
        # ------------------------------------------------

        context_parts = []

        for i, doc in enumerate(
            retrieved_docs,
            start=1
        ):

            text = doc.page_content[:4000]

            context_parts.append(
                f"[Source {i}]\n{text}"
            )

        context = "\n\n".join(
            context_parts
        )

        # ------------------------------------------------
        # Prompt
        # ------------------------------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are KnowledgeBot, an AI assistant
that answers questions about an uploaded book.

Answer ONLY using the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is available in the context,
   provide a clear and concise answer.
4. If the context partially answers the question,
   provide only the supported information.
5. If the answer is not present in the context,
   reply exactly:

'I could not find the answer in the document.'

6. When possible, mention the relevant source
   number such as [Source 1] or [Source 2].
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

        # ------------------------------------------------
        # Create final prompt
        # ------------------------------------------------

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        # ------------------------------------------------
        # Load LLM
        # ------------------------------------------------

        try:

            llm = load_llm()

        except Exception as e:

            st.error(
                "❌ Could not initialize Mistral."
            )

            st.code(str(e))

            st.stop()

        # ------------------------------------------------
        # Generate answer
        # ------------------------------------------------

        with st.spinner(
            "🤖 Generating Answer..."
        ):

            try:

                response = llm.invoke(
                    final_prompt
                )

            except Exception as e:

                error_message = str(e)

                # ----------------------------------------
                # Handle Rate Limit
                # ----------------------------------------

                if (
                    "429" in error_message
                    or "rate limit" in error_message.lower()
                    or "rate_limited" in error_message.lower()
                ):

                    st.error(
                        "⚠️ Mistral API rate limit exceeded."
                    )

                    st.warning(
                        "Your RAG pipeline is working, "
                        "but the Mistral API is currently "
                        "rejecting the generation request."
                    )

                    st.info(
                        "Please wait for the API limit "
                        "to reset or check your Mistral "
                        "account usage/billing."
                    )

                else:

                    st.error(
                        "❌ Mistral API request failed."
                    )

                    st.code(
                        error_message
                    )

                st.stop()

        # ------------------------------------------------
        # Display Answer
        # ------------------------------------------------

        st.subheader(
            "💡 Answer"
        )

        st.write(
            response.content
        )

        # ------------------------------------------------
        # Display Retrieved Context
        # ------------------------------------------------

        with st.expander(
            "📖 Retrieved Context"
        ):

            st.write(
                context
            )

        # ------------------------------------------------
        # Display Sources
        # ------------------------------------------------

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
