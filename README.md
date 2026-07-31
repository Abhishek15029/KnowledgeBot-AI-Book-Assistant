# 📚 KnowledgeBot-AI-Book-Assistant

An AI-powered Book Question Answering Assistant built using **LangChain**, **Mistral AI**, **ChromaDB**, and **Streamlit**. Upload a PDF book, ask questions in natural language, and receive context-aware answers using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

* 📄 Upload PDF books
* 🤖 Ask questions in natural language
* 🔍 Retrieves only the most relevant content using RAG
* 🧠 Generates accurate answers with Mistral AI
* 📚 Uses ChromaDB as a vector database
* ⚡ Interactive Streamlit web interface
* 📖 Displays retrieved context for better transparency

---

## 🛠️ Tech Stack

* **Python**
* **LangChain**
* **Mistral AI**
* **Mistral Embeddings**
* **ChromaDB**
* **PyPDFLoader**
* **RecursiveCharacterTextSplitter**
* **Streamlit**
* **python-dotenv**

---

## 📂 Project Structure

```text
KnowledgeBot-AI-Book-Assistant/
│
├── app.py                  # Streamlit application
├── main.py                 # Main RAG pipeline
├── create_database.py      # Creates Chroma vector database
├── requirements.txt
├── .gitignore
│
├── document_loaders/
│   ├── notes.txt
│   ├── page.py
│   ├── pdf.py
│   └── test.py
│
├── retrievers/
│   ├── mmr.py
│   ├── multiquery.py
│   └── arixv.py
│
└── vector_store/
    └── DB.py
```

---

## ⚙️ How It Works

1. Upload a PDF document.
2. Extract text using **PyPDFLoader**.
3. Split the document into smaller chunks.
4. Generate embeddings using **Mistral Embeddings**.
5. Store embeddings in **ChromaDB**.
6. Retrieve the most relevant chunks for a user query.
7. Send the retrieved context to **Mistral AI**.
8. Display the generated answer in the Streamlit interface.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Abhishek15029/KnowledgeBot-AI-Book-Assistant.git
```

Move into the project directory:

```bash
cd KnowledgeBot-AI-Book-Assistant
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

**Windows**

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 💡 Example Questions

* Summarize Chapter 1.
* What is Gradient Descent?
* Explain Neural Networks.
* What are the key concepts discussed in the book?
* List the important points from this chapter.

---

## 📸 Demo

You can add screenshots here after deployment.

Example:

```
assets/
├── home.png
├── upload.png
└── answer.png
```

---

## 🔮 Future Improvements

* Support multiple PDFs
* Chat history
* Conversation memory
* Source citations with page numbers
* Hybrid Search (Keyword + Vector Search)
* Multiple LLM support (Gemini, OpenAI, Mistral)
* Voice input and text-to-speech
* Deployment on Streamlit Cloud

---

## 👨‍💻 Author

**Abhishek Verma**

* M.Tech CSE (AI), IIIT Pune
* Passionate about AI, Machine Learning, Deep Learning, and Generative AI

---

## ⭐ If you like this project

If you found this project useful, consider giving it a **⭐ Star** on GitHub.
