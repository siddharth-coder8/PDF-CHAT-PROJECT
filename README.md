# 📄 PDF Chatbot – Ask Questions About Any PDF

An interactive Streamlit application that lets you upload a PDF, embed its contents using `OllamaEmbeddings`, and ask context-aware questions using a conversational AI powered by `phi`.

---

## 🚀 Features

* 📁 Upload and parse PDFs
* 🔎 Intelligent chunking of PDF content
* 🧠 Embedding with `OllamaEmbeddings` (LLM: `phi`)
* 💾 Persistent `Chroma` vector store for fast retrieval
* 🤖 LLM-powered question answering using `RetrievalQA`
* 💬 Stateful conversation with memory
* 🖼️ Chat UI with streaming responses and typing animation
* 🔧 Robust error handling and logging

---

## 🧱 Tech Stack

* Python
* Streamlit
* LangChain
* Ollama (local LLM server)
* ChromaDB
* PyPDFLoader

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/siddharth-coder8/PDF-CHAT-PROJECT
cd PDF-CHAT-PROJECT

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Requirements

* Python 3.8+
* [Ollama](https://ollama.com) running locally (tested with model `phi`)
* Required ports open (`http://localhost:11434` for Ollama)
* PDFs to test with

---

## 📂 Folder Structure

```bash
.
├── files/           # Uploaded PDFs
├── jj/              # Chroma vector store
├── main.py          # Main Streamlit app
├── requirements.txt
└── README.md
```

---

## ▶️ Usage

1. Start the Ollama server:

   ```bash
   ollama run phi
   ```

2. Run the app:

   ```bash
   streamlit run main.py
   ```

3. Upload a PDF and start chatting!

---

## 🛠️ Troubleshooting

* If you see connection errors:

  * Make sure Ollama is running at `http://localhost:11434`
  * Ensure the `phi` model is downloaded
* Logs are printed to the terminal via `logging`
