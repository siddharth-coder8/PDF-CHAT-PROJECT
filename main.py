import logging
import os
import time

import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create directories if they don't exist
os.makedirs('files', exist_ok=True)
os.makedirs('jj', exist_ok=True)

# Initialize session state
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions about uploaded PDFs. Provide accurate and concise answers based on the PDF content. For broad questions like 'what is this PDF about,' summarize the main topics or purpose of the document.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")
if 'vectorstore' not in st.session_state:
    try:
        st.session_state.vectorstore = Chroma(
            persist_directory='jj',
            embedding_function=OllamaEmbeddings(
                base_url='http://localhost:11434',
                model="phi"
            )
        )
    except Exception as e:
        logger.error(f"Failed to initialize Chroma vector store: {str(e)}")
        st.error("Failed to connect to Ollama server. Ensure it's running at http://localhost:11434.")
if 'llm' not in st.session_state:
    try:
        st.session_state.llm = Ollama(
            base_url="http://localhost:11434",
            model="phi",
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
        st.error("Failed to initialize LLM. Ensure Ollama server is running.")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("PDF Chatbot")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF", type='pdf', key="pdf_uploader")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    # Sanitize file name to avoid invalid characters
    safe_file_name = "".join(c for c in uploaded_file.name if c.isalnum() or c in ('.', '_')).rstrip()
    file_path = os.path.join("files", safe_file_name)
    
    try:
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved PDF to {file_path}")
        
        with st.status("Analyzing your document..."):
            # Load and process the PDF
            try:
                loader = PyPDFLoader(file_path)
                data = loader.load()
                if not data:
                    st.error("No content could be extracted from the PDF.")
                    logger.error("PDF content extraction failed.")
                    st.stop()
            except Exception as e:
                st.error(f"Failed to load PDF: {str(e)}")
                logger.error(f"PDF loading error: {str(e)}")
                st.stop()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)
            
            # Update vector store
            try:
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=OllamaEmbeddings(
                        base_url='http://localhost:11434',
                        model="phi"
                    ),
                    persist_directory='jj'
                )
                st.session_state.vectorstore.persist()
                logger.info("Vector store updated successfully.")
            except Exception as e:
                st.error(f"Failed to update vector store: {str(e)}")
                logger.error(f"Vector store update error: {str(e)}")

    except Exception as e:
        st.error(f"Failed to save or process PDF: {str(e)}")
        logger.error(f"File saving error: {str(e)}")

    # Initialize retriever and QA chain
    try:
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )
        st.success("PDF processed successfully!")
    except Exception as e:
        st.error(f"Failed to initialize QA chain: {str(e)}")
        logger.error(f"QA chain initialization error: {str(e)}")
        st.error(f"Failed to initialize QA chain: {str(e)}")
        logger.error(f"QA chain initialization error: {str(e)}")
    if user_input := st.chat_input("Ask about the PDF:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                try:
                    response = st.session_state.qa_chain(user_input)
                    full_response = response['result']
                except Exception as e:
                    full_response = f"Error processing query: {str(e)}"
                    logger.error(f"Query error: {str(e)}")
            message_placeholder = st.empty()
            displayed_response = ""
            for chunk in full_response.split():
                displayed_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(displayed_response + "▌")
            message_placeholder.markdown(displayed_response)
        chatbot_message = {"role": "assistant", "message": full_response}
        st.session_state.chat_history.append(chatbot_message)

else:
    st.info("Please upload a PDF file to start.")
