import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import SentenceTransformerEmbedding
import fitz  # PyMuPDF
import io

# Initialize the SentenceTransformer model (MiniLM-L6-v2)
embedding_path = r"C:\Users\USER\Desktop\ML\LLM\RAG Test\all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_path)

# Create a wrapper class for SentenceTransformer embeddings
class SentenceTransformerEmbeddings(SentenceTransformerEmbedding):
    def __init__(self, model):
        super().__init__(model)
    
    def embed(self, texts):
        return self.model.encode(texts)

# Streamlit UI setup
st.title("Chatbot UI with LLM and PDF Integration")
st.write("This chatbot can either interact with a language model or answer questions based on the content of an uploaded PDF.")

# Select interaction mode
mode = st.selectbox("Select mode", ["Chatbot", "PDF-Based Q&A"])

if mode == "Chatbot":
    # Input field for user query
    user_input = st.text_input("You: ", placeholder="Type your message here...")

    # Function to send the user's query to the LLM server and get a response
    def get_llm_response(query):
        url = "http://localhost:1234/v1/chat/completions"  # Correct endpoint
        headers = {"Content-Type": "application/json"}
        
        # Construct the payload based on the provided cURL example
        payload = {
            "model": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",  # Use the correct model
            "messages": [
                {"role": "system", "content": "Always answer in rhymes."},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": False  # Set to True if you want to handle streaming responses
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                # If streaming is off, process the full response
                return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Request failed: {str(e)}"

    # If the user enters a query, send it to the LLM and display the response
    if user_input:
        with st.spinner("Generating response..."):
            llm_response = get_llm_response(user_input)
            st.text_area("LLM:", llm_response, height=200)

elif mode == "PDF-Based Q&A":
    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Input field for user query
    user_input = st.text_input("You: ", placeholder="Type your message here...")

    if uploaded_file and user_input:
        # Read the file-like object into a byte stream
        file_stream = io.BytesIO(uploaded_file.read())

        # Load and process the PDF document using PyMuPDF
        with st.spinner("Processing PDF..."):
            doc = fitz.open(stream=file_stream, filetype="pdf")
            pdf_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pdf_text += page.get_text()

            # Create the vector store using MiniLM-L6-v2 embeddings
            embedding_instance = SentenceTransformerEmbeddings(embedding_model)
            index = VectorstoreIndexCreator(
                vectorstore_cls=Chroma,
                embedding=embedding_instance
            ).from_documents([{"text": pdf_text}])

            # Function to send the user's query along with PDF content to the LLM server and get a response
            def get_llm_response_with_context(query, context):
                url = "http://localhost:1234/v1/chat/completions"  # Correct endpoint
                headers = {"Content-Type": "application/json"}
                
                # Construct the payload with context from the PDF
                payload = {
                    "model": "TheBloke/MISTRAL-7B-INSTRUCT-GGUF/mistral.gguf",  # Use the correct model
                    "messages": [
                        {"role": "system", "content": "Always answer in rhymes."},
                        {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": -1,
                    "stream": False  # Set to True if you want to handle streaming responses
                }
                
                try:
                    response = requests.post(url, json=payload, headers=headers)
                    if response.status_code == 200:
                        # If streaming is off, process the full response
                        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
                    else:
                        return f"Error: {response.status_code} - {response.text}"
                except requests.exceptions.RequestException as e:
                    return f"Request failed: {str(e)}"

            # Get the response from the local chatbot with PDF content
            if user_input:
                with st.spinner("Generating response..."):
                    llm_response = get_llm_response_with_context(user_input, pdf_text)
                    st.text_area("LLM:", llm_response, height=200)
