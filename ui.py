import streamlit as st

# Streamlit UI setup
st.title("Assistant Chatbot with PDF Integration")
st.write("This chatbot can either interact with a language model or answer questions based on the content of an uploaded PDF.")

# Select interaction mode
mode = st.selectbox("Select mode", ["Chatbot", "PDF-Based Q&A"])

if mode == "Chatbot":
    # Input field for user query
    user_input = st.text_input("You: ", placeholder="Type your message here...")

    # Display response area
    st.text_area("LLM:", "", height=200)

elif mode == "PDF-Based Q&A":
    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Input field for user query
    user_input = st.text_input("You: ", placeholder="Type your message here...")

    # Display response area
    st.text_area("LLM:", "", height=200)