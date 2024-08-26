import streamlit as st
import requests

# Streamlit UI setup
st.title("Chatbot UI with LLM")
st.write("This chatbot is powered by a language model running on an LM Studio server.")

# Input field for user query
user_input = st.text_input("You: ", placeholder="Type your message here...")

# Function to send the user's query to the LLM server and get a response
def get_llm_response(query):
    url = "http://localhost:1234/v1/chat/completions"  # Correct endpoint
    headers = {"Content-Type": "application/json"}
    
    # Construct the payload based on the provided cURL example
    payload = {
        "model": "TheBloke/MISTRAL-7B-INSTRUCT-GGUF/mistral.gguf",  # Use the correct model
        "messages": [
            {"role": "system", "content": "Your are and research assistant chatbot."},
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
