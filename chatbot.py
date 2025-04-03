import streamlit as st
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Ensure the API token is set
if not API_TOKEN:
    st.error("Hugging Face API token is missing. Set it in the environment.")
    st.stop()

# Initialize the model with API token
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="text-generation",
    huggingfacehub_api_token=API_TOKEN  # ✅ Pass API token explicitly
)
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.title("My Chatbot")
st.header("My First AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your Prompt"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        result = model.invoke(prompt)  # Get AI response
        response = result.content if hasattr(result, 'content') else str(result)  # Handle response

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {e}")  # Display error if API call fails
