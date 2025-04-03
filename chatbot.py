import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Initialize the model (keep your existing code)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Set up the chat interface
st.title("My Chatbot")
st.header("My First AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your Prompt"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    result = model.invoke(prompt)
    
    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(result.content)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result.content})