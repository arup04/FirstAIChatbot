import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load API token from Streamlit Secrets
API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    task="text-generation",
    huggingfacehub_api_token=API_TOKEN
)
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.title("AI Chatbot")

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
        result = model.invoke(prompt)
        response = result.content if hasattr(result, 'content') else str(result)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {e}")
