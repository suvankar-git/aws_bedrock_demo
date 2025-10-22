import boto3
import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferMemory

# Set AWS profile
os.environ["AWS_PROFILE"] = "default"

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# LangChain Chat wrapper for Bedrock
llm = ChatBedrock(
    model_id="amazon.nova-pro-v1:0",
    model_provider="amazon",
    credentials_profile_name="default",
    region_name="us-east-1",
    model_kwargs={
        "maxTokens": 1000,
        "temperature": 0.7
    }
)

# Memory for chat (track only user freeform_text)
memory = ConversationBufferMemory(
    input_key="freeform_text",      # <-- important fix
    memory_key="chat_history",
    return_messages=True
)

# Prompt template with history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful chatbot. You must reply in {language}."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{freeform_text}")
])

# Build chain once
bedrock_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Chat function
def my_chatbot(language, freeform_text):
    # use invoke since we have multiple inputs
    response = bedrock_chain.invoke({"language": language, "freeform_text": freeform_text})
    return response["text"]

# ---- Streamlit UI ----
st.title("Bedrock Chatbot (Amazon Nova Pro)")

# Initialize session_state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

language = st.sidebar.selectbox("Language", ["english", "spanish"])

# Text input instead of text_area (better for chat)
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response from chatbot
    answer = my_chatbot(language, user_input)

    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display the conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
