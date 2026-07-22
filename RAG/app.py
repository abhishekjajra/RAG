import streamlit as st
from RagPipeline import RAGPipeline

st.set_page_config(
    page_title="Constitution of India Assistant",
    page_icon="🇮🇳",
    layout="wide"
)

st.title("🇮🇳 Constitution of India RAG Assistant")
st.write("Ask questions about the Constitution of India.")


@st.cache_resource
def load_rag():

    rag = RAGPipeline()
    rag.build_pipeline("ICI.pdf")

    return rag


rag = load_rag()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat
question = st.chat_input("Ask your question...")

if question:

    
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = rag.ask(question)

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
