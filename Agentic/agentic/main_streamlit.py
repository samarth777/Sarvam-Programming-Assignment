import streamlit as st
import os
from utils.schema import NCERTTextbookAssistant

def main():
    st.title("Agentic Chatbot")

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
        st.session_state.messages = []

    if st.session_state.pipeline is None:
        data_path = "./data"
        index_path = "./index"
        
        if os.path.exists(index_path):
            st.session_state.pipeline = NCERTTextbookAssistant(
                data_path=data_path,
                index_path=index_path
            )
            st.success("Loaded existing index.")
        else:
            st.session_state.pipeline = NCERTTextbookAssistant(
                data_path=data_path,
                index_path=index_path
            )
            st.success("Created and saved new index.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about your vehicle?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = st.session_state.pipeline.query(prompt)
            st.markdown(result.answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": result.answer}
        )

if __name__ == "__main__":
    main()