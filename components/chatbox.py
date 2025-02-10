import streamlit as st

from utils.ollama import chat, context_chat


def chatbox():
    if prompt := st.chat_input("How can I help?"):
        # Prevent submission if Ollama endpoint is not set
        if not st.session_state["query_engine"]:
            st.warning("Please confirm settings and upload files before proceeding.")
            st.stop()

        # Add the user input to messages state
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate llama-index stream with user input
        # prompt = prompt + "Please add $$ around all mathematical equations so that they can be shown in newline. Also do mention page number of the reference from the PDF."
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                prompt = prompt + "While answering these questions do mention the page number from the PDF."
                first_pass = []
                for chunk in context_chat(prompt=prompt, query_engine=st.session_state["query_engine"]):
                    first_pass.append(chunk)
                first_pass = "".join(first_pass)

                first_pass = "Please add $$ around all mathematical equations so that they can be shown in newline. To the following text and only return the text : " + first_pass

                response = st.write_stream(
                    chat(prompt=first_pass)
                )

        # Add the final response to messages state
        st.session_state["messages"].append({"role": "assistant", "content": response})
