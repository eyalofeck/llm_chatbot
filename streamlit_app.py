from datetime import datetime

import openai
import streamlit as st

import database
from models.message import save_message
import models

if 'database_initialized' not in st.session_state:
    database.create_database()
    st.session_state.database_initialized = True

new_user = models.user.User(name="Gigi Valas", email="gigi.vala@example.com")

if 'user_added' not in st.session_state:
    models.user.add_user(new_user)
    st.session_state.user_added = True

st.title("Jarvis")
# connect openai key
openai.api_key = st.secrets["OPENAI_API_KEY"]

character_prompt = ("You are a friendly and helpful chatbot who loves to assist people in a cheerful manner."
                    "Your name is Mufo Gapit a big talker! the son of Shosh and Vampir")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    current_time = datetime.now()
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "from": "assistant",
        "timestamp": current_time.isoformat()
    })
    # st.write(st.session_state)
    save_message("user",prompt,"user","assistant",current_time,new_user)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Simulate stream of response with milliseconds delay
        for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                             {"role": "system", "content": character_prompt}
                         ] + st.session_state.messages,
                # will provide lively writing
                stream=True,
        ):
            # get content in response
            full_response += response.choices[0].delta.get("content", "")
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    response_time = datetime.now()

    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "from": "user",
        "timestamp": response_time.isoformat()
    })
    save_message("user", full_response, "assistant", "user", response_time,new_user)
    # st.write(st.session_state)
