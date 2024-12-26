from datetime import datetime

import openai
import streamlit as st

import database
from models.message import save_message
from models.result import save_result
import models

if 'database_initialized' not in st.session_state:
    database.create_database()
    st.session_state.database_initialized = True

if 'chat_initialized' not in st.session_state:
    # connect openai key
    openai.api_key = st.secrets["OPENAI_API_KEY"]

    st.session_state.character_prompt = ("You are a friendly and helpful chatbot who loves to assist people in a cheerful manner."
                        "Your name is Mufo Gapit a big talker! the son of Shosh and Vampir")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.chat_initialized = True

new_user = models.user.User(name="Gigi Valas", email="gigi.vala@example.com")

if 'user_added' not in st.session_state:
    models.user.add_user(new_user)
    st.session_state.user_added = True

if 'page' not in st.session_state:
    st.session_state.page = "Home"  # Default page is Home

def page_chat():
    st.title("Jarvis")
    home_button = st.button("Finish chat...")
    if home_button:
        st.session_state.page = "Result"
        st.rerun()

        # # connect openai key
    # openai.api_key = st.secrets["OPENAI_API_KEY"]
    #
    # character_prompt = ("You are a friendly and helpful chatbot who loves to assist people in a cheerful manner."
    #                     "Your name is Mufo Gapit a big talker! the son of Shosh and Vampir")
    #
    # if "openai_model" not in st.session_state:
    #     st.session_state["openai_model"] = "gpt-3.5-turbo"
    #
    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

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
        save_message("user",prompt,st.session_state.user_name,"assistant",current_time,new_user)
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
                                 {"role": "system", "content": st.session_state.character_prompt}
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
            "from": st.session_state.user_name,
            "timestamp": response_time.isoformat()
        })
        save_message("user", full_response, "assistant", st.session_state.user_name, response_time,new_user)
        # st.write(st.session_state)



def page_home():
    st.title("HOME")
    st.write("Welcome to the Home Page!")
    user_name = st.text_input("Your ID please")
    chat_button = st.button("Go to Chat")
    if chat_button and user_name:
        st.session_state.user_name = user_name
        st.session_state.page = "Chat"
        st.rerun()


def page_result():
    st.title("RESULT")
    st.write("This is the Result page.")
    summarize = summarize_chat()
    st.write(summarize)
    result_time = datetime.now()
    save_result(summarize, result_time, new_user)

def summarize_chat():
    if len(st.session_state.messages) == 0:
        return "No conversation to summarize."

    # Concatenate the chat history
    chat_history = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages
    )

    # Use OpenAI to summarize the chat
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": f"Please summarize the following conversation:\n{chat_history}"}
        ]
    )

    summary = response['choices'][0]['message']['content']
    return summary


# page = st.radio("Choose a page", ("home", "Chat", "Result"))
# Display the corresponding page
if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Chat":
    page_chat()
elif st.session_state.page == "Result":
    page_result()