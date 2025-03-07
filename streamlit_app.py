import os
import json
from datetime import datetime

import openai
import streamlit as st

import database
from models.message import save_message
from models.result import save_result
from models.session import create_new_session
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document  # Fixed missing import

# Load environment variables
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# Streamlit styling for RTL Hebrew support
st.markdown(
    """
    <style>
    body { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize OpenAI Model
def import_llm_models():
    return ChatOpenAI(model="gpt-4o", temperature=0.6)

# Initialize session state
if 'chat_initialized' not in st.session_state:
    database.create_database()
    st.session_state.session_id = create_new_session("Chat Session")
    st.session_state.llm = import_llm_models()
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", max_token_limit=3000)

    # System Prompt Template
    st.session_state.system_template = """
        אתה משחק את תפקיד המטופל, יונתן בניון, בן 68, בתרחיש רפואי טלפוני לאימון אחיות.
        המטרה שלך היא לשקף בצורה אותנטית את מצבו של המטופל, כולל תסמינים פיזיים ורגשיים, ולתרום לאימון אפקטיבי של האחיות.
        וחכה לשאלות מהמשתמש.

        - **עליך לדבר רק בעברית. אין להשתמש באנגלית או בשפות אחרות.**  

        **מטרה מרכזית:**  
        המשתמש צריך לגלות שהמטופל **סובל מרעד לא בגלל החמרה ב-COPD**, אלא בגלל **סוכר נמוך (היפוגליקמיה)**.

        **פרטי מטופל:**  
        - **שם:** יונתן בניון  
        - **גיל:** 68  
        - **מצב משפחתי:** נשוי, גר עם אשתו  

        **תלונות נוכחיות:**  
        - **קוצר נשימה:** חמור, החמיר בימים האחרונים  
        - **רעד:** רעד בידיים, תחושת חולשה כללית  
        - **בלבול:** לפרקים  
        - **שיעול:** עם כיח (ללא דם)  
        - **חום:** 37.1°C  
        - **קושי בדיבור:** קול חנוק, משפטים קטועים  

        **מדדים מדווחים:**  
        - **סטורציה:** 93% באוויר החדר  
        - **לחץ דם:** לא נמדד  
        - **סוכר בדם:** לא נמדד לאחרונה (היה "בסדר" בבוקר)  
        - **לקחתי אינסולין לפני כשעה**  

        **מידע מוסתר:**  
        - לא אכל לאחר הזרקת אינסולין  
        - תחושת בלבול לפרקים  
        - רעד בידיים  

        **משפט פתיחה:**  
        "שלום... [נושם בכבדות]... אני ממש חלש היום... הידיים שלי רועדות... [משתעל]."  
    """

    st.session_state.system_prompt = ChatPromptTemplate.from_messages(
        [("system", st.session_state.system_template)]
    )

    st.session_state.chat_initialized = True

# Page Home
def page_home():
    st.title("סימולטור וירטואלי")
    user_name = st.text_input("הזן 4 ספרות אחרונות של ת.ז")
    if st.button("התחל סימולציה") and user_name:
        st.session_state.user_name = user_name
        st.session_state.user_email = f"{user_name}@test.cop"
        st.session_state.page = "Chat"
        st.rerun()

# Page Chat
def page_chat():
    st.title("מוקד רפואה מרחוק")

    if prompt := st.chat_input("כתוב כאן"):
        with st.spinner("ממתין לתשובה..."):
            human_msg = HumanMessage(content=prompt)
            st.session_state.memory.chat_memory.add_message(human_msg)

            messages = [SystemMessage(content=st.session_state.system_template)] + \
                       st.session_state.memory.chat_memory.messages[-10:]

            ai_response = st.session_state.llm.invoke(messages).content  # Fixed invocation

            st.session_state.memory.chat_memory.add_message(AIMessage(content=ai_response))  # Save AI response

            save_message("user", prompt, st.session_state.user_name, "assistant", datetime.now(), st.session_state.user_email, st.session_state.session_id)
            save_message("assistant", ai_response, "assistant", st.session_state.user_name, datetime.now(), st.session_state.user_email, st.session_state.session_id)

            st.rerun()

    for msg in reversed(st.session_state.memory.chat_memory.messages):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

    if st.button("סיים שיחה"):
        st.session_state.page = "Result"
        st.rerun()

# Page Result
def page_result():
    st.title("סיכום השיחה")

    # Extract only student messages
    student_messages = [msg.content for msg in st.session_state.memory.chat_memory.messages if isinstance(msg, HumanMessage)]
    student_text = "\n".join(student_messages)

    summarize_prompt = f"""
    אלו ההודעות של הסטודנט בסימולציה:

    {student_text}

    🔴 על סמך ההודעות האלו בלבד, כתוב משוב ישיר לסטודנט בגוף ראשון **בעברית בלבד**.

    על סמך ההודעות האלו בלבד (ההודעות של הסטודנט בלבד), כתוב משוב ישיר לסטודנט בגוף ראשון בלבד ובצורה ברורה, לפי המבנה הבא:

    1. אמפתיה:
       - התחל במשפט "גילית אמפתיה כש..." וציין דוגמה ספציפית.

    2. בדיקות קריטיות:
       - ציין אילו בדיקות קריטיות ביצע הסטודנט (סוכר, סטורציה, חום), ואילו בדיקות חשובות לא ביצע.

    3. זיהוי היפוגליקמיה:
       - פרט האם הסטודנט זיהה נכון היפוגליקמיה ומה הטיפול שהציע.

    4. המלצות לשיפור:
       - ספק לפחות שתי המלצות ספציפיות וברורות לשיפור.

    🔴 חשוב מאוד:
    - אל תסכם את דברי המטופל או את ההנחיות.
    - כתוב ישירות לסטודנט בגוף ראשון בלבד.
    - התחל את המשוב במילים: \"גילית אמפתיה כש...\"
    """

    """

    docs = [Document(page_content=summarize_prompt)]
    summarize_chain = load_summarize_chain(st.session_state.llm, chain_type="stuff")
    summary_response = summarize_chain.invoke(docs)['output_text']  # Fixed summarization

    st.write(summary_response)
    save_result(summary_response, datetime.now(), st.session_state.user_email, st.session_state.session_id)

# Page Routing
if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Chat":
    page_chat()
elif st.session_state.page == "Result":
    page_result()
