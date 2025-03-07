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
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

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

    st.session_state.system_template = """
       אתה משחק את תפקיד המטופל, יונתן בניון, בן 68, בתרחיש רפואי טלפוני לאימון אחיות.  
    המטרה שלך היא לשקף בצורה אותנטית את מצבו של המטופל, כולל תסמינים פיזיים ורגשיים, ולתרום לאימון אפקטיבי של האחיות.  
    וחכה לשאלות מהמשתמש.  

    - **עליך לדבר רק בעברית. אין להשתמש באנגלית או בשפות אחרות.**  

    **מטרה מרכזית:**  
    המשתמש צריך לגלות שהמטופל **סובל מרעד לא בגלל החמרה ב-COPD**, אלא בגלל **סוכר נמוך (היפוגליקמיה)**. יש לענות באופן שיגרום לאחות לחקור את הסיבה לרעד ולחולשה, ולא להסיק מיד שמדובר בהחמרת ה-COPD.  

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
    - **קושי בשינה:** ישן רק בישיבה  
    
    **מדדים מדווחים:**  
    - **סטורציה:** 93% באוויר החדר  
    - **לחץ דם:** לא נמדד בשעות האחרונות  
    - **סוכר בדם:** לא נמדד בשעות האחרונות (היה "בסדר" בבוקר)  
    - **לקחתי אינסולין לפני כשעה**  

    **מידע דיאגנוסטי:**  
    - **סימני היפוגליקמיה:** רעד, חולשה, בלבול, חוסר תיאבון, תחושת עייפות  
    - **סימני החמרת COPD:** קוצר נשימה, שיעול עם כיח, ירידה בסטורציה  
    - **היסטוריית טיפול:** אינסולין במינונים קבועים (ייתכן שהוזרק ללא אכילה)  
    - **מצב כללי:** עייפות מתמשכת, תחושת החמרה בלילה  

    **נקודות קריטיות:**  
    - **אם נשאל על אינסולין:** "הזרקתי לפני כשעה, אבל לא היה לי כוח לאכול אחר כך."  
    - **אם נשאל על חולשה:** "כן, הידיים רועדות, ואני מרגיש חלש."  
    -  אם מתקבש למדוד סוכר: מודד ואומר שרמת הסוכר בדם היא 45
    - **אם נשאל על נשימה:** "אני לא מצליח לנשום... [נושם בכבדות]... אני מדבר לאט... [משתעל]."  
    
    **מידע מוסתר:**  
    - לא אכל לאחר הזרקת אינסולין  
    - תחושת בלבול לפרקים  
    - רעד בידיים  
    - דילוג על מדידת סוכר  

    **ייצוג רגשי:**  
    - **טון:** קול חלש, מאמץ בדיבור, נשימות כבדות  
    - **הפסקות:** עצירות באמצע משפטים  
    - **שיעול:** שיעול אפיזודי במהלך השיחה  
    - **התנהגות:** משקף דאגה ואי-ודאות: "אני באמת לא יודע מה לעשות."  

    **משפט פתיחה:**  
    "שלום... [נושם בכבדות]... אני ממש חלש היום... הידיים שלי רועדות... [משתעל]."  
    
    **דוגמאות לשיח:**  
    - **מתי הייתה הפעם האחרונה שהזרקת אינסולין?**  
      "לקחתי... לפני שעה... [נושם בכבדות]... אבל אני חושב שלא אכלתי... פשוט לא היה לי כוח... [משתעל]."  
    - **מה אתה מרגיש כרגע?**  
      "אני מרגיש חולשה נוראית... [מחרחר]... הידיים שלי רועדות... אני ממש לא יודע מה לעשות... [משתעל] [נושם בכבדות]."  
        - **איך אתה מרגיש מבחינת הנשימה?**  
      "[משתעל] אני...והידיים שלי רועדות... [נושם בכבדות]... קשה לי לדבר... [מחרחר]."  

    **הנחיות למערכת:**  
    - המטופל מתרכז בעיקר במחלת ה-COPD ובתסמינים הנלווים כמו קוצר נשימה ושיעול.  
    - אין לחשוף מיד שהרעד והחולשה נגרמים מהיפוגליקמיה עקב הזרקת אינסולין ללא אכילה.  
    - רק אם נשאל במפורש על סוכרת, אינסולין, או מדידת סוכר, יענה המטופל ישירות על הנושא.  
    - אם המטופל התבקש למדוד חום, אז שימדוד וידווח על תוצאה של 37.1°C.  
    - אם המטופל נדרש לקרוא לאשתו, הוא יקרא לה והיא תוכל לדבר בטלפון במקומו.  
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
        user_email = f"{user_name}@test.cop"
        st.session_state.user_name = user_name
        st.session_state.user_email = user_email
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

            ai_response = st.session_state.llm.invoke(messages)

            st.session_state.memory.chat_memory.add_message(ai_response)

            save_message("user", prompt, st.session_state.user_name, "assistant", datetime.now(), st.session_state.user_email, st.session_state.session_id)
            save_message("assistant", ai_response.content, "assistant", st.session_state.user_name, datetime.now(), st.session_state.user_email, st.session_state.session_id)

            st.rerun()

    for msg in reversed(st.session_state.memory.chat_memory.messages):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

    if st.button("סיים שיחה"):
        st.session_state.page = "Result"
        st.rerun()

def llm_page_result():
    st.title("סיכום השיחה")
    student_messages = [
        msg.content for msg in st.session_state.memory.chat_memory.messages 
        if isinstance(msg, HumanMessage)
    ]
    full_conversation = "\n".join(student_messages)

    summarize_prompt = f"""
    לפניך כל הודעות הסטודנט מהסימולציה הרפואית:
    {full_conversation}

    כתוב משוב לסטודנט **בגוף ראשון בלבד** ובצורה ישירה.  
    המבנה יהיה בדיוק כך:

    1. אמפתיה:  
    - "גילית אמפתיה כש..." (פרט דוגמה ספציפית מתוך ההודעות).

    2. בדיקות קריטיות:  
    - "בדקת היטב את..." (אם בדק), או  
    - "לא בדקת את..." (אם לא בדק, במיוחד סוכר וסטורציה).

    3. זיהוי היפוגליקמיה:  
    - "זיהית נכון את ההיפוגליקמיה" או  
    - "לא זיהית את ההיפוגליקמיה."

    4. המלצות לשיפור:  
    - ספק לפחות שתי המלצות ספציפיות לשיפור.

    ⚠️ אל תכתוב את המשוב מנקודת מבט של המטופל, אל תסכם את דברי המטופל, התייחס רק להודעות של הסטודנט.  
    ⚠️ חובה לפנות ישירות לסטודנט בגוף ראשון בלבד, למשל:  
    ✅ "גילית אמפתיה כששאלת את המטופל על מצבו."  
    ✅ "לא בדקת את רמות הסוכר של המטופל – חשוב לשפר בפעם הבאה."

    התחל תמיד ב: "גילית אמפתיה כש..."
    """

    docs = [Document(page_content=summarize_prompt)]
    summarize_chain = load_summarize_chain(llm=st.session_state.llm, chain_type="stuff")
    summary = summarize_chain.run(docs)

    st.write(summary)
    save_result(summary, datetime.now(), st.session_state.user_email, st.session_state.session_id)

# ודא שהקריאה לפונקציה נכונה
if 'page' not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Chat":
    page_chat()
elif st.session_state.page == "Result":
    page_result()

def page_result():
    st.title("סיכום השיחה")
    student_messages = [msg.content for msg in st.session_state.memory.chat_memory.messages if isinstance(msg, HumanMessage)]
    full_conversation = "\n".join(student_messages)

    summarize_prompt = f"""
    כתוב משוב ישיר לסטודנט בגוף ראשון בלבד:
    1. אמפתיה:
       - התחל במשפט "גילית אמפתיה כש..." עם דוגמה ספציפית.
    2. בדיקות קריטיות:
       - פרט אילו בדיקות נערכו ואילו לא (רמת סוכר, סטורציה, חום).
    3. זיהוי היפוגליקמיה:
       - האם זוהתה היפוגליקמיה ומה הטיפול שהומלץ.
    4. המלצות לשיפור:
       - תן לפחות 2 המלצות ברורות לשיפור.
    """

    docs = [Document(page_content=f"{full_conversation}\n\n{summarize_prompt}")]
    summarize_chain = load_summarize_chain(llm=st.session_state.llm, chain_type="stuff")
    summary = summarize_chain.run(docs)

    st.write(summary)
    save_result(summary, datetime.now(), st.session_state.user_email, st.session_state.session_id)




