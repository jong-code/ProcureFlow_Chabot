import streamlit as st
from difflib import SequenceMatcher
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
import pandas as pd

# --------------------------------------------------------
# SNOWFLAKE CONNECTION (uses Streamlit Secrets)
# --------------------------------------------------------
# Initialize connection
@st.cache_resource
def init_connection():
    try:
        # Try to get active session first (for Snowsight)
        return get_active_session()
    except Exception:
        # Fall back to manual connection using secrets
        connection_parameters = {
            "account": st.secrets["snowflake"]["account"],
            "user": st.secrets["snowflake"]["user"],
            "password": st.secrets["snowflake"]["password"],
            "warehouse": st.secrets["snowflake"]["warehouse"],
            "database": st.secrets["snowflake"]["database"],
            "schema": st.secrets["snowflake"]["schema"],
            "role": st.secrets["snowflake"].get("role", "ACCOUNTADMIN")
        }
        return Session.builder.configs(connection_parameters).create()

# Get Snowflake session
session = init_connection()

st.title("💬 ProcureFlow Procurement Chatbot (AI-Enhanced)")


# --------------------------------------------------------
# LOAD Q&A TABLE
# --------------------------------------------------------
@st.cache_data
def load_qa():
    if session is None:
        st.stop()

    df = session.table("PROCURE_TABLE").select("QUESTION", "ANSWER").to_pandas()
    df["QUESTION"] = df["QUESTION"].astype(str).str.strip()
    df["ANSWER"] = df["ANSWER"].astype(str).str.strip()
    return df


df = load_qa()
st.write(f"📌 Loaded **{len(df)}** Q&A pairs.")


# --------------------------------------------------------
# FUZZY MATCH FUNCTION
# --------------------------------------------------------
def best_match(user_input):
    scores = []
    cleaned = user_input.lower().strip()

    for q in df["QUESTION"].tolist():
        score = SequenceMatcher(None, cleaned, q.lower()).ratio()
        scores.append(score)

    best_index = scores.index(max(scores))
    return df.iloc[best_index]["QUESTION"], df.iloc[best_index]["ANSWER"], max(scores)


# --------------------------------------------------------
# CHECK IF CORTEX IS AVAILABLE
# --------------------------------------------------------
def check_cortex():
    if session is None:
        return False
    try:
        q = "SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2','test') AS r"
        session.sql(q).collect()
        return True
    except:
        return False


cortex_enabled = check_cortex()


# --------------------------------------------------------
# CHAT UI
# --------------------------------------------------------
st.markdown("### 🤖 Ask anything about Procurement")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --------------------------------------------------------
# PROCESS USER INPUT
# --------------------------------------------------------
if prompt := st.chat_input("Type your procurement question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    q, a, score = best_match(prompt)

    qa_context = f"""
    You are an AI assistant for a procurement office.

    The closest matched FAQ entry is:
    Question: {q}
    Answer: {a}
    Match confidence: {score:.3f}

    User Question: {prompt}

    If confidence is high (>0.60), refine & expand the given answer.
    If confidence is low (<=0.60), explain that you're using reasoning instead.
    Provide a clear, helpful procurement answer.
    """

    # --------------------------------------------------------
    # AI MODE (Cortex)
    # --------------------------------------------------------
    if cortex_enabled:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    query = f"""
                        SELECT SNOWFLAKE.CORTEX.COMPLETE(
                            'mistral-large2',
                            '{qa_context.replace("'", "''")}'
                        ) AS RESPONSE
                    """
                    res = session.sql(query).collect()
                    bot_response = res[0]["RESPONSE"]

                    st.markdown(bot_response)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": bot_response}
                    )

                except Exception as e:
                    error_msg = f"❌ Cortex Error: {e}"
                    st.markdown(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # --------------------------------------------------------
    # FALLBACK MODE (No Cortex)
    # --------------------------------------------------------
    else:
        fallback = f"**Closest Match:** {q}\n\n**Answer:** {a}\n\n⚠ Cortex is disabled."
        with st.chat_message("assistant"):
            st.markdown(fallback)

        st.session_state.messages.append(
            {"role": "assistant", "content": fallback}
        )


# --------------------------------------------------------
# RELOAD DATA BUTTON
# --------------------------------------------------------
if st.button("🔄 Reload Data"):
    load_qa.clear()
    st.success("Data reloaded!")

