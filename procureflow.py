import streamlit as st
from difflib import SequenceMatcher
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
import pandas as pd
import os

# --------------------------------------------------------
# SNOWFLAKE CONNECTION (external connection)
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

st.title("💬 ProcureFlow Procurement Chatbot (External App)")

# --------------------------------------------------------
# LOAD Q&A DATA FROM SNOWFLAKE
# --------------------------------------------------------
@st.cache_data
def load_qa():
    df = session.table("PROCURE_TABLE").select("QUESTION", "ANSWER").to_pandas()
    df["QUESTION"] = df["QUESTION"].astype(str).str.strip()
    df["ANSWER"]  = df["ANSWER"].astype(str).str.strip()
    return df

df = load_qa()
st.write(f"📌 Loaded **{len(df)}** Q&A pairs from Snowflake.")

# --------------------------------------------------------
# FUZZY MATCHING
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
# CHECK CORTEX AVAILABILITY
# --------------------------------------------------------
def cortex_available():
    try:
        test = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2','hello') as r"
        ).collect()
        return True
    except:
        return False

cortex_enabled = cortex_available()

# --------------------------------------------------------
# CHAT UI
# --------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("### 🤖 Ask anything about procurement")

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------------
# CHAT INPUT HANDLING
# --------------------------------------------------------
if prompt := st.chat_input("Your procurement question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get best matched FAQ
    q, a, score = best_match(prompt)

    # Build context for AI
    cortex_prompt = f"""
    You are an AI procurement assistant.

    Closest matched FAQ:
    Question: {q}
    Answer: {a}
    Confidence Score: {score:.3f}

    User Question: {prompt}

    If confidence > 0.60 → refine and improve the FAQ answer.
    If confidence ≤ 0.60 → produce a knowledgeable procurement explanation.
    Always answer clearly and professionally.
    """

    # ----------------------------------------------------
    # THINKING MODE (CORTEX) IF AVAILABLE
    # ----------------------------------------------------
    if cortex_enabled:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    cortex_query = f"""
                        SELECT SNOWFLAKE.CORTEX.COMPLETE(
                            'mistral-large2',
                            '{cortex_prompt.replace("'", "''")}'
                        ) AS RESPONSE
                    """
                    result = session.sql(cortex_query).collect()
                    response = result[0]["RESPONSE"]

                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    err = f"❌ Cortex Error: {str(e)}"
                    st.markdown(err)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err}
                    )
    else:
        # Fallback to regular match only
        fallback = f"""
        **Closest Match:** {q}

        **Answer:** {a}

        ⚠ Cortex AI is not enabled in your Snowflake account.
        """
        with st.chat_message("assistant"):
            st.markdown(fallback)

        st.session_state.messages.append(
            {"role": "assistant", "content": fallback}
        )

# Reload Q&A
if st.button("🔄 Reload Data"):
    load_qa.clear()
    df = load_qa()
    st.success("Data reloaded from Snowflake!")
