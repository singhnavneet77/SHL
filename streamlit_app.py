
import os
import streamlit as st
import requests

API_URL = os.environ.get(
    "RENDER_API_URL",
    "http://localhost:8000"
).rstrip("/") + "/chat"


st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 SHL Assessment Recommender")
st.markdown(
    "Find the right SHL assessments using conversational AI"
)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("recommendations"):
            st.markdown("### Recommendations")

            for rec in msg["recommendations"]:
                st.markdown(
                    f"- **{rec['name']}** "
                    f"({rec['test_type']})  \n"
                    f"{rec['url']}"
                )

# User input
prompt = st.chat_input(
    "Example: Hiring Java developer with stakeholder communication"
)

if prompt:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare payload for backend
    payload = {
        "messages": [
            {
                "role": m["role"],
                "content": m["content"]
            }
            for m in st.session_state.messages
        ]
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=60
        )

        data = response.json()

        assistant_reply = data.get("reply", "No response")
        recommendations = data.get("recommendations", [])

        # Add assistant response to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_reply,
                "recommendations": recommendations
            }
        )

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

            if recommendations:
                st.markdown("### Recommendations")

                for rec in recommendations:
                    st.markdown(
                        f"- **{rec['name']}** "
                        f"({rec['test_type']})  \n"
                        f"{rec['url']}"
                    )

    except Exception as e:

        st.error(f"Error connecting to backend: {e}")
