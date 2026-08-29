"""AI Analytics Copilot -- natural-language interface grounded in real platform data."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import settings

st.set_page_config(page_title="AI Copilot | SupplySense", page_icon="🤖", layout="wide")
st.title("🤖 AI Analytics Copilot")
st.caption(
    "Ask questions in plain language. The copilot is grounded in this platform's real data and "
    "engines -- it queries the database and runs the live forecasting/optimization/scenario "
    "engines rather than answering from general knowledge."
)

if not settings.has_llm_key:
    st.warning(
        "**No LLM API key is configured.** Add `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` to your "
        "`.env` file to enable the AI Copilot. Everything else in SupplySense works without it -- "
        "this page alone requires an API key."
    )
    st.stop()

st.caption(f"Currently using: **{settings.active_llm_provider}**")

from ai.copilot import ask_copilot

EXAMPLE_QUESTIONS = [
    "Which stores have the highest stockout risk right now?",
    "Why was XGBoost selected as the forecasting model?",
    "What happens to store 1's order if demand increases by 20%?",
    "What's the recommended order quantity for store 710 and why?",
    "How many data quality checks failed in the latest pipeline run?",
]

if "copilot_history" not in st.session_state:
    st.session_state.copilot_history = []

with st.sidebar:
    st.subheader("Example Questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"example_{hash(q)}", width="stretch"):
            st.session_state.pending_question = q
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.copilot_history = []
        st.rerun()

for turn in st.session_state.copilot_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("tool_calls"):
            with st.expander(f"🔧 {len(turn['tool_calls'])} tool call(s) used"):
                for call in turn["tool_calls"]:
                    st.markdown(f"**{call.tool_name}**({call.tool_input})")
                    st.json(call.tool_result, expanded=False)

pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Ask about forecasts, recommendations, risk, or model performance...")
question = pending or user_input

if question:
    st.session_state.copilot_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking (querying live data and engines)..."):
            try:
                # Build conversation history in Anthropic message format from prior turns
                history = []
                for turn in st.session_state.copilot_history[:-1]:
                    if turn["role"] in ("user", "assistant"):
                        history.append({"role": turn["role"], "content": turn["content"]})
                response = ask_copilot(question, conversation_history=history)
                st.markdown(response.answer)
                if response.tool_calls:
                    with st.expander(f"🔧 {len(response.tool_calls)} tool call(s) used"):
                        for call in response.tool_calls:
                            st.markdown(f"**{call.tool_name}**({call.tool_input})")
                            st.json(call.tool_result, expanded=False)
                st.session_state.copilot_history.append({
                    "role": "assistant", "content": response.answer, "tool_calls": response.tool_calls,
                })
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"The copilot ran into an error: {exc}")
