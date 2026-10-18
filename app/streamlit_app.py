"""Render the Streamlit chatbot UI and forward indexing and chat requests to FastAPI."""

from __future__ import annotations

import httpx
import streamlit as st

from app.config import API_BASE_URL, APP_TITLE


st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")


# Preserve the current chat and selected repository across Streamlit reruns.
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "repo_info" not in st.session_state:
    st.session_state.repo_info = None


def call_api(method: str, path: str, payload: dict | None = None) -> dict:
    """Call the backend and turn non-success HTTP responses into visible UI errors."""
    url = f"{API_BASE_URL}{path}"
    if method == "GET":
        response = httpx.get(url, timeout=180)
    else:
        if path == "/clear":
            response = httpx.post(url, params=payload, timeout=180)
        else:
            response = httpx.post(url, json=payload, timeout=180)
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(detail)
    return response.json()


st.title(APP_TITLE)

# Collect a repository URL and expose the primary index and reset controls.
with st.container():
    repo_url = st.text_input(
        "GitHub repository URL",
        value="",
        placeholder="https://github.com/microsoft/vscode",
    )
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("Index Repository", type="primary"):
            if not repo_url.strip():
                st.error("Please enter a GitHub repository URL.")
            else:
                try:
                    with st.spinner("Validating and indexing repository..."):
                        result = call_api("POST", "/index", {"url": repo_url})
                    st.session_state.repo_info = result
                    st.session_state.conversation = []
                    st.success(f"Repository indexed: {result['repo_name']} ({result['repo_id']})")
                except Exception as exc:
                    st.error(str(exc))
                    st.session_state.repo_info = None
    with col2:
        if st.button("Clear Conversation"):
            st.session_state.conversation = []
            st.toast("Conversation cleared.")
    with col3:
        if st.button("Reset Repository"):
            if st.session_state.repo_info:
                try:
                    call_api("POST", "/clear", {"repo_id": st.session_state.repo_info["repo_id"]})
                except Exception:
                    pass
            st.session_state.repo_info = None
            st.session_state.conversation = []
            st.toast("Repository reset.")

if st.session_state.repo_info:
    # Show repository statistics before enabling the repository chat.
    repo = st.session_state.repo_info
    st.subheader("Repository status")
    st.write(f"Repository: {repo['repo_name']}")
    st.write(f"Repository ID: {repo['repo_id']}")
    stats = repo["stats"]
    st.metric("Files discovered", stats["files_discovered"])
    st.metric("Files indexed", stats["files_indexed"])
    st.metric("Files skipped", stats["files_skipped"])
    st.metric("Chunks created", stats["chunks_created"])

    st.markdown("---")
    st.subheader("Ask a question")

    # Re-render earlier turns after Streamlit reruns the script.
    for message in st.session_state.conversation:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])
            if message.get("references"):
                for ref in message["references"]:
                    with st.expander(f"{ref['file_path']}:{ref['start_line']}-{ref['end_line']}"):
                        st.code(ref["snippet"])

    prompt = st.chat_input("Ask a question about the indexed repository")
    if prompt:
        # Ask the backend to retrieve code and generate a grounded response.
        st.session_state.conversation.append({"role": "user", "content": prompt, "references": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("Retrieving relevant code and generating an answer..."):
                result = call_api("POST", "/ask", {"question": prompt, "repo_id": repo["repo_id"]})
            answer_text = result["answer"]
            st.session_state.conversation.append({
                "role": "assistant",
                "content": answer_text,
                "references": result.get("references", []),
            })
            with st.chat_message("assistant"):
                st.markdown(answer_text)
                for ref in result.get("references", []):
                    with st.expander(f"{ref['file_path']}:{ref['start_line']}-{ref['end_line']}"):
                        st.code(ref["snippet"])
        except Exception as exc:
            st.session_state.conversation.append({
                "role": "assistant",
                "content": f"I hit an error while generating the answer: {exc}",
                "references": [],
            })
            with st.chat_message("assistant"):
                st.error(str(exc))
else:
    st.info("Index a public GitHub repository to begin asking questions.")
