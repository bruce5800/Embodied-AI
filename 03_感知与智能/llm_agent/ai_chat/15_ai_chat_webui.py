import json
import time
from typing import Dict, List, Generator

import streamlit as st
import requests

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="AI Chat WebUI (Ollama)",
    page_icon="🤖",
    layout="centered",
)

# Simple, clean style tweaks
CUSTOM_CSS = """
<style>
/**** Global tweaks ****/
:root { --primary: #3b82f6; }
html, body, [class*="css"] { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }

/**** Chat message cards ****/
.stChatMessage .stMarkdown { line-height: 1.55; }
.stChatMessage[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p { margin: 0.2rem 0; }
.stChatMessage[data-testid="stChatMessage"] {
  border-radius: 12px;
  padding: 0.5rem 0.75rem;
}
.stChatMessage[data-testid="stChatMessage"]:has([data-testid="user-avatar"]) { background: #f8fafc; }
.stChatMessage[data-testid="stChatMessage"]:has([data-testid="assistant-avatar"]) { background: #f1f5f9; }

/**** Input box ****/
section[data-testid="stChatInput"] textarea {
  border-radius: 12px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def list_ollama_models(base_url: str) -> List[str]:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def stream_ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    options: Dict,
) -> Generator[str, None, None]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options or {},
    }
    try:
        with requests.post(
            f"{base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message", {})
                content = msg.get("content")
                if content:
                    yield content
                if data.get("done"):
                    break
    except requests.RequestException as e:
        raise RuntimeError(f"与 Ollama 通信失败: {e}")


# ---------------------------
# Sidebar settings
# ---------------------------
st.sidebar.title("设置aaaa")
base_url = st.sidebar.text_input("Ollama 地址", value="http://localhost:11434", help="默认 Ollama 本地 API 地址")

models = list_ollama_models(base_url)
if models:
    model = st.sidebar.selectbox("选择模型", options=models, index=0)
else:
    model = st.sidebar.text_input("模型名称", value="llama3.1:8b", help="未能获取模型列表时手动输入")

system_prompt = st.sidebar.text_area("系统提示词", value="You are a helpful assistant.", height=100)

col1, col2 = st.sidebar.columns(2)
with col1:
    temperature = st.slider("概率温度", 0.0, 1.0, 0.7, 0.05)
with col2:
    top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)

if st.sidebar.button("清空对话", type="primary"):
    st.session_state.messages = []
    st.experimental_rerun()

# ---------------------------
# Chat state init
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prepend system message when present (only for API call, not rendered)
def build_api_messages() -> List[Dict[str, str]]:
    api_msgs = []
    if system_prompt.strip():
        api_msgs.append({"role": "system", "content": system_prompt.strip()})
    api_msgs.extend(st.session_state.messages)
    return api_msgs

# ---------------------------
# Header
# ---------------------------
st.title("🤖 AI Chat WebUI")
st.caption("连接本地 Ollama，大模型对话界面简洁美观")

# ---------------------------
# Render existing messages
# ---------------------------
for m in st.session_state.messages:
    avatar = "🧑" if m["role"] == "user" else "🤖"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# ---------------------------
# Input and stream reply
# ---------------------------
user_input = st.chat_input("输入你的问题…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Render the user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Assistant streaming container
    with st.chat_message("assistant", avatar="🤖"):
        try:
            options = {"temperature": temperature, "top_p": top_p}
            # Use Streamlit write_stream for smooth streaming
            def generator():
                for chunk in stream_ollama_chat(base_url, model, build_api_messages(), options):
                    yield chunk
            full_text = st.write_stream(generator())
        except Exception as e:
            st.error(str(e))
            full_text = ""

    # Persist assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_text or ""})

# Footer tip
st.caption(f"当前模型: `{model}` | API: `{base_url}`")