import streamlit as st
from openai import OpenAI, OpenAIError
import mimetypes
import tiktoken
import uuid
import logging
import time

logging.basicConfig(level=logging.DEBUG)

# --- MODEL PRICING PER 1K TOKENS ---
MODEL_PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.0015, "output": 0.003},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0004, "output": 0.0004},
    "gpt-3.5-turbo-16k": {"input": 0.0006, "output": 0.0006},
    "gpt-4-32k": {"input": 0.06, "output": 0.12},
    "gpt-4-32k-turbo": {"input": 0.003, "output": 0.006},
    "text-davinci-003": {"input": 0.02, "output": 0.02},
    "text-curie-001": {"input": 0.002, "output": 0.002},
}

def cost_label(model_name):
    price = MODEL_PRICING.get(model_name)
    if price:
        return f"{model_name} (${price['input']*1000:.3f}/K in, ${price['output']*1000:.3f}/K out)"
    return f"{model_name} (price unknown)"

def count_tokens(text, model="gpt-3.5-turbo"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"messages": [], "cost": 0.0, "name": "Chat 1"}
    st.session_state.current_chat_id = new_id

if "show_more_models" not in st.session_state:
    st.session_state.show_more_models = False

st.title("💬 Smart Multi-Model Chatbot")

openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please enter your OpenAI API key to continue.", icon="🗝️")
    st.stop()

try:
    client = OpenAI(api_key=openai_api_key)
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    st.stop()

try:
    models_response = client.models.list()
    all_models = sorted([m.id for m in models_response.data if "gpt" in m.id])
except OpenAIError as e:
    st.error(f"❌ Failed to fetch models: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error fetching models: {e}")
    st.stop()

if not all_models:
    st.error("No GPT models found for your API key.")
    st.stop()

logging.debug(f"Available GPT models: {all_models}")

default_models = ["gpt-4o-mini", "gpt-3.5-turbo"]
default_available = [m for m in default_models if m in all_models]
other_models = [m for m in all_models if m not in default_available]

if not st.session_state.show_more_models:
    display_models = default_available
else:
    display_models = default_available + other_models

model_labels = [cost_label(m) for m in display_models]

col1, col2 = st.columns([8, 1])
with col1:
    selected_label = st.selectbox("🧠 Choose a model", model_labels)
with col2:
    if st.button("🧩 More Models"):
        st.session_state.show_more_models = True

selected_model = selected_label.split(" ")[0]

custom_model_input = st.text_input("Or type a custom model name (overrides above)")
model_name = custom_model_input.strip() if custom_model_input else selected_model

with st.sidebar:
    st.header("💬 Chats")
    chat_names = {chat_id: st.session_state.chats[chat_id]["name"] for chat_id in st.session_state.chats}
    selected_chat_id = st.selectbox("Select a chat", chat_names.keys(), format_func=lambda x: chat_names[x])
    if selected_chat_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat_id

    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"messages": [], "cost": 0.0, "name": f"Chat {len(chat_names)+1}"}
        st.session_state.current_chat_id = new_id

    new_name = st.text_input("Rename chat", st.session_state.chats[selected_chat_id]["name"])
    if new_name and new_name != st.session_state.chats[selected_chat_id]["name"]:
        st.session_state.chats[selected_chat_id]["name"] = new_name

    if st.button("🗑️ Delete Chat"):
        if selected_chat_id in st.session_state.chats:
            del st.session_state.chats[selected_chat_id]
        if st.session_state.chats:
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
        else:
            new_id = str(uuid.uuid4())
            st.session_state.chats[new_id] = {"messages": [], "cost": 0.0, "name": "Chat 1"}
            st.session_state.current_chat_id = new_id

    st.markdown("---")
    chat = st.session_state.chats.get(st.session_state.current_chat_id, None)
    if chat:
        st.markdown(f"💰 **Estimated Cost**: `${chat['cost']:.4f}`")

uploaded_files = st.file_uploader(
    "📁 Upload files (images, PDFs, text)",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "txt", "pdf"]
)

chat = st.session_state.chats[st.session_state.current_chat_id]

for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    file_descriptions = []
    for file in uploaded_files:
        try:
            file_type, _ = mimetypes.guess_type(file.name)
            if file_type and file_type.startswith("image/"):
                file_descriptions.append(f"📷 Image uploaded: {file.name}")
            elif file_type == "application/pdf":
                file_descriptions.append(f"📄 PDF uploaded: {file.name}")
            elif file_type and file_type.startswith("text/"):
                try:
                    text = file.read().decode("utf-8", errors="ignore")
                    file_descriptions.append(f"📄 Text file ({file.name}):\n{text[:1000]}...")
                except Exception:
                    file_descriptions.append(f"📄 Text file ({file.name}): Unable to read content.")
            else:
                file_descriptions.append(f"📁 File uploaded: {file.name}")
        except Exception:
            file_descriptions.append(f"📁 File uploaded: {file.name} (error inspecting type)")
        finally:
            try:
                file.seek(0)
            except Exception:
                pass

    if file_descriptions:
        prompt += "\n\nAttached files:\n" + "\n".join(file_descriptions)
        chat["messages"][-1]["content"] = prompt

    try:
        model_cost = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
        input_tokens = sum(count_tokens(m["content"], model_name) for m in chat["messages"])
        input_cost = (input_tokens / 1000) * model_cost["input"]

        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in chat["messages"]],
            stream=True,
        )

        full_response = ""
        message_placeholder = st.empty()
        with st.chat_message("assistant"):
            for chunk in stream:
                delta = chunk.choices[0].delta.get("content", "")
                full_response += delta
                message_placeholder.markdown(full_response)
                # Tiny sleep to help Streamlit render intermediate chunks smoothly
                time.sleep(0.01)

        output_tokens = count_tokens(full_response, model_name)
        output_cost = (output_tokens / 1000) * model_cost["output"]

        chat["messages"].append({"role": "assistant", "content": full_response})
        chat["cost"] += input_cost + output_cost

    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
