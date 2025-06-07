import streamlit as st
from openai import OpenAI, OpenAIError
import mimetypes
import tiktoken
import uuid

# --- MODEL PRICING PER 1K TOKENS ---
MODEL_PRICING = {
    # GPT-4 family
    "gpt-4o": {"input": 0.005, "output": 0.015},            # GPT-4o (GPT-4 Turbo) approx pricing
    "gpt-4o-mini": {"input": 0.005, "output": 0.015},       # same as gpt-4o
    "gpt-4-turbo": {"input": 0.0015, "output": 0.003},      # GPT-4 Turbo pricing
    "gpt-4": {"input": 0.03, "output": 0.06},               # Standard GPT-4 pricing

    # GPT-3.5 family
    "gpt-3.5-turbo": {"input": 0.0004, "output": 0.0004},   # GPT-3.5 Turbo pricing
    "gpt-3.5-turbo-16k": {"input": 0.0006, "output": 0.0006}, # GPT-3.5 Turbo 16k context

    # GPT-4 with extended context
    "gpt-4-32k": {"input": 0.06, "output": 0.12},           # GPT-4 32k tokens
    "gpt-4-32k-turbo": {"input": 0.003, "output": 0.006},   # GPT-4 Turbo 32k tokens

    # Other common models (if used)
    "text-davinci-003": {"input": 0.02, "output": 0.02},    # Davinci (old)
    "text-curie-001": {"input": 0.002, "output": 0.002},    # Curie (old)
}


# --- COST LABELS FOR UI ---
def cost_label(model_name):
    price = MODEL_PRICING.get(model_name)
    if price:
        return f"{model_name} (${price['input']*1000:.3f}/K in, ${price['output']*1000:.3f}/K out)"
    return f"{model_name} (price unknown)"

# --- TOKEN COUNTING ---
def count_tokens(text, model="gpt-3.5-turbo"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# --- SESSION INITIALIZATION ---
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"messages": [], "cost": 0.0, "name": "Chat 1"}
    st.session_state.current_chat_id = new_id

# --- TITLE ---
st.title("💬 Smart Multi-Model Chatbot")

# --- API KEY INPUT ---
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please enter your OpenAI API key to continue.", icon="🗝️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# --- FETCH MODELS ---
try:
    models_response = client.models.list()
    available_models = sorted([m.id for m in models_response.data if "gpt" in m.id])
except OpenAIError as e:
    st.error(f"❌ Failed to fetch models: {e}")
    st.stop()

# --- MODEL SELECTION ---
default_model = "gpt-4o" if "gpt-4o" in available_models else "gpt-3.5-turbo"
if default_model not in available_models:
    default_model = available_models[0]

model_labels = [cost_label(m) for m in available_models]
selected_label = st.selectbox("🧠 Choose a model", model_labels, index=available_models.index(default_model))
selected_model = selected_label.split(" ")[0]

# --- CUSTOM MODEL INPUT ---
custom_model_input = st.text_input("Or type a custom model name (overrides above)")
model_name = custom_model_input.strip() if custom_model_input else selected_model

# --- CHAT MANAGEMENT SIDEBAR ---
with st.sidebar:
    st.header("💬 Chats")
    chat_ids = list(st.session_state.chats.keys())
    chat_names = {chat_id: st.session_state.chats[chat_id]["name"] for chat_id in chat_ids}
    
    # Calculate selected index based on current chat id
    selected_index = chat_ids.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in chat_ids else 0
    
    selected_chat_id = st.selectbox(
        "Select a chat",
        chat_ids,
        index=selected_index,
        format_func=lambda x: chat_names[x]
    )
    
    if selected_chat_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat_id

    # Add new chat
    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"messages": [], "cost": 0.0, "name": f"Chat {len(chat_ids) + 1}"}
        st.session_state.current_chat_id = new_id
        st.experimental_rerun()  # rerun to update UI and selectbox properly

    # Rename chat
    new_name = st.text_input("Rename chat", st.session_state.chats[st.session_state.current_chat_id]["name"])
    if new_name:
        st.session_state.chats[st.session_state.current_chat_id]["name"] = new_name

    # Delete chat
    if st.button("🗑️ Delete Chat"):
        del st.session_state.chats[st.session_state.current_chat_id]
        if st.session_state.chats:
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
        else:
            new_id = str(uuid.uuid4())
            st.session_state.chats[new_id] = {"messages": [], "cost": 0.0, "name": "Chat 1"}
            st.session_state.current_chat_id = new_id
        st.experimental_rerun()  # rerun to update UI after deletion

# --- FILE UPLOAD ---
uploaded_files = st.file_uploader(
    "📁 Upload files (images, PDFs, text)",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "txt", "pdf"]
)

# --- DISPLAY CHAT HISTORY ---
chat = st.session_state.chats[st.session_state.current_chat_id]
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- CHAT INPUT ---
if prompt := st.chat_input("Type your message..."):
    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- ATTACH FILE DETAILS ---
    file_descriptions = []
    for file in uploaded_files:
        file_type, _ = mimetypes.guess_type(file.name)
        if file_type and file_type.startswith("image/"):
            file_descriptions.append(f"📷 Image uploaded: {file.name}")
        elif file_type == "application/pdf":
            file_descriptions.append(f"📄 PDF uploaded: {file.name}")
        elif file_type and file_type.startswith("text/"):
            text = file.read().decode("utf-8", errors="ignore")
            file_descriptions.append(f"📄 Text file ({file.name}):\n{text[:1000]}...")
        file.seek(0)

    if file_descriptions:
        prompt += "\n\nAttached files:\n" + "\n".join(file_descriptions)
        chat["messages"][-1]["content"] = prompt

    try:
        # Token cost estimation
        model_cost = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
        input_tokens = sum(count_tokens(m["content"], model_name) for m in chat["messages"])
        input_cost = (input_tokens / 1000) * model_cost["input"]

        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in chat["messages"]],
            stream=True,
        )

        full_response = ""
        with st.chat_message("assistant"):
            for chunk in stream:
                delta = chunk.choices[0].delta.get("content", "") if hasattr(chunk.choices[0].delta, "get") else getattr(chunk.choices[0].delta, "content", "")
                full_response += delta
                st.write(delta, end="")

        output_tokens = count_tokens(full_response, model_name)
        output_cost = (output_tokens / 1000) * model_cost["output"]

        chat["messages"].append({"role": "assistant", "content": full_response})
        chat["cost"] += input_cost + output_cost

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

# --- COST DISPLAY ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"💰 **Estimated Cost**: `${chat['cost']:.4f}`")
