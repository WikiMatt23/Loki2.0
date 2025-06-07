import streamlit as st
from openai import OpenAI, OpenAIError
import mimetypes
import tiktoken
import uuid
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

# App starts here
st.title("💬 Smart Multi-Model Chatbot")

# API Key input
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please enter your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# Initialize OpenAI client with better error handling
try:
    client = OpenAI(api_key=openai_api_key)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {e}")
    logger.error(f"Client initialization error: {e}")
    st.stop()

# Fetch models with timeout and better error handling
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_available_models(api_key):
    try:
        temp_client = OpenAI(api_key=api_key)
        models_response = temp_client.models.list()
        available_models = sorted([m.id for m in models_response.data if "gpt" in m.id])
        logger.info(f"Successfully fetched {len(available_models)} models")
        return available_models, None
    except OpenAIError as e:
        error_msg = f"OpenAI API error: {e}"
        logger.error(error_msg)
        return [], error_msg
    except Exception as e:
        error_msg = f"Unexpected error fetching models: {e}"
        logger.error(error_msg)
        return [], error_msg

# Show loading spinner while fetching models
with st.spinner("Loading available models..."):
    all_models, error = fetch_available_models(openai_api_key)

if error:
    st.error(f"❌ {error}")
    st.info("Using default models instead.")
    # Fallback to common models
    all_models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

if not all_models:
    st.error("No GPT models available. Please check your API key.")
    st.stop()

logger.debug(f"Available GPT models: {all_models}")

# Model selection logic
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
        st.rerun()

selected_model = selected_label.split(" ")[0] if selected_label else "gpt-3.5-turbo"

custom_model_input = st.text_input("Or type a custom model name (overrides above)")
model_name = custom_model_input.strip() if custom_model_input else selected_model

# Sidebar for chat management
with st.sidebar:
    st.header("💬 Chats")
    
    # Chat selection
    chat_names = {chat_id: st.session_state.chats[chat_id]["name"] for chat_id in st.session_state.chats}
    selected_chat_id = st.selectbox("Select a chat", chat_names.keys(), format_func=lambda x: chat_names[x])
    
    if selected_chat_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat_id
        st.rerun()

    # New chat button
    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"messages": [], "cost": 0.0, "name": f"Chat {len(chat_names)+1}"}
        st.session_state.current_chat_id = new_id
        st.rerun()

    # Rename chat
    new_name = st.text_input("Rename chat", st.session_state.chats[selected_chat_id]["name"])
    if new_name and new_name != st.session_state.chats[selected_chat_id]["name"]:
        st.session_state.chats[selected_chat_id]["name"] = new_name
        st.rerun()

    # Delete chat
    if st.button("🗑️ Delete Chat") and len(st.session_state.chats) > 1:
        if selected_chat_id in st.session_state.chats:
            del st.session_state.chats[selected_chat_id]
        if st.session_state.chats:
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
        else:
            new_id = str(uuid.uuid4())
            st.session_state.chats[new_id] = {"messages": [], "cost": 0.0, "name": "Chat 1"}
            st.session_state.current_chat_id = new_id
        st.rerun()

    st.markdown("---")
    
    # Display cost
    chat = st.session_state.chats.get(st.session_state.current_chat_id, None)
    if chat:
        st.markdown(f"💰 **Estimated Cost**: `${chat['cost']:.4f}`")

# File upload
uploaded_files = st.file_uploader(
    "📁 Upload files (images, PDFs, text)",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "txt", "pdf"]
)

# Get current chat
chat = st.session_state.chats[st.session_state.current_chat_id]

# Display chat messages
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process uploaded files
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
                except Exception as e:
                    logger.error(f"Error reading text file {file.name}: {e}")
                    file_descriptions.append(f"📄 Text file ({file.name}): Unable to read content.")
            else:
                file_descriptions.append(f"📁 File uploaded: {file.name}")
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            file_descriptions.append(f"📁 File uploaded: {file.name} (error inspecting type)")
        finally:
            try:
                file.seek(0)
            except Exception:
                pass

    # Add file descriptions to prompt if any
    if file_descriptions:
        prompt += "\n\nAttached files:\n" + "\n".join(file_descriptions)
        chat["messages"][-1]["content"] = prompt

    # Generate response
    try:
        # Calculate costs
        model_cost = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
        input_tokens = sum(count_tokens(m["content"], model_name) for m in chat["messages"])
        input_cost = (input_tokens / 1000) * model_cost["input"]

        # Create streaming response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": m["role"], "content": m["content"]} for m in chat["messages"]],
                    stream=True,
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)  # Small delay for smooth rendering
                
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                message_placeholder.error(f"Error generating response: {e}")
                full_response = f"Error: {e}"

        # Calculate output cost and update
        if full_response and not full_response.startswith("Error:"):
            output_tokens = count_tokens(full_response, model_name)
            output_cost = (output_tokens / 1000) * model_cost["output"]
            
            chat["messages"].append({"role": "assistant", "content": full_response})
            chat["cost"] += input_cost + output_cost
        
    except OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        logger.error(f"OpenAI API error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
