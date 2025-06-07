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

# Default models (no API call needed)
DEFAULT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini", 
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]

def cost_label(model_name):
    """Generate a label showing model name and pricing."""
    price = MODEL_PRICING.get(model_name)
    if price:
        return f"{model_name} (${price['input']*1000:.3f}/K in, ${price['output']*1000:.3f}/K out)"
    return f"{model_name} (price unknown)"

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text for a given model."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {"messages": [], "cost": 0.0, "name": "Chat 1"}
        st.session_state.current_chat_id = new_id
    if "available_models" not in st.session_state:
        st.session_state.available_models = None
    if "models_fetched" not in st.session_state:
        st.session_state.models_fetched = False
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False

def fetch_available_models(client):
    """Fetch available models from OpenAI API."""
    try:
        models_response = client.models.list()
        fetched_models = sorted([m.id for m in models_response.data if "gpt" in m.id])
        return fetched_models, None
    except Exception as e:
        return None, str(e)

def process_uploaded_files(uploaded_files):
    """Process uploaded files and return descriptions."""
    file_descriptions = []
    if not uploaded_files:
        return file_descriptions
    
    for file in uploaded_files:
        try:
            file_type, _ = mimetypes.guess_type(file.name)
            if file_type and file_type.startswith("text/"):
                try:
                    content = file.read().decode("utf-8", errors="ignore")
                    truncated_content = content[:1000]
                    if len(content) > 1000:
                        truncated_content += "..."
                    file_descriptions.append(f"📄 {file.name}:\n{truncated_content}")
                    file.seek(0)  # Reset file pointer
                except Exception as e:
                    file_descriptions.append(f"📄 {file.name}: Error reading file - {e}")
            elif file_type and file_type.startswith("image/"):
                file_descriptions.append(f"📷 Image: {file.name}")
            elif file_type == "application/pdf":
                file_descriptions.append(f"📄 PDF: {file.name}")
            else:
                file_descriptions.append(f"📁 File: {file.name} (type: {file_type or 'unknown'})")
        except Exception as e:
            file_descriptions.append(f"📁 {file.name}: Error processing - {e}")
    
    return file_descriptions

def calculate_cost(messages, response_text, model_name):
    """Calculate the cost of a conversation."""
    model_cost = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
    
    # Calculate input tokens
    input_tokens = sum(count_tokens(m["content"], model_name) for m in messages)
    input_cost = (input_tokens / 1000) * model_cost["input"]
    
    # Calculate output tokens
    output_tokens = count_tokens(response_text, model_name)
    output_cost = (output_tokens / 1000) * model_cost["output"]
    
    return input_cost, output_cost, input_tokens, output_tokens

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Set page config
    st.set_page_config(
        page_title="Smart Multi-Model Chatbot",
        page_icon="💬",
        layout="wide"
    )
    
    # App header
    st.title("💬 Smart Multi-Model Chatbot")
    st.markdown("Chat with OpenAI's GPT models with cost tracking and file upload support.")
    
    # Debug toggle in sidebar
    with st.sidebar:
        st.session_state.show_debug = st.checkbox("🔧 Show Debug Info", value=st.session_state.show_debug)
    
    # Debug info
    if st.session_state.show_debug:
        with st.expander("🔧 Debug Information"):
            st.write("**Session State Keys:**", list(st.session_state.keys()))
            st.write("**Models Fetched:**", st.session_state.models_fetched)
            st.write("**Available Models:**", st.session_state.available_models)
            st.write("**Current Chat ID:**", st.session_state.current_chat_id)
            st.write("**Number of Chats:**", len(st.session_state.chats))
    
    # API Key input
    openai_api_key = st.text_input("🔑 OpenAI API Key", type="password", help="Enter your OpenAI API key to get started")
    
    if not openai_api_key:
        st.info("Please enter your OpenAI API key to continue.", icon="🗝️")
        st.markdown("""
        **How to get an API key:**
        1. Go to [OpenAI's website](https://platform.openai.com/api-keys)
        2. Sign in or create an account
        3. Navigate to API Keys section
        4. Create a new secret key
        5. Copy and paste it above
        """)
        st.stop()
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=openai_api_key)
        st.success("✅ OpenAI client initialized successfully")
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI client: {e}")
        st.stop()
    
    # Model selection section
    st.subheader("🧠 Model Selection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not st.session_state.models_fetched:
            model_source = st.radio(
                "Choose model source:",
                ["Use default models (recommended)", "Fetch my available models"],
                help="Default models load faster, fetching shows only models available to your API key"
            )
        else:
            model_source = "Using fetched models"
            st.info("Using models fetched from your API key")
    
    with col2:
        if st.button("🔄 Refresh", help="Reset and fetch models again"):
            st.session_state.models_fetched = False
            st.session_state.available_models = None
            st.rerun()
    
    # Handle model fetching
    if model_source == "Fetch my available models" and not st.session_state.models_fetched:
        with st.spinner("🔍 Fetching your available models..."):
            fetched_models, error = fetch_available_models(client)
            
            if error:
                st.error(f"❌ Error fetching models: {error}")
                st.info("💡 Falling back to default models")
                st.session_state.available_models = DEFAULT_MODELS
            else:
                st.session_state.available_models = fetched_models
                st.success(f"✅ Found {len(fetched_models)} models")
            
            st.session_state.models_fetched = True
    
    # Determine which models to show
    if st.session_state.models_fetched and st.session_state.available_models:
        display_models = st.session_state.available_models
        st.info(f"📋 Using {len(display_models)} fetched models")
    else:
        display_models = DEFAULT_MODELS
        st.info(f"📋 Using {len(display_models)} default models")
    
    # Model selection dropdown
    model_labels = [cost_label(m) for m in display_models]
    selected_label = st.selectbox(
        "Choose a model:",
        model_labels,
        help="Different models have different capabilities and costs"
    )
    selected_model = selected_label.split(" ")[0] if selected_label else display_models[0]
    
    # Custom model input
    custom_model = st.text_input(
        "Or enter a custom model name:",
        help="Override the selection above with any model name"
    )
    model_name = custom_model.strip() if custom_model else selected_model
    
    st.success(f"🎯 Using model: **{model_name}**")
    
    # File upload section
    st.subheader("📁 File Upload")
    uploaded_files = st.file_uploader(
        "Upload files to include in your conversation:",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "txt", "pdf"],
        help="Supported: Images (PNG, JPG), Text files (TXT), and PDFs"
    )
    
    if uploaded_files:
        st.success(f"📎 {len(uploaded_files)} file(s) uploaded")
        for file in uploaded_files:
            st.caption(f"• {file.name} ({file.size} bytes)")
    
    # Sidebar for chat management
    with st.sidebar:
        st.header("💬 Chat Management")
        
        # Current chat info
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        st.info(f"**Current:** {current_chat['name']}")
        
        # Chat selection
        chat_names = {chat_id: st.session_state.chats[chat_id]["name"] for chat_id in st.session_state.chats}
        
        if len(chat_names) > 1:
            selected_chat_id = st.selectbox(
                "Switch to chat:",
                list(chat_names.keys()),
                format_func=lambda x: chat_names[x],
                index=list(chat_names.keys()).index(st.session_state.current_chat_id)
            )
            
            if selected_chat_id != st.session_state.current_chat_id:
                st.session_state.current_chat_id = selected_chat_id
                st.rerun()
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            chat_count = len(st.session_state.chats) + 1
            st.session_state.chats[new_id] = {
                "messages": [], 
                "cost": 0.0, 
                "name": f"Chat {chat_count}"
            }
            st.session_state.current_chat_id = new_id
            st.rerun()
        
        # Rename chat
        with st.expander("✏️ Rename Chat"):
            new_name = st.text_input(
                "New name:",
                value=current_chat["name"],
                key=f"rename_{st.session_state.current_chat_id}"
            )
            if st.button("💾 Save", key="save_name"):
                st.session_state.chats[st.session_state.current_chat_id]["name"] = new_name
                st.rerun()
        
        # Delete chat (only if more than one exists)
        if len(st.session_state.chats) > 1:
            with st.expander("🗑️ Delete Chat"):
                st.warning("This action cannot be undone!")
                if st.button("🗑️ Confirm Delete", type="secondary"):
                    del st.session_state.chats[st.session_state.current_chat_id]
                    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                    st.rerun()
        
        st.markdown("---")
        
        # Chat statistics
        st.subheader("📊 Chat Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Cost", f"${current_chat['cost']:.4f}")
        with col2:
            st.metric("💬 Messages", len(current_chat["messages"]))
        
        # Total stats across all chats
        total_cost = sum(chat["cost"] for chat in st.session_state.chats.values())
        total_messages = sum(len(chat["messages"]) for chat in st.session_state.chats.values())
        
        st.markdown("**Total Across All Chats:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Total Cost", f"${total_cost:.4f}")
        with col2:
            st.metric("💬 Total Messages", total_messages)
    
    # Main chat interface
    st.subheader("💬 Conversation")
    
    # Get current chat
    chat = st.session_state.chats[st.session_state.current_chat_id]
    
    # Display chat messages
    if not chat["messages"]:
        st.info("👋 Start a conversation by typing a message below!")
    
    for i, msg in enumerate(chat["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        chat["messages"].append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process uploaded files
        file_descriptions = process_uploaded_files(uploaded_files)
        
        # Add file content to the user's message if any files were uploaded
        if file_descriptions:
            file_content = "\n\n**Uploaded files:**\n" + "\n".join(file_descriptions)
            chat["messages"][-1]["content"] += file_content
        
        # Generate AI response
        with st.chat_message("assistant"):
            try:
                # Show thinking indicator
                with st.spinner(f"🤔 {model_name} is thinking..."):
                    # Prepare messages for API
                    messages = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]
                    
                    # Create completion
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                        temperature=0.7
                    )
                    
                    ai_response = response.choices[0].message.content
                
                # Display the response
                st.markdown(ai_response)
                
                # Calculate costs
                input_cost, output_cost, input_tokens, output_tokens = calculate_cost(
                    messages, ai_response, model_name
                )
                total_cost = input_cost + output_cost
                
                # Update chat with AI response
                chat["messages"].append({"role": "assistant", "content": ai_response})
                chat["cost"] += total_cost
                
                # Show cost information
                st.caption(
                    f"💰 Cost: ${total_cost:.4f} | "
                    f"📝 Tokens: {input_tokens} → {output_tokens} | "
                    f"🧠 Model: {model_name}"
                )
                
            except OpenAIError as e:
                st.error(f"🚫 OpenAI API Error: {e}")
                logger.error(f"OpenAI error: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {e}")
                logger.error(f"General error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** "
        "• Try different models to compare responses and costs "
        "• Upload text files to analyze content "
        "• Use the sidebar to manage multiple conversations"
    )

if __name__ == "__main__":
    main()
