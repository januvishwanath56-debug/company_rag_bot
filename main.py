# main.py
# Streamlit Web UI for the RAG Customer Support Bot
# Run with: streamlit run main.py

import os
import sys
import streamlit as st

# Make sure src/ imports work from root folder
sys.path.insert(0, os.path.dirname(__file__))

from src.rag_pipeline import build_graph, ingest_document, SupportState

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TechCorp Support Bot",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .chat-user {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        text-align: right;
        color: #1a1a2e;
    }
    .chat-bot {
        background-color: #f3e5f5;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        color: #1a1a2e;
    }
    .chat-hitl {
        background-color: #fff3e0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        border-left: 4px solid #ff9800;
        color: #1a1a2e;
    }
    .escalation-box {
        background-color: #fff8e1;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-auto {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .badge-human {
        background-color: #ffe0b2;
        color: #e65100;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # list of message dicts

if "ingested" not in st.session_state:
    st.session_state.ingested = False       # has knowledge base been loaded?

if "graph" not in st.session_state:
    st.session_state.graph = None           # compiled LangGraph

if "pending_escalation" not in st.session_state:
    st.session_state.pending_escalation = None   # holds state waiting for human reply

if "ingestion_info" not in st.session_state:
    st.session_state.ingestion_info = ""

# ─────────────────────────────────────────────
# SIDEBAR — Knowledge Base Setup
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Setup")
    st.markdown("---")

    st.markdown("### 📄 Knowledge Base")

    # Option 1: Use default knowledge base
    if st.button("✅ Use Default (TechCorp FAQ)", use_container_width=True):
        default_path = os.path.join("data", "knowledge_base.txt")
        if not os.path.exists(default_path):
            st.error("data/knowledge_base.txt not found. Make sure the file is in the data/ folder.")
        else:
            with st.spinner("Ingesting knowledge base..."):
                try:
                    count = ingest_document(default_path)
                    st.session_state.ingested       = True
                    st.session_state.graph          = build_graph()
                    st.session_state.ingestion_info = f"✅ Loaded {count} chunks from TechCorp FAQ"
                    st.success(f"Ready! {count} chunks loaded.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("**— or —**")

    # Option 2: Upload a PDF
    uploaded = st.file_uploader("Upload your own PDF", type=["pdf", "txt"])
    if uploaded and st.button("📥 Ingest Uploaded File", use_container_width=True):
        save_path = os.path.join("data", uploaded.name)
        os.makedirs("data", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner("Ingesting uploaded file..."):
            try:
                count = ingest_document(save_path)
                st.session_state.ingested       = True
                st.session_state.graph          = build_graph()
                st.session_state.ingestion_info = f"✅ Loaded {count} chunks from {uploaded.name}"
                st.success(f"Ready! {count} chunks loaded.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.ingestion_info:
        st.info(st.session_state.ingestion_info)

    st.markdown("---")
    st.markdown("### 🗑️ Clear Chat")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history       = []
        st.session_state.pending_escalation = None
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Tech Stack:**
    - 🔗 LangGraph (workflow)
    - 🧠 Groq LLaMA3 (LLM)
    - 📦 ChromaDB (vector DB)
    - 🔢 MiniLM (embeddings)
    - 👤 HITL escalation
    """)

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 TechCorp Customer Support Bot</h1>
    <p>RAG-powered assistant with LangGraph & Human-in-the-Loop escalation</p>
</div>
""", unsafe_allow_html=True)

# Show warning if not ingested yet
if not st.session_state.ingested:
    st.warning("👈 Please load a knowledge base from the sidebar first before chatting.")

# ─────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)

    elif msg["role"] == "bot":
        badge = '<span class="status-badge badge-auto">🤖 AI</span>'
        st.markdown(f'<div class="chat-bot">{badge} <b>Bot:</b> {msg["content"]}</div>', unsafe_allow_html=True)

    elif msg["role"] == "hitl":
        badge = '<span class="status-badge badge-human">👤 Human Agent</span>'
        st.markdown(f'<div class="chat-hitl">{badge} <b>Agent:</b> {msg["content"]}</div>', unsafe_allow_html=True)

    elif msg["role"] == "escalation_notice":
        st.markdown(f"""
        <div class="escalation-box">
            ⚠️ <b>Escalated to Human Agent</b><br>
            <small>Reason: {msg["content"]}</small>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HUMAN AGENT INPUT (shown when escalation is pending)
# ─────────────────────────────────────────────
if st.session_state.pending_escalation is not None:
    st.markdown("---")
    st.markdown("### 🧑‍💼 Human Agent Response Required")
    st.markdown(f"**User asked:** {st.session_state.pending_escalation['user_query']}")

    with st.form("hitl_form"):
        agent_reply = st.text_area(
            "Type your response as the human agent:",
            placeholder="Type a helpful response for the customer...",
            height=120
        )
        submitted = st.form_submit_button("✅ Send Agent Response")

        if submitted:
            reply = agent_reply.strip()
            if not reply:
                reply = "Thank you for reaching out. Our support team will contact you within 24 hours."

            # Record the human reply in chat history
            st.session_state.chat_history.append({
                "role":    "hitl",
                "content": reply
            })

            # Clear the pending escalation
            st.session_state.pending_escalation = None
            st.rerun()

# ─────────────────────────────────────────────
# USER INPUT — Chat box at bottom
# ─────────────────────────────────────────────
st.markdown("---")

if st.session_state.ingested and st.session_state.pending_escalation is None:
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Your message:",
                placeholder="Ask me anything about TechCorp...",
                label_visibility="collapsed"
            )
        with col2:
            send = st.form_submit_button("Send 📨", use_container_width=True)

        if send and user_input.strip():
            query = user_input.strip()

            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Build initial graph state
            initial_state: SupportState = {
                "user_query":        query,
                "retrieved_chunks":  [],
                "intent":            "",
                "confidence":        "",
                "escalation_reason": None,
                "llm_response":      None,
                "final_response":    None,
                "hitl_input":        None
            }

            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.graph.invoke(initial_state)

                    if result["final_response"] == "__HITL_WAITING__":
                        # Escalation triggered — need human input
                        st.session_state.chat_history.append({
                            "role":    "escalation_notice",
                            "content": result.get("escalation_reason", "Query needs human attention.")
                        })
                        st.session_state.pending_escalation = {
                            "user_query": query,
                            "state":      result
                        }
                    else:
                        # Normal bot answer
                        st.session_state.chat_history.append({
                            "role":    "bot",
                            "content": result["final_response"]
                        })

                except Exception as e:
                    st.session_state.chat_history.append({
                        "role":    "bot",
                        "content": f"Sorry, an error occurred: {str(e)}"
                    })

            st.rerun()

elif not st.session_state.ingested:
    st.info("Load a knowledge base from the sidebar to start chatting.")

elif st.session_state.pending_escalation is not None:
    st.info("⏳ Waiting for human agent response above before accepting new messages.")

# ─────────────────────────────────────────────
# SAMPLE QUESTIONS
# ─────────────────────────────────────────────
if st.session_state.ingested and st.session_state.pending_escalation is None:
    st.markdown("---")
    st.markdown("**💡 Try these sample questions:**")
    cols = st.columns(3)
    samples = [
        "How do I reset my password?",
        "What is the price of the Pro plan?",
        "How do I enable two-factor authentication?",
        "I want a refund, I was charged wrongly!",
        "How do I connect Slack?",
        "What payment methods are accepted?"
    ]
    for i, sample in enumerate(samples):
        with cols[i % 3]:
            if st.button(sample, key=f"sample_{i}", use_container_width=True):
                query = sample
                st.session_state.chat_history.append({"role": "user", "content": query})

                initial_state: SupportState = {
                    "user_query":        query,
                    "retrieved_chunks":  [],
                    "intent":            "",
                    "confidence":        "",
                    "escalation_reason": None,
                    "llm_response":      None,
                    "final_response":    None,
                    "hitl_input":        None
                }

                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.graph.invoke(initial_state)
                        if result["final_response"] == "__HITL_WAITING__":
                            st.session_state.chat_history.append({
                                "role":    "escalation_notice",
                                "content": result.get("escalation_reason", "Needs human attention.")
                            })
                            st.session_state.pending_escalation = {
                                "user_query": query,
                                "state":      result
                            }
                        else:
                            st.session_state.chat_history.append({
                                "role":    "bot",
                                "content": result["final_response"]
                            })
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "bot",
                            "content": f"Error: {str(e)}"
                        })
                st.rerun()
