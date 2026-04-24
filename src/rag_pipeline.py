# src/rag_pipeline.py
# LangGraph workflow: Input → Router → [LLM or HITL] → Output

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from src.retriever import retrieve_chunks, store_chunks
from src.llm import classify_intent, generate_answer
from src.loader import load_document
from src.chunker import chunk_text

# ─────────────────────────────────────────────
# STATE — data passed between all nodes
# ─────────────────────────────────────────────
class SupportState(TypedDict):
    user_query:        str
    retrieved_chunks:  list
    intent:            str
    confidence:        str
    escalation_reason: Optional[str]
    llm_response:      Optional[str]
    final_response:    Optional[str]
    hitl_input:        Optional[str]   # human agent's typed reply (from Streamlit)

# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def input_node(state: SupportState) -> SupportState:
    """Retrieves relevant chunks from ChromaDB for the user query."""
    chunks = retrieve_chunks(state["user_query"], top_k=3)
    state["retrieved_chunks"] = chunks
    return state

def router_node(state: SupportState) -> SupportState:
    """Classifies intent: answerable or escalate."""
    result = classify_intent(state["user_query"], state["retrieved_chunks"])
    state["intent"]            = result["intent"]
    state["confidence"]        = result["confidence"]
    state["escalation_reason"] = result["reason"]
    return state

def llm_node(state: SupportState) -> SupportState:
    """Generates an answer using Groq LLM + retrieved context."""
    answer = generate_answer(state["user_query"], state["retrieved_chunks"])
    state["llm_response"]   = answer
    state["final_response"] = answer
    return state

def hitl_node(state: SupportState) -> SupportState:
    """
    Human-in-the-Loop node.
    In Streamlit mode: reads the human reply from state["hitl_input"].
    If no human input provided yet, sets a waiting message.
    """
    human_reply = state.get("hitl_input", "").strip()

    if human_reply:
        state["final_response"] = f"🧑‍💼 Human Agent: {human_reply}"
    else:
        # This means we need to ask the human — Streamlit UI will handle this
        state["final_response"] = "__HITL_WAITING__"

    return state

def output_node(state: SupportState) -> SupportState:
    """Final node — response is already set, just passes through."""
    return state

# ─────────────────────────────────────────────
# ROUTING FUNCTION
# ─────────────────────────────────────────────
def route_decision(state: SupportState) -> str:
    return "llm_node" if state.get("intent") == "answerable" else "hitl_node"

# ─────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("input_node",  input_node)
    graph.add_node("router_node", router_node)
    graph.add_node("llm_node",    llm_node)
    graph.add_node("hitl_node",   hitl_node)
    graph.add_node("output_node", output_node)

    graph.set_entry_point("input_node")
    graph.add_edge("input_node", "router_node")
    graph.add_conditional_edges(
        "router_node",
        route_decision,
        {"llm_node": "llm_node", "hitl_node": "hitl_node"}
    )
    graph.add_edge("llm_node",   "output_node")
    graph.add_edge("hitl_node",  "output_node")
    graph.add_edge("output_node", END)

    return graph.compile()

# ─────────────────────────────────────────────
# INGESTION FUNCTION (called from Streamlit sidebar)
# ─────────────────────────────────────────────
def ingest_document(filepath: str) -> int:
    """
    Full ingestion pipeline: load → chunk → embed → store in ChromaDB.
    Returns the number of chunks stored.
    """
    text   = load_document(filepath)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    count  = store_chunks(chunks)
    return count
