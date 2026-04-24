# src/llm.py
# Handles all calls to the Groq LLM API

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"   # updated model - llama3-8b-8192 is decommissioned

_client = None

def get_groq_client():
    """Returns Groq client, initializing only once."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file.")
        _client = Groq(api_key=api_key)
    return _client

def classify_intent(query: str, context_chunks: list) -> dict:
    """
    Uses the LLM to decide if the query can be answered from context,
    or needs to be escalated to a human agent.
    Returns a dict: { intent, confidence, reason }
    """
    if not context_chunks:
        return {
            "intent":     "escalate",
            "confidence": "low",
            "reason":     "No relevant information found in knowledge base."
        }

    client       = get_groq_client()
    context_text = "\n---\n".join(context_chunks)

    prompt = f"""You are a customer support routing system.

User asked: "{query}"

Retrieved context:
{context_text}

Can this question be answered using ONLY the context above?

Reply ONLY in this exact format:
INTENT: answerable
CONFIDENCE: high
REASON: one short sentence

Rules:
- INTENT must be: answerable OR escalate
- CONFIDENCE must be: high OR medium OR low
- Use escalate if: question is not in context, is a complaint, needs account access, or is complex/emotional
- Use low confidence if context is only vaguely related"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0
    )

    raw    = response.choices[0].message.content.strip()
    result = {"intent": "escalate", "confidence": "low", "reason": "Could not classify."}

    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("INTENT:"):
            val = line.split(":", 1)[1].strip().lower()
            result["intent"] = "answerable" if "answerable" in val else "escalate"
        elif line.upper().startswith("CONFIDENCE:"):
            val = line.split(":", 1)[1].strip().lower()
            result["confidence"] = "high" if "high" in val else ("medium" if "medium" in val else "low")
        elif line.upper().startswith("REASON:"):
            result["reason"] = line.split(":", 1)[1].strip()

    # Safety: low confidence → escalate
    if result["intent"] == "answerable" and result["confidence"] == "low":
        result["intent"] = "escalate"
        result["reason"] = "Confidence too low — " + result["reason"]

    return result

def generate_answer(query: str, context_chunks: list) -> str:
    """
    Generates a customer support answer using retrieved context chunks.
    Returns the answer as a string.
    """
    client       = get_groq_client()
    context_text = "\n---\n".join(context_chunks)

    prompt = f"""You are a helpful customer support agent for TechCorp.
Answer the user's question using ONLY the information in the context below.
Be concise, friendly, and clear. Do NOT make up anything not in the context.

Context:
{context_text}

User question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()