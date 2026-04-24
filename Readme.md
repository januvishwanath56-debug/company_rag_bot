# 🤖 RAG-Based Customer Support Assistant

> Built with LangGraph · ChromaDB · Groq LLaMA 3.1 · Streamlit · HuggingFace Embeddings

A production-style **Retrieval-Augmented Generation (RAG)** system that answers customer queries from a knowledge base and escalates complex queries to a human agent via **Human-in-the-Loop (HITL)** — built as part of the Innomatics Research Labs Internship Project.

---



## 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) combines:
- A **retrieval system** that finds relevant chunks from a document
- A **generation system** (LLM) that produces answers grounded in those chunks

This prevents hallucinations and enables accurate, domain-specific answers without retraining the model.

---

## ⚙️ System Architecture

```
PDF / TXT File
      │
      ▼
  loader.py  ──►  chunker.py  ──►  embeddings.py  ──►  ChromaDB
                                                          │
User Query ──► embed query ──► retrieve top-3 chunks ◄───┘
                                        │
                                  LangGraph Workflow
                                        │
                              ┌─────────▼──────────┐
                              │    input_node       │  retrieve chunks
                              └─────────┬───────────┘
                                        │
                              ┌─────────▼───────────┐
                              │    router_node       │  classify intent
                              └────┬────────────┬───┘
                                   │            │
                          answerable          escalate
                                   │            │
                        ┌──────────▼──┐   ┌────▼──────────┐
                        │  llm_node   │   │   hitl_node   │
                        │ Groq LLaMA  │   │ Human Agent   │
                        └──────────┬──┘   └────┬──────────┘
                                   │            │
                              ┌────▼────────────▼───┐
                              │     output_node      │
                              └──────────┬───────────┘
                                         │
                                   Streamlit UI
```

---

## 📁 Project Structure

```
rag_support_bot/
│
├── src/
│   ├── __init__.py          # makes src a Python package
│   ├── loader.py            # reads PDF / TXT knowledge base
│   ├── chunker.py           # splits text into overlapping chunks
│   ├── embeddings.py        # HuggingFace MiniLM embedding model
│   ├── retriever.py         # ChromaDB store + retrieval
│   ├── llm.py               # Groq LLM — intent classification + answer generation
│   └── rag_pipeline.py      # LangGraph workflow + ingestion pipeline
│
├── data/
│   └── knowledge_base.txt   # TechCorp FAQ knowledge base
│
├── screenshots/             # UI screenshots for README
├── main.py                  # Streamlit web app (entry point)
├── requirements.txt         # all dependencies
├── .env                     # API key (not committed to git)
├── .gitignore               # excludes venv, chroma_db, .env
└── README.md                # this file
```

---

## 🛠️ Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| LLM | Groq llama-3.1-8b-instant | Free, fast (<1s), strong reasoning |
| Embeddings | all-MiniLM-L6-v2 | Lightweight, runs on CPU, high quality |
| Vector DB | ChromaDB | Local, persistent, no API key needed |
| Workflow | LangGraph | Stateful graph with conditional routing |
| UI | Streamlit | Pure Python web app, rapid development |
| Env Vars | python-dotenv | Secure API key management |

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/rag-support-bot.git
cd rag-support-bot
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free API key from [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run main.py --server.fileWatcherType none
```

App opens at `http://localhost:8501`

---

## 💬 How to Use

1. Open the app in your browser
2. In the **sidebar**, click **" Use Default (TechCorp FAQ)"** to load the knowledge base
3. Wait for **"Ready! X chunks loaded"** message
4. Start chatting in the input box

### Try these sample queries:

| Query | Expected Behaviour |
|-------|-------------------|
| `How do I reset my password?` | 🤖 Bot answers automatically |
| `What is the price of the Pro plan?` | 🤖 Bot answers automatically |
| `How do I enable two-factor authentication?` | 🤖 Bot answers automatically |
| `I was charged twice, this is unacceptable!` | 👤 Escalates to human agent |
| `Can you access my account directly?` | 👤 Escalates to human agent |
| `What is the meaning of life?` | 👤 Escalates to human agent |

When escalation triggers, an **Agent Response Form** appears — type a reply as the human agent and click Send.

---

## 🔁 LangGraph Workflow

```
input_node  →  router_node  →  llm_node   →  output_node  →  END
                           ↘  hitl_node  ↗
```

**Routing Logic:**
- `answerable` + `high/medium` confidence → **llm_node** (auto answer)
- `answerable` + `low` confidence → **hitl_node** (safety escalation)
- `escalate` (any reason) → **hitl_node** (human agent)
- No chunks found → **hitl_node** (immediately)

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [HLD_Document.pdf](HLD_Document.pdf) | High-Level Design — system architecture, components, data flow |
| [LLD_Document.pdf](LLD_Document.pdf) | Low-Level Design — module internals, data structures, error handling |
| [Technical_Documentation.pdf](Technical_Documentation.pdf) | Full technical writeup — design decisions, trade-offs, testing |

---

## 🔮 Future Enhancements

- Multi-document support with metadata filtering
- Async HITL queue with FastAPI + Redis
- Streaming LLM responses in Streamlit
- Feedback loop for agent corrections
- Docker deployment on AWS / Streamlit Cloud

---

## 👤 Author

**Janu Vishwanath**  
Innomatics Research Labs — AI/ML Internship  
📧 januvishwanath56@gmail.com

---

## 📝 License

This project is built for educational purposes as part of the Innomatics Research Labs internship program.