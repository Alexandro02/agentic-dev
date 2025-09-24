# Mini-{insertname} (RAG + Self-Critique + Minimal UI)

An agent that "thinks like me" with:
- **RAG** (TF-IDF) over your `knowledge/` folder
- **Self-critique** based on a **rubric** (retries once if it fails)
- **Light memory** (`memory.json`)
- **Minimal UI** using **Gradio**

> Designed to quickly prototype your own voice/style (copy, decisions, flow design).

---

## 🚀 Requirements

- **Python 3.10+**
- OpenAI **API Key**
- **git**

---

## 📦 Clone the repository

```bash
git clone https://github.com/Alexandro02/agentic-dev.git
cd agentic-dev
```

---

## 🧰 Installation

> **Do not push `.venv` to GitHub.** Use virtual environments (PEP 668).

```bash
python -m venv .venv
# Activate:

# macOS/Linux:
source .venv/bin/activate

# Windows (PowerShell):
# .venv\Scripts\activate

pip install -r requirements.txt
```

If `requirements.txt` doesn’t exist yet:

```bash
pip install openai pyyaml scikit-learn gradio python-dotenv
pip freeze > requirements.txt
```

---

## 🔑 Configure environment variables

Create a `.env` file at the project root:

```
OPENAI_API_KEY=sk-your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## 🧠 Configure agent “personality”

Edit `alex_agent.yaml` (example included). Adjust:
- **persona.north_star**, **tone**, **values**
- **rubric** (the 5 quality checks)
- **response_recipe** (how outputs should be structured)

---

## 📚 Add knowledge (RAG)

Place `.md` or `.txt` files into `knowledge/`:

```
knowledge/
├─ 001-{project}-buildinpublic.md
├─ 002-backend-journey.md
├─ 003-ai-tools-thread.md
├─ 004-onbrand-style.md
└─ 005-offbrand-corporate.txt   # anti-example (start with "OFFBRAND:")
```

---

## ▶️ Run the app

```bash
python app.py
```

Go to:  
**http://localhost:7860**

You’ll see:
- **Answer** (final output)
- **Context used** (RAG docs)
- **Self-critique** (score & fixes)

---

## 📁 Project structure

```
alex-agent/
├─ app.py                 # agent + Gradio UI
├─ alex_agent.yaml        # “brain” (voice, values, rubric, recipe)
├─ .env                   # secrets (DO NOT commit)
├─ knowledge/             # personal corpus (.md/.txt)
├─ memory.json            # lightweight memory (auto-generated)
└─ requirements.txt
```

---

## 🔒 Recommended .gitignore {add more if needed}

```
.venv/
__pycache__/
*.pyc
.env
memory.json
```

---

## 🧯 Troubleshooting

**PEP 668 / “externally-managed-environment”**  
Don’t use system Python. Instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**App doesn’t start**  
- Check `OPENAI_API_KEY` in `.env`  
- Verify port 7860 is free  

**Responses too “nice” but not actionable**  
- Strengthen **rubric** (require example + CTA)  
- Adjust **response_recipe** (bullets, checklist, etc.)

---

## 🗺️ Roadmap

- [ ] **Fast/Quality mode** (single call vs self-critique call)
- [ ] **Embeddings + FAISS** (semantic search)
- [ ] **Response caching** (SQLite/Redis)
- [ ] **Incremental corpus refresh**`
- [ ] **Draft export** (save outputs with timestamp)

---
