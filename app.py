import os, json, glob, re, time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
import yaml

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gradio as gr

# ---------- Config ----------
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()

ROOT = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(ROOT, "knowledge")
YAML_PATH = os.path.join(ROOT, "alex_agent.yaml")
MEM_PATH = os.path.join(ROOT, "memory.json")

# ---------- Utils ----------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_memory(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"preferences": {}, "history": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_corpus(folder: str) -> List[Tuple[str, str]]:
    files = []
    for ext in ("*.txt","*.md"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    corpus = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                corpus.append((os.path.basename(fp), f.read()))
        except Exception as e:
            print(f"Skipping {fp}: {e}")
    return corpus

# ---------- Simple RAG (TF-IDF) ----------
@dataclass
class RagIndex:
    vectorizer: Any
    matrix: Any
    docs: List[Tuple[str, str]]

def build_rag_index(corpus: List[Tuple[str,str]]) -> RagIndex:
    texts = [c[1] for c in corpus]
    vect = TfidfVectorizer(stop_words="english", max_df=0.8)
    mat = vect.fit_transform(texts) if texts else None
    return RagIndex(vect, mat, corpus)

def retrieve(index: RagIndex, query: str, k: int = 3) -> List[Tuple[str,str,float]]:
    if index.matrix is None or index.matrix.shape[0] == 0:
        return []
    qv = index.vectorizer.transform([query])
    sims = cosine_similarity(qv, index.matrix).flatten()
    top = sims.argsort()[::-1][:k]
    results = []
    for i in top:
        fname, text = index.docs[i]
        score = float(sims[i])
        # Penalizar anti-ejemplos marcados
        if text.strip().upper().startswith("OFFBRAND:"):
            score *= 0.5
        results.append((fname, text, score))
    return results

# ---------- Prompt Fabric ----------
def system_prompt(cfg: Dict[str, Any]) -> str:
    persona = cfg.get("persona", {})
    do_not = cfg.get("do_not", [])
    rubric = cfg.get("rubric", [])
    recipe = cfg.get("response_recipe", [])

    sys = f"""You are "{persona.get('name','Mini-Alex')}", a pragmatic founder/dev.
North star: {persona.get('north_star','Clarity-first')}
Stack: {', '.join(persona.get('stack',[]))}
Tone: {cfg.get('voice_style',{}).get('tone','direct-minimal')}
Lexicon: {', '.join(cfg.get('voice_style',{}).get('lexicon',[]))}

Never do: {', '.join(do_not)}.

Decision Rubric (must pass at least 4/5):
- {rubric[0] if len(rubric)>0 else 'Clarity'}
- {rubric[1] if len(rubric)>1 else 'User impact'}
- {rubric[2] if len(rubric)>2 else 'Flow over features'}
- {rubric[3] if len(rubric)>3 else 'Example included'}
- {rubric[4] if len(rubric)>4 else 'One CTA'}

Response Recipe:
- {recipe[0] if len(recipe)>0 else 'Short bullets'}
- {recipe[1] if len(recipe)>1 else 'Add example'}
- {recipe[2] if len(recipe)>2 else 'End with CTA'}
- {recipe[3] if len(recipe)>3 else 'Ask 1 question if needed'}
"""
    return sys

def build_user_prompt(query: str, ctx_snippets: List[Tuple[str,str,float]], mode: str) -> str:
    ctx = ""
    for fname, text, score in ctx_snippets:
        short = re.sub(r"\s+", " ", text.strip())[:600]
        ctx += f"\n[CTX:{fname} | score={score:.3f}]\n{short}\n"
    return f"""MODE: {mode}
USER QUERY:
{query}

RELEVANT CONTEXT (from Alex's corpus):
{ctx if ctx_snippets else '(no context found)'}

Write the answer in Alex's voice. Follow the Decision Rubric and Response Recipe strictly.
"""

def llm_chat(messages: List[Dict[str,str]], temperature: float = 0.5) -> str:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=messages
    )
    return resp.choices[0].message.content

def self_critique(cfg: Dict[str,Any], draft: str) -> str:
    rubric = "\n".join([f"- {r}" for r in cfg.get("rubric",[])])
    critique_prompt = [
        {"role":"system","content":"You are a strict reviewer scoring against a rubric (0-5)."},
        {"role":"user","content":f"""Rubric:
{rubric or '- Creates immediate clarity\n- Solves friction today\n- Flow > feature\n- Includes example\n- Ends with one CTA'}

Draft:
{draft}

Rate 0-5. If score <4, list 2-3 concrete fixes. Respond as:
SCORE: X
NOTES: ...
FIXES (if any): ...
"""}
    ]
    return llm_chat(critique_prompt, temperature=0.0)

def maybe_revise(cfg: Dict[str,Any], original_messages: List[Dict[str,str]], draft: str, critique: str) -> str:
    m = re.search(r"SCORE:\s*([0-5])", critique)
    score = int(m.group(1)) if m else 3
    if score >= 4:
        return draft
    fixes = ""
    fx = re.search(r"FIXES.*?:\s*(.*)", critique, re.S)
    if fx:
        fixes = fx.group(1).strip()[:800]
    revise_messages = original_messages + [
        {"role":"user","content":f"""Revise the previous draft.
Apply ONLY these fixes:
{fixes or 'Improve clarity, add one example, end with one CTA. Keep it concise.'}
Return final answer."""}
    ]
    return llm_chat(revise_messages, temperature=0.4)

# ---------- Agent Core ----------
class AlexAgent:
    def __init__(self):
        self.cfg = load_yaml(YAML_PATH)
        self.mem = load_memory(MEM_PATH)
        self.corpus = read_corpus(KNOWLEDGE_DIR)
        self.index = build_rag_index(self.corpus)

    def refresh(self):
        self.cfg = load_yaml(YAML_PATH)
        self.mem = load_memory(MEM_PATH)
        self.corpus = read_corpus(KNOWLEDGE_DIR)
        self.index = build_rag_index(self.corpus)

    def run(self, query: str, mode: str = "copy", use_ctx: bool = True) -> Tuple[str, List[Tuple[str,str,float]], str]:
        ctx_snippets = retrieve(self.index, query, k=3) if use_ctx else []
        sys = system_prompt(self.cfg)
        userp = build_user_prompt(query, ctx_snippets, mode)
        messages = [
            {"role":"system","content":sys},
            {"role":"user","content":userp}
        ]
        draft = llm_chat(messages, temperature=0.5)
        critique = self_critique(self.cfg, draft)
        final = maybe_revise(self.cfg, messages, draft, critique)
        # memoria liviana (últimas 10 queries)
        self.mem["history"] = (self.mem.get("history", []) + [{"t": int(time.time()), "q": query}])[-10:]
        save_memory(MEM_PATH, self.mem)
        return final, ctx_snippets, critique

agent = AlexAgent()

# ---------- Gradio UI ----------
def infer(query, mode, use_ctx):
    try:
        out, ctx, critique = agent.run(query, mode, use_ctx)
        ctx_view = "\n\n".join([f"• {f} (score {s:.3f})\n{t[:240]}..." for f,t,s in ctx]) or "No context"
        return out, ctx_view, critique
    except Exception as e:
        return f"Error: {e}", "", ""

with gr.Blocks(title="Mini-Alex") as demo:
    gr.Markdown("# Mini-Alex · think like Alex\nMinimal agent with RAG + rubric self-critique.")
    with gr.Row():
        query = gr.Textbox(label="Prompt / Task", lines=5, placeholder="e.g. Draft a 5-bullet post about Corvia's dashboard UX...")
    with gr.Row():
        mode = gr.Dropdown(choices=["copy","design","decision"], value="copy", label="Mode")
        use_ctx = gr.Checkbox(value=True, label="Use knowledge/ context (RAG)")
    btn = gr.Button("Run")
    out = gr.Markdown(label="Answer")
    ctx = gr.Textbox(label="Context used (top matches)", lines=8)
    crt = gr.Textbox(label="Self-critique", lines=6)

    btn.click(infer, [query, mode, use_ctx], [out, ctx, crt])

if __name__ == "__main__":
    # gradio app
    demo.launch(server_name="0.0.0.0", server_port=7860)
