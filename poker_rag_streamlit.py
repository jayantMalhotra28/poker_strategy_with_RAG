"""
Streamlit Poker RAG Assistant
File: poker_rag_streamlit.py

Requirements:
- Reads from requirements.txt when deployed.

Usage locally:
1. pip install -r requirements.txt
2. export OPENAI_API_KEY="sk-..."
3. streamlit run poker_rag_streamlit.py

On Streamlit Cloud:
- Add requirements.txt with dependencies
- Set OPENAI_API_KEY in app Secrets
"""

import os
import streamlit as st
from docx import Document
import numpy as np
import faiss
import random
import eval7
from typing import List
from openai import OpenAI

# ---------- Models ----------
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

# ---------- API Key ----------
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Poker RAG Assistant", layout="wide")
st.title("Poker RAG Assistant — Streamlit")

with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Retriever top-k", value=3, min_value=1, max_value=10)
    trials = st.number_input("Monte Carlo trials", value=2000, min_value=100, max_value=20000, step=100)
    st.markdown("---")
    st.markdown("**Instructions**: Upload a .docx playbook, enter cards, then press **Get Recommendation**.")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def read_docx_bytes(bytes_data) -> str:
    from io import BytesIO
    bio = BytesIO(bytes_data)
    doc = Document(bio)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

def chunk_text_simple(text: str, max_chars: int = 800) -> List[str]:
    paras = [p for p in text.split('\n\n') if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + '\n\n' + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks

@st.cache_resource
def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def embed_texts_openai(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    embs = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return embs

# ---------- Poker Utilities ----------
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = [r + s for r in RANKS for s in SUITS]

def parse_cards(text: str) -> List[str]:
    tokens = text.replace(',', ' ').split()
    return [t.upper() for t in tokens if len(t) == 2]

def monte_carlo_equity_eval7(hole: List[str], board: List[str], n_opps: int = 1, trials: int = 2000) -> float:
    deck = [c for c in DECK if c.upper() not in [*hole, *board]]
    wins = ties = 0
    for _ in range(trials):
        random.shuffle(deck)
        idx = 0
        opps = [[deck[idx + 2*i], deck[idx + 2*i +1]] for i in range(n_opps)]
        idx += 2*n_opps
        new_board = board.copy() + deck[idx:idx+(5-len(board))]
        idx += 5 - len(board)
        our_score = eval7.evaluate([eval7.Card(c) for c in hole + new_board])
        opp_scores = [eval7.evaluate([eval7.Card(c) for c in o + new_board]) for o in opps]
        max_opp = max(opp_scores) if opp_scores else -1
        if our_score > max_opp: wins += 1
        elif our_score == max_opp: ties += 1
    return (wins + ties*0.5)/trials

# ---------- RAG ----------
SYSTEM_PROMPT = (
    "You are a poker assistant. Use the supplied playbook excerpts to inform recommendations.\n"
    "Return JSON with keys: action (fold/check/call/bet/raise), bet_size, equity_pct (0-100), reasoning, used_passages.\n"
    "Highlight contradictions and prefer conservative play when uncertain."
)

def build_rag_prompt(retrieved_passages: List[str], game_state: dict, equity: float) -> str:
    passages_block = "\n\n".join([f"- {p}" for p in retrieved_passages])
    user_block = (
        f"Retrieved passages:\n{passages_block}\n\n"
        f"Game state:\nHole: {game_state['hole']}\nBoard: {game_state['board']}\n"
        f"Pot: {game_state['pot']}\nTo_call: {game_state['to_call']}\nOpponents: {game_state['opponents']}\n"
        f"Equity (sim): {equity:.2%}\nReturn JSON only."
    )
    return SYSTEM_PROMPT + "\n" + user_block

# ---------- Main UI ----------
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Hand & Table")
    hole_input = st.text_input("Hole cards (e.g. Ah Kd)")
    board_input = st.text_input("Board cards (e.g. 7h 9d 2s)")
    pot = st.number_input("Pot size", value=100.0, min_value=0.0)
    to_call = st.number_input("Bet to you (to call)", value=0.0, min_value=0.0)
    opponents = st.number_input("Number of opponents", value=1, min_value=1)

with col2:
    st.subheader("Actions")
    run_btn = st.button("Get Recommendation")
    st.caption("Ensure OPENAI_API_KEY is set in Streamlit Cloud Secrets.")

# ---------- Playbook Handling ----------
index = None
chunks = []
embeddings = None

uploaded_file = st.file_uploader("Upload Word playbook (.docx)", type=["docx"])
if uploaded_file:
    raw_text = read_docx_bytes(uploaded_file.getvalue())
    chunks = chunk_text_simple(raw_text, max_chars=800)
    st.success(f"Playbook loaded — {len(chunks)} chunks created.")
    try:
        with st.spinner('Embedding playbook...'):
            embeddings = embed_texts_openai(chunks)
            index = build_faiss_index(embeddings)
        st.info('FAISS index ready.')
    except Exception as e:
        st.error(f"Embedding/index build failed: {e}")
        index = None

def retrieve_top_k(query: str, k: int = 3):
    if index is None or embeddings is None: return []
    q_emb = embed_texts_openai([query])[0]
    D, I = index.search(np.array([q_emb]), k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

# ---------- Run Recommendation ----------
if run_btn:
    if not client.api_key:
        st.error("OPENAI_API_KEY not found.")
    elif not uploaded_file:
        st.error("Please upload a .docx playbook first.")
    else:
        hole = parse_cards(hole_input)
        board = parse_cards(board_input)
        if len(hole) != 2:
            st.error("Enter exactly two hole cards (e.g., Ah Kd).")
        else:
            with st.spinner('Computing equity & retrieving passages...'):
                equity = monte_carlo_equity_eval7(hole, board, n_opps=int(opponents), trials=int(trials))
                query_text = f"hole {' '.join(hole)} board {' '.join(board)} pot {pot} to_call {to_call} opponents {opponents}"
                retrieved = retrieve_top_k(query_text, k=int(top_k))

            prompt = build_rag_prompt(retrieved, {
                'hole': ' '.join(hole),
                'board': ' '.join(board),
                'pot': pot,
                'to_call': to_call,
                'opponents': opponents
            }, equity)

            st.markdown("### Retrieved passages")
            for i, p in enumerate(retrieved, start=1):
                st.write(f"{i}. {p[:400]}{'...' if len(p)>400 else ''}")

            st.markdown("### LLM Recommendation")
            try:
                resp = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.2,
                )
                text = resp.choices[0].message.content
                st.code(text, language='json')
            except Exception as e:
                st.error(f"LLM call failed: {e}")

            st.metric("Estimated win probability", f"{equity*100:.2f}%")

st.caption("Starter template. Tune settings before using for real-money poker.")
