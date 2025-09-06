"""
Streamlit Poker RAG Assistant
File: poker_rag_streamlit.py

Requirements:
pip install streamlit python-docx openai faiss-cpu numpy eval7

Usage:
1. Set environment variable OPENAI_API_KEY with your OpenAI API key.
   export OPENAI_API_KEY="sk-..."
2. Run:
   streamlit run poker_rag_streamlit.py

What it does:
- Upload a Word (.docx) playbook.
- Chunks and embeds the playbook into a FAISS vector index using OpenAI embeddings.
- Accepts hole cards, board cards, pot, to_call, # opponents.
- Estimates equity using eval7 Monte Carlo simulator.
- Retrieves top-k relevant passages from the playbook and sends a RAG prompt to the LLM to recommend action + bet sizing.

Notes:
- This is a starter app. Tune chunk sizes, embedding model, and LLM model as needed.
- The app avoids storing API keys; ensure OPENAI_API_KEY is set in your environment.
"""

import os
import streamlit as st
from docx import Document
import numpy as np
import faiss
import openai
import eval7
import random
import time
from typing import List

# ---------- Helpers: text processing & embeddings ----------
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"  # change if you prefer another model

openai.api_key = os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title="Poker RAG Assistant", layout="wide")
st.title("Poker RAG Assistant — Streamlit")

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Retriever top-k", value=3, min_value=1, max_value=10)
    trials = st.number_input("Monte Carlo trials", value=2000, min_value=100, max_value=20000, step=100)
    use_faiss_cpu = st.checkbox("Use FAISS (recommended)", value=True)
    st.markdown("---")
    st.markdown("**Instructions**: Upload a .docx playbook, enter cards, then press **Get Recommendation**.")

# ---------- Upload playbook ----------
uploaded_file = st.file_uploader("Upload Word playbook (.docx)", type=["docx"]) 

@st.cache_data(show_spinner=False)
def read_docx_bytes(bytes_data) -> str:
    from io import BytesIO
    bio = BytesIO(bytes_data)
    doc = Document(bio)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

@st.cache_resource
def build_faiss_index(chunks: List[str], embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def chunk_text_simple(text: str, max_chars: int = 800) -> List[str]:
    # naive chunk by paragraphs, merge until approx max_chars
    paras = [p for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = ""
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

def embed_texts_openai(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    # OpenAI embeddings API supports batching
    resp = openai.Embedding.create(model=model, input=texts)
    embs = np.array([r['embedding'] for r in resp['data']], dtype=np.float32)
    return embs

# ---------- Poker utilities (eval7-based equity) ----------

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = [r + s for r in RANKS for s in SUITS]

def parse_cards(text: str) -> List[str]:
    # accepts space-separated like 'Ah Kd' or 'AhKd' grouped
    text = text.strip()
    if not text:
        return []
    tokens = text.replace(',', ' ').split()
    cards = []
    for t in tokens:
        for sep in ['',' ']:
            tt = t.replace(sep, '')
        c = t.strip()
        if len(c) == 2:
            cards.append(c.upper())
    return cards

def monte_carlo_equity_eval7(hole: List[str], board: List[str], n_opps: int = 1, trials: int = 2000) -> float:
    # uses eval7 to simulate
    deck = [c for c in DECK if c.upper() not in [x.upper() for x in hole + board]]
    wins = 0
    ties = 0
    for _ in range(trials):
        random.shuffle(deck)
        idx = 0
        opps = []
        for _ in range(n_opps):
            opps.append([deck[idx], deck[idx+1]])
            idx += 2
        needed = 5 - len(board)
        new_board = board.copy()
        for i in range(needed):
            new_board.append(deck[idx]); idx += 1
        our_hand = [eval7.Card(card) for card in hole + new_board]
        our_score = eval7.evaluate(our_hand)
        opp_scores = []
        for o in opps:
            o_hand = [eval7.Card(card) for card in o + new_board]
            opp_scores.append(eval7.evaluate(o_hand))
        max_opp = max(opp_scores) if opp_scores else -1
        if our_score > max_opp:
            wins += 1
        elif our_score == max_opp:
            ties += 1
    equity = (wins + ties * 0.5) / trials
    return equity

# ---------- RAG prompt builder ----------

SYSTEM_PROMPT = (
    "You are a poker assistant. Use the supplied playbook excerpts to inform recommendations.\n"
    "For each query return JSON with keys: action (fold/check/call/bet/raise), bet_size (absolute or fraction), equity_pct (0-100), reasoning (1-3 lines), used_passages (list of excerpts).\n"
    "If playbook contradicts itself, highlight that. Prefer conservative play when uncertain.\n"
)

def build_rag_prompt(retrieved_passages: List[str], game_state: dict, equity: float) -> str:
    passages_block = "\n\n".join([f"- {p}" for p in retrieved_passages])
    user_block = (
        f"Retrieved passages:\n{passages_block}\n\n"
        f"Game state:\nHole: {game_state['hole']}\nBoard: {game_state['board']}\nPot: {game_state['pot']}\n"
        f"To_call: {game_state['to_call']}\nOpponents: {game_state['opponents']}\nEquity (sim): {equity:.2%}\n"
        "Return the result as a JSON object only."
    )
    return SYSTEM_PROMPT + "\n" + user_block

# ---------- Main UI ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Hand & Table")
    hole_input = st.text_input("Hole cards (e.g. Ah Kd)")
    board_input = st.text_input("Board cards (space-separated, e.g. 7h 9d 2s)")
    pot = st.number_input("Pot size", value=100.0, min_value=0.0)
    to_call = st.number_input("Bet to you (to call)", value=0.0, min_value=0.0)
    opponents = st.number_input("Number of opponents", value=1, min_value=1)

with col2:
    st.subheader("Actions")
    run_btn = st.button("Get Recommendation")
    st.markdown("---")
    st.caption("Make sure OPENAI_API_KEY is set in your environment.")

# Store or compute index when playbook uploaded
index = None
chunks = []
embeddings = None

if uploaded_file is not None:
    raw_text = read_docx_bytes(uploaded_file.getvalue())
    chunks = chunk_text_simple(raw_text, max_chars=800)
    st.success(f"Playbook loaded — {len(chunks)} chunks created.")

    try:
        with st.spinner('Embedding playbook...'):
            embeddings = embed_texts_openai(chunks)
            index = build_faiss_index(chunks, embeddings)
        st.info('FAISS index built and ready.')
    except Exception as e:
        st.error(f"Embedding/index build failed: {e}")
        index = None

# Retrieval
def retrieve_top_k(query: str, k: int = 3):
    if index is None or embeddings is None:
        return []
    q_emb = embed_texts_openai([query])[0]
    D, I = index.search(np.array([q_emb]), k)
    retrieved = [chunks[i] for i in I[0] if i < len(chunks)]
    return retrieved

# Handle run
if run_btn:
    if not openai.api_key:
        st.error("OPENAI_API_KEY environment variable is not set. Set it and refresh.")
    elif not uploaded_file:
        st.error("Please upload a .docx playbook first.")
    else:
        hole = parse_cards(hole_input)
        board = parse_cards(board_input)
        if len(hole) != 2:
            st.error("Please enter exactly two hole cards (e.g., Ah Kd).")
        else:
            with st.spinner('Computing equity and retrieving passages...'):
                try:
                    equity = monte_carlo_equity_eval7(hole, board, n_opps=int(opponents), trials=int(trials))
                except Exception as e:
                    st.error(f"Equity calc failed: {e}")
                    equity = 0.0
                query_text = f"hole {' '.join(hole)} board {' '.join(board)} pot {pot} to_call {to_call} opponents {opponents}"
                retrieved = retrieve_top_k(query_text, k=int(top_k))

            prompt = build_rag_prompt(retrieved, {
                'hole': ' '.join(hole),
                'board': ' '.join(board),
                'pot': pot,
                'to_call': to_call,
                'opponents': opponents
            }, equity)

            st.markdown("### Retrieved passages used")
            for i, p in enumerate(retrieved, start=1):
                st.write(f"{i}. {p[:500]}{'...' if len(p)>500 else ''}")

            st.markdown("### LLM Recommendation")
            with st.spinner('Calling LLM...'):
                try:
                    resp = openai.ChatCompletion.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=400,
                        temperature=0.2,
                    )
                    text = resp['choices'][0]['message']['content']
                    st.code(text, language='json')
                except Exception as e:
                    st.error(f"LLM call failed: {e}")

            st.markdown("### Raw equity (simulator)")
            st.metric("Estimated win probability", f"{equity*100:.2f}%")

st.markdown("---")
st.caption("This app is a starter template. Tweak models, chunking, and evaluation settings before using for real-money decisions.")

# EOF
