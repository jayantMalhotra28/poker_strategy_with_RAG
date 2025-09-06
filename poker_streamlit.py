"""
Streamlit Poker LLM Assistant (No RAG)
File: poker_llm_streamlit.py

Usage locally:
1. pip install -r requirements.txt
2. export OPENAI_API_KEY="sk-..."
3. streamlit run poker_llm_streamlit.py

On Streamlit Cloud:
- Set OPENAI_API_KEY in app Secrets
"""

import os
import streamlit as st
import eval7
import random
from typing import List
import openai

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Poker LLM Assistant", layout="wide")
st.title("Poker LLM Assistant — Streamlit")

# ---------- Helpers ----------
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = [r + s for r in RANKS for s in SUITS]

def parse_cards(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    tokens = text.replace(',', ' ').split()
    cards = []
    for t in tokens:
        if len(t) == 2:
            cards.append(t.upper())
    return cards

def monte_carlo_equity_eval7(hole: List[str], board: List[str], n_opps: int = 1, trials: int = 2000) -> float:
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

# ---------- Sidebar inputs ----------
with st.sidebar:
    st.header("Settings")
    trials = st.number_input("Monte Carlo trials", value=2000, min_value=100, max_value=20000, step=100)
    st.markdown("---")
    st.markdown("**Instructions**: Enter cards and table info, then press **Get Recommendation**.")

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
    st.caption("Ensure OPENAI_API_KEY is set in environment or Streamlit Secrets.")

# ---------- LLM call ----------
SYSTEM_PROMPT = """
You are a poker assistant. 
For each hand, return JSON with keys: action (fold/check/call/bet/raise), bet_size, equity_pct (0-100), reasoning.
Prefer conservative play when uncertain.
"""

# Get key from environment or Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")

if run_btn:
    if not openai.api_key:
        st.error("OPENAI_API_KEY not found. Set it in environment or Streamlit Secrets.")
    else:
        hole = parse_cards(hole_input)
        board = parse_cards(board_input)
        if len(hole) != 2:
            st.error("Enter exactly two hole cards (e.g., Ah Kd).")
        else:
            with st.spinner("Computing equity..."):
                equity = monte_carlo_equity_eval7(hole, board, n_opps=int(opponents), trials=int(trials))
            
            user_prompt = f"""
Game state:
Hole cards: {' '.join(hole)}
Board cards: {' '.join(board)}
Pot: {pot}
To call: {to_call}
Opponents: {opponents}
Estimated equity: {equity:.2%}

Return your recommendation as JSON.
"""
            with st.spinner("Querying LLM..."):
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=400,
                        temperature=0.2
                    )
                    text = resp['choices'][0]['message']['content']
                    st.markdown("### LLM Recommendation")
                    st.code(text, language="json")
                    st.metric("Estimated win probability", f"{equity*100:.2f}%")
                except Exception as e:
                    st.error(f"LLM call failed: {e}")

st.caption("LLM-only template. No playbook embeddings or RAG involved.")
