"""
Streamlit Poker LLM Assistant
Usage:
1. pip install -r requirements.txt
2. export OPENAI_API_KEY="sk-..."
3. streamlit run poker_llm_streamlit.py
"""

import os
import streamlit as st
import eval7
import random
import openai
from typing import List

# ---------- Settings ----------
LLM_MODEL = "gpt-4o"

# Load API key
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai.api_key = os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title="Poker LLM Assistant", layout="wide")
st.title("Poker LLM Assistant — Streamlit")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    trials = st.number_input("Monte Carlo trials", value=2000, min_value=100, max_value=20000, step=100)
    st.markdown("---")
    st.markdown("**Instructions**: Enter your hand & table, then press **Get Recommendation**.")

# ---------- Poker utilities ----------
RANKS = '23456789TJQKA'
SUITS = 'CDHS'
DECK = [r + s for r in RANKS for s in SUITS]

def parse_cards(text: str) -> List[str]:
    """Parse & validate card input (e.g., Ah Kd)."""
    text = text.strip().upper()
    if not text:
        return []
    tokens = text.replace(',', ' ').split()
    cards = []
    for t in tokens:
        if len(t) == 2 and t[0] in RANKS and t[1] in SUITS:
            cards.append(t)
    return cards

def monte_carlo_equity_eval7(hole: List[str], board: List[str], n_opps: int = 1, trials: int = 2000) -> float:
    deck = [c for c in DECK if c.upper() not in [x.upper() for x in hole + board]]
    wins, ties = 0, 0
    for _ in range(trials):
        random.shuffle(deck)
        idx = 0
        opps = [[deck[idx + 2*i], deck[idx + 2*i + 1]] for i in range(n_opps)]
        idx += 2 * n_opps
        needed = 5 - len(board)
        new_board = board.copy()
        for i in range(needed):
            new_board.append(deck[idx]); idx += 1
        try:
            our_score = eval7.evaluate([eval7.Card(c) for c in hole + new_board])
            opp_scores = [eval7.evaluate([eval7.Card(c) for c in o + new_board]) for o in opps]
        except Exception:
            continue
        max_opp = max(opp_scores) if opp_scores else -1
        if our_score > max_opp:
            wins += 1
        elif our_score == max_opp:
            ties += 1
    equity = (wins + ties * 0.5) / trials
    return equity

# ---------- Main UI ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Hand & Table")
    hole_input = st.text_input("Hole cards (e.g., Ah Kd)")
    board_input = st.text_input("Board cards (e.g., 7h 9d 2s)")
    pot = st.number_input("Pot size", value=100.0, min_value=0.0)
    to_call = st.number_input("Bet to you (to call)", value=0.0, min_value=0.0)
    opponents = st.number_input("Number of opponents", value=1, min_value=1)

with col2:
    st.subheader("Actions")
    run_btn = st.button("Get Recommendation")
    st.caption("Ensure OPENAI_API_KEY is set in Streamlit Cloud Secrets.")

# ---------- LLM prompt ----------
SYSTEM_PROMPT = (
    "You are a poker assistant. Provide advice based on hand strength and board context.\n"
    "Return JSON with keys: action (fold/check/call/bet/raise), bet_size, equity_pct (0-100), reasoning."
)

if run_btn:
    if not openai.api_key:
        st.error("OPENAI_API_KEY not found. Set it in Streamlit Cloud Secrets.")
    else:
        hole = parse_cards(hole_input)
        board = parse_cards(board_input)
        if len(hole) != 2:
            st.error("Enter exactly 2 valid hole cards (e.g., Ah Kd).")
        elif len(board) > 5:
            st.error("Board can have at most 5 cards (e.g., 7h 9d 2s).")
        else:
            with st.spinner('Computing equity...'):
                equity = monte_carlo_equity_eval7(hole, board, n_opps=int(opponents), trials=int(trials))

            # Build prompt
            prompt = (
                f"Hole: {' '.join(hole)}\n"
                f"Board: {' '.join(board)}\n"
                f"Pot: {pot}\nTo call: {to_call}\nOpponents: {opponents}\n"
                f"Estimated equity: {equity*100:.2f}%\n\n"
                "Based on this, recommend the best poker action."
            )

            st.markdown("### LLM Recommendation")
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

            st.metric("Estimated win probability", f"{equity*100:.2f}%")

st.caption("LLM-only poker assistant. No RAG or playbook embeddings.")
