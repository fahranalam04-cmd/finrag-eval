"""Minimal UI over the finrag API: answer, citations (checked against the context) and passages.

uv run uvicorn finrag.api:create_default_app --factory --port 8000
uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.environ.get("FINRAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="finrag-eval", layout="wide")
st.title("finrag-eval")
st.caption("Cited answers over FiQA-2018 forum passages. Answers are generated only from retrieved text.")

try:
    health = httpx.get(f"{API}/health", timeout=5).json()
    st.sidebar.write(f"API: `{API}`")
    st.sidebar.write(f"Reranker: `{health['reranker']}`")
    st.sidebar.write(f"Generator: `{health['generator']}`")
    modes = health["modes"]
except httpx.HTTPError as e:
    st.error(f"API not reachable at {API}: {e}")
    st.stop()

query = st.text_input("Question", "Should I pay off my mortgage early or invest the money?")
c1, c2, c3 = st.columns([2, 2, 1])
mode = c1.selectbox("Retrieval", modes[::-1])
k = c2.slider("Passages in context", 1, 10, 5)
go = c3.button("Answer", type="primary")

if go and query.strip():
    with st.spinner("Retrieving and generating..."):
        r = httpx.post(f"{API}/answer", json={"query": query, "k_context": k, "mode": mode}, timeout=180)
    if r.status_code != 200:
        st.error(f"{r.status_code}: {r.text}")
        st.stop()
    data = r.json()
    st.subheader("Answer")
    if data["abstained"]:
        st.warning("The model abstained: not enough information in the retrieved passages.")
    st.markdown(data["answer"])
    t = data["timings_ms"]
    st.caption(" | ".join(f"{name}: {ms:.0f} ms" for name, ms in t.items()))

    st.subheader("Citations")
    if not data["citations"]:
        st.write("No citations.")
    for c in data["citations"]:
        status = "in retrieved context" if c["in_context"] else "NOT in context (invalid citation)"
        st.write(f"- `doc{c['doc_id']}`: {status}")

    st.subheader("Retrieved passages")
    cited = {c["doc_id"] for c in data["citations"]}
    for p in data["passages"]:
        tag = " (cited)" if p["doc_id"] in cited else ""
        with st.expander(f"#{p['rank']}  doc{p['doc_id']}  score={p['score']:.3f}{tag}", expanded=bool(tag)):
            st.write(p["text"])
