import os
import json
import requests
import streamlit as st

# Backend URL (FastAPI)
BACKEND_URL = os.environ.get("MIRAG_BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Multi-Index Router RAG", page_icon="🔎", layout="wide")

with st.sidebar:
    st.header("Settings")
    backend = "http://127.0.0.1:8000"
    topk = st.slider("Top-K Contexts", 2, 12, 6, 1)
    st.markdown("---")
q = st.text_input("Ask a question", placeholder="e.g., Total revenue by month this year?")
go = st.button("Ask", type="primary")

def show_sql_block(sql_block):
    if not sql_block:
        return
    st.subheader("SQL")
    st.code(sql_block.get("sql", ""), language="sql")
    if sql_block.get("error"):
        st.error(sql_block["error"])
    else:
        cols = sql_block.get("columns", [])
        rows = sql_block.get("rows", [])
        if cols and rows:
            import pandas as pd
            df = pd.DataFrame(rows, columns=cols)
            st.dataframe(df.head(50), use_container_width=True)

def show_citations(citations):
    if not citations:
        return
    st.subheader("Citations")
    for i, c in enumerate(citations, start=1):
        st.write(f"[{i}] {c}")

if go and q.strip():
    try:
        resp = requests.post(f"{backend}/ask", json={"question": q, "topk": topk}, timeout=120)
        if resp.status_code != 200:
            st.error(f"Backend error: {resp.status_code} {resp.text}")
        else:
            data = resp.json()
            col1, col2 = st.columns([1,2], gap="large")
            with col1:
                st.metric("Route", data.get("route", "?"))
                st.caption(data.get("reason", ""))
                show_sql_block(data.get("sql"))
                show_citations(data.get("citations", []))
            with col2:
                st.subheader("Answer")
                st.markdown(data.get("answer", ""))
    except Exception as e:
        st.error(f"Request failed: {e}")
else:
    st.info("Enter a question and click **Ask**.")

st.markdown("---")
st.caption("Powered by Groq LLM + FAISS + DuckDB. This is a demo client for the Multi-Index Router service.")
