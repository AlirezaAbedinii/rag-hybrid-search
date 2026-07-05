"""Streamlit UI for the RAG service.

Talks to the FastAPI service over HTTP (``RAG_API_URL``, default
``http://localhost:8000``). Features: ask box, answer with clickable ``[n]``
citations that jump to their source chunk, ranked retrieved chunks, the
confidence signal, a dense/hybrid toggle, and a latency/cost panel backed by
``/v1/stats``.

Run locally:  streamlit run ui/app.py
In compose:   the ``ui`` service (http://localhost:8501)
"""
from __future__ import annotations

import os
import re

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="RAG Hybrid Search", page_icon="📚", layout="wide")


def _linkify_citations(answer: str) -> str:
    """Turn [n] markers into anchor links that jump to the source chunk."""
    return re.sub(r"\[(\d+)\]", r'<a href="#src-\1">[\1]</a>', answer)


def _ask(question: str, mode: str, top_k: int) -> tuple[dict | None, str | None]:
    """POST /v1/ask; return (body, error_message)."""
    try:
        resp = requests.post(
            f"{API_URL}/v1/ask",
            json={"question": question, "mode": mode, "top_k": top_k},
            timeout=120,
        )
    except requests.RequestException as exc:
        return None, f"Could not reach the API at {API_URL}: {exc}"
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return None, f"API returned {resp.status_code}: {detail}"
    return resp.json(), None


# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("📚 RAG Hybrid Search")
    st.caption(f"API: `{API_URL}`")

    mode = st.radio(
        "Retrieval mode",
        ["dense", "hybrid"],
        help="Dense = embedding similarity. Hybrid (dense + BM25 + rerank) "
        "ships with the V1 retrieval work and currently returns 501.",
    )
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 20, 5)

    st.divider()
    if st.button("Refresh service stats"):
        st.session_state.pop("stats", None)
    if "stats" not in st.session_state:
        try:
            st.session_state["stats"] = requests.get(f"{API_URL}/v1/stats", timeout=10).json()
        except requests.RequestException:
            st.session_state["stats"] = None
    stats = st.session_state["stats"]
    if stats and stats.get("requests"):
        st.subheader("Service totals")
        st.metric("Requests", stats["requests"])
        st.metric("Refusal rate", f"{stats['refusal_rate']:.0%}")
        st.metric("Total cost", f"${stats['total_cost_usd']:.4f}")
        total = stats["latency_ms"].get("total_ms", {})
        if total:
            st.metric("P95 latency", f"{total.get('p95', 0):.0f} ms")
    else:
        st.caption("No requests logged yet.")

# ------------------------------------------------------------------ main --
st.header("Ask the docs")
question = st.text_input(
    "Question",
    placeholder="e.g. What does FERRY-429 mean?",
    label_visibility="collapsed",
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving and generating..."):
        body, error = _ask(question.strip(), mode, top_k)
    st.session_state["last"] = (body, error)

body, error = st.session_state.get("last", (None, None))

if error:
    st.error(error)

if body:
    # --- Answer + confidence ------------------------------------------------
    answer_col, meta_col = st.columns([2.2, 1])

    with answer_col:
        if body["refused"]:
            st.warning(body["answer"], icon="🤷")
        else:
            st.markdown(_linkify_citations(body["answer"]), unsafe_allow_html=True)

    with meta_col:
        st.metric("Retrieval confidence", f"{body['confidence']:.2f}")
        st.progress(min(max(body["confidence"], 0.0), 1.0))
        st.caption(
            "Composite confidence (citation coverage + completeness) arrives "
            "with the V1 verification work."
        )

    # --- Latency / cost panel ------------------------------------------------
    st.subheader("Latency & cost")
    timings = dict(body.get("timings_ms", {}))
    total_ms = timings.pop("total_ms", 0.0)
    cols = st.columns(4)
    cols[0].metric("Total latency", f"{total_ms:.0f} ms")
    cols[1].metric("Request cost", f"${body.get('cost_usd', 0):.6f}")
    cols[2].metric("Prompt tokens", body["usage"]["prompt_tokens"])
    cols[3].metric("Completion tokens", body["usage"]["completion_tokens"])
    if timings:
        st.dataframe(
            {"stage": list(timings.keys()), "ms": [round(v, 1) for v in timings.values()]},
            hide_index=True,
            use_container_width=True,
        )

    # --- Citations -----------------------------------------------------------
    if body["citations"]:
        st.subheader("Citations")
        contexts = body.get("contexts", [])
        for cit in body["citations"]:
            st.markdown(f'<div id="src-{cit["index"]}"></div>', unsafe_allow_html=True)
            label = f"[{cit['index']}] {cit['source_file'] or 'unresolved'}"
            if cit.get("section_heading"):
                label += f" § {cit['section_heading']}"
            if not cit["resolved"]:
                st.error(f"{label} — cites a chunk that was not retrieved", icon="⚠️")
                continue
            chunk = next((c for c in contexts if c["chunk_id"] == cit["chunk_id"]), None)
            with st.expander(label, expanded=False):
                if chunk:
                    st.caption(f"chunk `{chunk['chunk_id']}` · score {chunk['score']:.3f}")
                    st.text(chunk["text"])

    # --- Retrieved chunks, ranked --------------------------------------------
    if body.get("contexts"):
        st.subheader("Retrieved chunks (ranked)")
        for rank, chunk in enumerate(body["contexts"], start=1):
            meta = chunk.get("metadata", {})
            heading = meta.get("section_heading") or ""
            title = f"#{rank} · {meta.get('source_file', '?')}"
            if heading:
                title += f" § {heading}"
            with st.expander(f"{title} — score {chunk['score']:.3f}"):
                st.text(chunk["text"])
