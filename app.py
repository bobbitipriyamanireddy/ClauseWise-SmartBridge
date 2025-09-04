import os
import requests
import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="ClauseWise — Legal Document Analyzer", layout="wide")
st.title("⚖ ClauseWise — Legal Document Analyzer")

API_BASE = os.getenv("CLAUSEWISE_API", "http://localhost:8000")

if "doc_text" not in st.session_state:
    st.session_state["doc_text"] = ""
if "api_base" not in st.session_state:
    st.session_state["api_base"] = API_BASE

with st.sidebar:
    st.markdown("### Setup")
    st.text_input("FastAPI base URL", value=st.session_state["api_base"], key="api_base")
    st.markdown("Tip: Run the backend first, then this UI.")
    st.divider()
    st.markdown("Features")
    st.checkbox("Enable NER", value=True, key="enable_ner")
    st.checkbox("Enable classification", value=True, key="enable_cls")

def get_api_base():
    return st.session_state.get("api_base", API_BASE).rstrip("/")

def api_post(path: str, json_body: dict):
    url = get_api_base() + path
    try:
        r = requests.post(url, json=json_body, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
        return {}

def api_upload(file_bytes: bytes, filename: str):
    url = get_api_base() + "/extract"
    files = {"file": (filename, file_bytes)}
    try:
        r = requests.post(url, files=files, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {e}")
        return {}

# Tabs
tab_upload, tab_simplify, tab_cls, tab_ner, tab_search, tab_bulk = st.tabs(
    ["📄 Upload", "📝 Simplify Clauses", "🏷 Classify Document", "🔎 Extract Entities (NER)", "🔍 Clause Search", "📑 Bulk Simplify"]
)

# === Upload Tab ===
with tab_upload:
    st.subheader("Upload a legal document (PDF, DOCX, or TXT)")
    uploaded = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
    if uploaded is not None:
        data = uploaded.read()
        uploaded.seek(0)
        with st.spinner("Extracting text…"):
            resp = api_upload(data, uploaded.name)
            text = resp.get("text", "")
            if text:
                st.session_state["doc_text"] = text
                st.success(f"Extracted {len(text)} characters.")
                st.text_area("Extracted Text", value=text[:5000], height=300)
            else:
                st.error("Extraction failed.")

doc_text = st.session_state.get("doc_text", "")

# === Simplify Tab ===
with tab_simplify:
    st.subheader("Simplify specific clause(s)")
    default_clause = doc_text[:1200] if doc_text else ""
    clause = st.text_area("Paste a clause to simplify", value=default_clause, height=200, key="clause_to_simplify")
    if st.button("Simplify"):
        if not clause.strip():
            st.warning("Please paste some text to simplify.")
        else:
            with st.spinner("Simplifying with model…"):
                resp = api_post("/simplify", {"text": clause})
                simplified = resp.get("simplified", "")
                if simplified:
                    st.success("Simplified clause:")
                    st.text_area("Plain-English", value=simplified, height=220)
                else:
                    st.error("Simplification failed.")

# === Classification Tab ===
with tab_cls:
    st.subheader("Classify document type")
    st.caption("Options: NDA, Lease, Employment, Service Agreement")
    sample = doc_text[:4000] if doc_text else ""
    txt = st.text_area("Text to classify", value=sample, height=200, key="text_to_classify")
    if st.session_state.get("enable_cls") and st.button("Classify"):
        with st.spinner("Classifying…"):
            resp = api_post("/classify", {"text": txt})
            label = resp.get("label")
            if label:
                st.success(f"Predicted: {label}")
            else:
                st.error("Classification failed.")

# === NER Tab ===
with tab_ner:
    st.subheader("Named Entity Recognition")
    ner_text = st.text_area(
        "Text to extract entities from",
        value=doc_text[:3000],
        height=200,
        key="ner_text"
    )
    if st.session_state.get("enable_ner") and st.button("Extract Entities"):
        with st.spinner("Finding entities…"):
            resp = api_post("/ner", {"text": ner_text})
            ents = resp.get("entities", [])
            if not ents:
                st.info("No entities found.")
            else:
                # Display only the token text for clarity
                df = pd.DataFrame(ents)
                if "text" in df.columns:
                    st.dataframe(df[["text"]], use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv, file_name="entities.csv", mime="text/csv")

# === Search Tab ===
with tab_search:
    st.subheader("Quick clause search (keyword)")
    query = st.text_input("Keyword or phrase")
    if st.button("Search"):
        if not doc_text.strip():
            st.warning("Upload a document first.")
        elif not query.strip():
            st.warning("Enter a search query.")
        else:
            lines = [ln.strip() for ln in doc_text.splitlines() if ln.strip()]
            hits = [ln for ln in lines if query.lower() in ln.lower()]
            if not hits:
                st.info("No matches.")
            else:
                st.write(f"Found {len(hits)} matching lines:")
                for h in hits[:200]:
                    highlighted = re.sub(f"({query})", r"\1**", h, flags=re.IGNORECASE)
                    st.markdown("- " + highlighted)
                if len(hits) > 200:
                    st.caption("Showing first 200 matches.")

# === Bulk Simplify Tab ===
with tab_bulk:
    st.subheader("📑 Simplify All Clauses in Document")
    if not doc_text.strip():
        st.info("Upload a document first.")
    else:
        if st.button("Simplify All Clauses"):
            clauses = [c.strip() for c in re.split(r"\n+|\.\s+", doc_text) if len(c.strip()) > 20]
            max_clauses = 10  # ✅ limit to prevent timeout
            clauses_to_send = clauses[:max_clauses] if len(clauses) > max_clauses else clauses
            with st.spinner("Simplifying all clauses..."):
                resp = api_post("/bulk_simplify", {"clauses": clauses_to_send})
                results = resp.get("results", [])
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Simplified Clauses", data=csv, file_name="simplified_clauses.csv", mime="text/csv")
            else:
                st.error("Bulk simplification failed.")
