"""
Predictive Maintenance Chatbot - Demo
Siemens Mobility DevOps DataLab - Architecture Design Challenge

Run: streamlit run app.py
Requires OPENAI_API_KEY in .env.
Run generate_pdfs.py then ingest.py first to build the PDF-based RAG index.
"""

import json
import os
import socket
import time
import datetime as dt

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Rail Asset Maintenance Assistant",
    page_icon=":material/train:",
    layout="wide",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
RAG_INDEX_PATH = os.path.join(DATA_DIR, "rag_index.json")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    with open(os.path.join(DATA_DIR, "assets.json")) as f:
        assets = json.load(f)
    with open(os.path.join(DATA_DIR, "failure_patterns.json")) as f:
        failure_patterns = json.load(f)
    with open(os.path.join(DATA_DIR, "risk_scores.json")) as f:
        risk_scores = json.load(f)
    with open(os.path.join(DATA_DIR, "repair_logs.json")) as f:
        repair_logs = json.load(f)
    telemetry = pd.read_csv(os.path.join(DATA_DIR, "telemetry.csv"))
    return assets, failure_patterns, risk_scores, repair_logs, telemetry


@st.cache_data
def load_cached_responses():
    path = os.path.join(DATA_DIR, "cached_responses.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_rag_index():
    """Load PDF-based RAG index. Returns (chunks, embeddings_matrix) or ([], None)."""
    if not os.path.exists(RAG_INDEX_PATH):
        return [], None
    with open(RAG_INDEX_PATH) as f:
        chunks = json.load(f)
    matrix = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / (norms + 1e-10)
    return chunks, matrix


assets, failure_patterns, risk_scores, repair_logs, telemetry = load_data()
rag_chunks, rag_matrix = load_rag_index()
cached_responses = load_cached_responses()

assets_by_id = {a["asset_id"]: a for a in assets}
risk_by_id = {r["asset_id"]: r for r in risk_scores}
patterns_by_id = {p["pattern_id"]: p for p in failure_patterns}

PDF_RAG_AVAILABLE = rag_matrix is not None


# ---------------------------------------------------------------------------
# Connectivity check  (not cached — must reflect live network state)
# ---------------------------------------------------------------------------
def check_connectivity(host="api.openai.com", port=443, timeout=2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_connection_status() -> tuple[str, str, str]:
    """Returns (label, icon, colour-keyword) for the current connection state."""
    if st.session_state.get("simulate_offline"):
        return "Offline – No connectivity", ":material/wifi_off:", "error"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Offline – No API key", ":material/key_off:", "error"
    if check_connectivity():
        return "Online", ":material/wifi:", "success"
    return "Offline – No connectivity", ":material/wifi_off:", "error"


def is_offline() -> bool:
    return st.session_state.get("simulate_offline", False) or client is None


def find_cached_response(query: str, asset_type: str = "") -> str | None:
    """Keyword-overlap match against cached Q&A. Returns answer text or None."""
    if not cached_responses:
        return None
    query_tokens = set(query.lower().split())
    best_score, best_answer = 0, None
    for entry in cached_responses:
        keywords = set(k.lower() for k in entry["keywords"])
        score = len(query_tokens & keywords)
        if asset_type and entry.get("asset_type") == asset_type:
            score *= 1.4
        if score > best_score:
            best_score = score
            best_answer = entry["answer"]
    return best_answer if best_score >= 2 else None


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


client = get_client()


# ---------------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------------
def _embed_query(query: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-10)


def semantic_retrieve(query: str, asset_type: str, language: str = "EN", k: int = 4) -> list[dict]:
    """Return top-k chunks by cosine similarity, boosting same asset_type."""
    if not PDF_RAG_AVAILABLE or client is None:
        return []

    query_vec = _embed_query(query)
    scores = rag_matrix @ query_vec  # cosine similarity (pre-normalised)

    # Boost chunks matching the selected asset type
    boosts = np.array([
        1.3 if c["asset_type"] == asset_type else 1.0
        for c in rag_chunks
    ], dtype=np.float32)
    scores = scores * boosts

    # Optionally prefer language match
    lang_boosts = np.array([
        1.2 if c["language"] == language else 1.0
        for c in rag_chunks
    ], dtype=np.float32)
    scores = scores * lang_boosts

    top_indices = np.argsort(scores)[::-1][:k]
    return [rag_chunks[i] for i in top_indices]


def retrieve_repair_logs(asset_id=None, asset_type=None, max_logs=3):
    logs = repair_logs
    if asset_id:
        logs = [l for l in logs if l["asset_id"] == asset_id]
    elif asset_type:
        logs = [l for l in logs if l["asset_type"] == asset_type]
    return logs[:max_logs]


def build_context_block(asset, rag_hits, logs, telemetry_summary=None, candidate_patterns=None):
    parts = ["## Asset details", json.dumps(asset, indent=2)]

    if rag_hits:
        parts.append("\n## Relevant maintenance manual excerpts (from PDF knowledge base)")
        for hit in rag_hits:
            parts.append(f"### {hit['doc_id']} — page {hit['page']} ({hit['language']})")
            parts.append(hit["text"])

    if logs:
        parts.append("\n## Relevant historical repair logs")
        for log in logs:
            parts.append(json.dumps(log, indent=2))

    if telemetry_summary:
        parts.append("\n## Recent telemetry summary (last 30 days)")
        parts.append(telemetry_summary)

    if candidate_patterns:
        parts.append("\n## Candidate known failure patterns")
        for p in candidate_patterns:
            parts.append(json.dumps(p, indent=2))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call — streaming generator
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a maintenance assistant chatbot for Siemens Mobility rail infrastructure "
    "technicians (point machines, track circuits, axle counters). You answer technical "
    "questions, help diagnose faults by correlating telemetry with known failure "
    "patterns, and explain predictive maintenance recommendations.\n\n"
    "Rules:\n"
    "- Ground every answer in the provided context (manual excerpts, repair logs, "
    "telemetry, failure patterns). Cite document IDs and page numbers (e.g. MAN-PM-002 "
    "p.3) when referencing manual content.\n"
    "- If the context does not contain enough information to answer confidently, say so "
    "explicitly and recommend escalation to a senior engineer rather than guessing — "
    "especially for anything safety-related.\n"
    "- Keep responses concise and field-friendly: technicians may be reading this on a "
    "phone in the field.\n"
    "- Always respond in {lang} regardless of the language the question was written in. "
    "The technician has set {lang} as their preferred language — honour it even if they "
    "typed their question in another language.\n"
    "- When recommending an action, be specific: reference manual sections, torque "
    "specs, part numbers, and time estimates where available in the context."
)

CACHE_MISS_RESPONSE = (
    "**[Offline mode — no cache match]** No network connection and no relevant entry "
    "found in the local knowledge cache for this query.\n\n"
    "Try rephrasing, or sync the manual knowledge base next time you have connectivity."
)


def _build_system_prompt(lang: str) -> str:
    lang_full = "German" if lang == "DE" else "English"
    return SYSTEM_PROMPT.format(lang=lang_full)


def stream_llm(user_message: str, context_block: str, history: list[dict] | None = None,
               asset_type: str = "", lang: str = "EN"):
    """Streaming generator — yields text tokens. Use with st.write_stream()."""
    if is_offline():
        cached = find_cached_response(user_message, asset_type)
        response = cached if cached else CACHE_MISS_RESPONSE
        prefix = "📦 **[Cached response — offline mode]**\n\n" if cached else ""
        full = prefix + response
        words = full.split(" ")
        for i, word in enumerate(words):
            yield word + ("" if i == len(words) - 1 else " ")
            time.sleep(0.015)
        return

    messages = [{"role": "system", "content": _build_system_prompt(lang)}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context_block}\n\nTECHNICIAN QUESTION:\n{user_message}",
    })

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def call_llm(user_message: str, context_block: str, asset_type: str = "", lang: str = "EN") -> str:
    """Non-streaming call for tabs that don't use write_stream."""
    if is_offline():
        cached = find_cached_response(user_message, asset_type)
        if cached:
            return "📦 **[Cached response — offline mode]**\n\n" + cached
        return CACHE_MISS_RESPONSE
    messages = [
        {"role": "system", "content": _build_system_prompt(lang)},
        {"role": "user", "content": f"CONTEXT:\n{context_block}\n\nTECHNICIAN QUESTION:\n{user_message}"},
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, max_tokens=1024, messages=messages)
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title(":material/train: Rail Asset Maintenance Assistant")
st.sidebar.caption("Siemens Mobility DevOps DataLab — Predictive Maintenance Chatbot demo")

# Connection status indicator
conn_label, conn_icon, conn_colour = get_connection_status()
conn_col, btn_col = st.sidebar.columns([3, 1])
with conn_col:
    if conn_colour == "success":
        st.success(conn_label, icon=conn_icon)
    else:
        st.error(conn_label, icon=conn_icon)
with btn_col:
    st.write("")  # vertical alignment nudge
    if st.button("↺", help="Recheck connectivity", use_container_width=True):
        st.rerun()

with st.sidebar.expander(":material/science: Demo controls", expanded=False):
    st.toggle(
        "Simulate offline",
        key="simulate_offline",
        help="Forces offline mode so you can demo tunnel/connectivity loss without losing network.",
    )

st.sidebar.divider()

if PDF_RAG_AVAILABLE:
    st.sidebar.success(
        f"PDF knowledge base loaded ({len(rag_chunks)} chunks from {len(set(c['doc_id'] for c in rag_chunks))} manuals)",
        icon=":material/auto_stories:",
    )
else:
    st.sidebar.warning(
        "PDF RAG index not found. Run `python generate_pdfs.py` then `python ingest.py`.",
        icon=":material/description:",
    )

depots = sorted({a["depot"] for a in assets})
selected_depot = st.sidebar.selectbox("Select depot", depots)

depot_assets = [a for a in assets if a["depot"] == selected_depot]
asset_options = {f"{a['asset_id']} — {a['asset_type']} ({a['location']})": a["asset_id"] for a in depot_assets}
selected_label = st.sidebar.selectbox("Select asset", list(asset_options.keys()))
selected_asset_id = asset_options[selected_label]
selected_asset = assets_by_id[selected_asset_id]

with st.sidebar.expander("Asset details", expanded=True):
    st.write(f"**Depot:** {selected_asset['depot']}")
    st.write(f"**Type:** {selected_asset['asset_type']}")
    st.write(f"**Manufacturer:** {selected_asset['manufacturer']}")
    st.write(f"**Location:** {selected_asset['location']}")
    st.write(f"**Installed:** {selected_asset['install_date']}")
    st.write(f"**Last inspection:** {selected_asset['last_inspection']}")

risk_info = risk_by_id.get(selected_asset_id)
if risk_info:
    risk_pct = int(risk_info["risk_score"] * 100)
    if risk_info["risk_score"] >= 0.6:
        st.sidebar.error(
            f"Failure risk: {risk_pct}% — predicted within {risk_info['predicted_days_to_failure']} days",
            icon=":material/warning:",
        )
    else:
        st.sidebar.success(f"Failure risk: {risk_pct}% — nominal", icon=":material/check_circle:")

language = st.sidebar.radio("Response language", ["EN", "DE"], horizontal=True)


# ---------------------------------------------------------------------------
# SAP PM workflow
# ---------------------------------------------------------------------------
SAP_STAGES = [
    "Open",
    "Submitted to SAP",
    "Pending Approval",
    "Released",
    "In Progress",
    "Completed",
]

SAP_STAGE_ACTIONS = {
    "Open":             (":material/send: Confirm to SAP",          "Pushes the work order into SAP PM for supervisor review."),
    "Submitted to SAP": (":material/approval: Approve",             "Supervisor approves scope, parts, and labour hours."),
    "Pending Approval": (":material/assignment_ind: Release to Technician", "Releases the WO and assigns it to the maintenance team."),
    "Released":         (":material/play_arrow: Start Work",        "Technician confirms work has begun on-site."),
    "In Progress":      (":material/task_alt: Mark Complete (TECO)", "Technical completion — work done, awaiting cost settlement."),
}


def render_sap_stepper(current_stage: str) -> str:
    """Returns an HTML horizontal stepper for the SAP approval cycle."""
    current_idx = SAP_STAGES.index(current_stage) if current_stage in SAP_STAGES else 0
    C_DONE    = "#22c55e"
    C_CURRENT = "#f59e0b"
    C_PENDING = "#374151"
    T_DONE    = "#9ca3af"
    T_CURRENT = "#f59e0b"
    T_PENDING = "#6b7280"

    nodes = []
    for i, stage in enumerate(SAP_STAGES):
        if i < current_idx:
            bg, border, symbol, tc = C_DONE, C_DONE, "✓", T_DONE
            fc = "white"
        elif i == current_idx:
            bg, border, symbol, tc = "transparent", C_CURRENT, str(i + 1), T_CURRENT
            fc = C_CURRENT
        else:
            bg, border, symbol, tc = "transparent", C_PENDING, str(i + 1), T_PENDING
            fc = C_PENDING

        nodes.append(f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:5px;min-width:72px;">
            <div style="width:28px;height:28px;border-radius:50%;border:2px solid {border};
                background:{bg};display:flex;align-items:center;justify-content:center;
                color:{fc};font-size:11px;font-weight:700;">{symbol}</div>
            <div style="font-size:9.5px;color:{tc};text-align:center;line-height:1.3;">{stage}</div>
        </div>""")

    parts = []
    for i, node in enumerate(nodes):
        parts.append(node)
        if i < len(nodes) - 1:
            lc = C_DONE if i < current_idx else C_PENDING
            parts.append(f'<div style="flex:1;height:2px;background:{lc};margin-top:14px;margin-bottom:18px;"></div>')

    return f'<div style="display:flex;align-items:flex-start;padding:10px 0 4px 0;">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Risk ticker
# ---------------------------------------------------------------------------
def render_risk_ticker() -> str:
    flagged = sorted(
        [r for r in risk_scores if r["classification"] in ("Red", "Yellow")],
        key=lambda r: r["risk_score"], reverse=True,
    )
    if not flagged:
        return ""

    items = []
    for r in flagged:
        asset  = assets_by_id.get(r["asset_id"], {})
        pattern = patterns_by_id.get(r["matched_pattern"] or "", {})
        color  = "#ef4444" if r["classification"] == "Red" else "#f59e0b"
        icon   = "▲" if r["classification"] == "Red" else "●"
        pct    = int(r["risk_score"] * 100)
        name   = asset.get("name", r["asset_id"])
        pat    = pattern.get("name", "—")
        days   = r["predicted_days_to_failure"]
        glow   = f"text-shadow:0 0 8px {color}88;" if r["classification"] == "Red" else ""
        items.append(
            f'<span style="color:{color};{glow} margin:0 28px;font-size:12.5px;letter-spacing:.3px;">'
            f'<span style="opacity:.7;">{icon}</span>&nbsp;'
            f'<strong>{r["asset_id"]}</strong>&nbsp;·&nbsp;{name}&nbsp;·&nbsp;'
            f'<strong>{pct}%</strong>&nbsp;·&nbsp;{pat}&nbsp;·&nbsp;'
            f'<span style="opacity:.8;">{days}d to failure</span>'
            f'</span>'
        )

    sep = '<span style="color:#374151;margin:0 4px;">◆</span>'
    chunk = sep.join(items)
    content = chunk + sep + chunk  # duplicate for seamless loop
    duration = max(25, len(flagged) * 7)

    bg = "#0f1117"
    return f"""
<style>
@keyframes risk-ticker {{
  0%   {{ transform: translateX(0); }}
  100% {{ transform: translateX(-50%); }}
}}
.risk-ticker-scroll {{
  display: inline-block;
  white-space: nowrap;
  animation: risk-ticker {duration}s linear infinite;
  cursor: default;
}}
.risk-ticker-scroll:hover {{
  animation-play-state: paused;
}}
</style>
<div style="
  position:relative; display:flex; align-items:center;
  background:{bg};
  border-top:1px solid #1f2937; border-bottom:1px solid #1f2937;
  height:36px; overflow:hidden; margin-bottom:14px;
">
  <!-- fixed badge -->
  <div style="
    flex-shrink:0; padding:0 12px;
    font-size:10px; font-weight:800; letter-spacing:2px;
    color:#ef4444; border-right:1px solid #1f2937;
    height:100%; display:flex; align-items:center;
    background:{bg}; z-index:3;
  ">LIVE RISK</div>
  <!-- left fade -->
  <div style="position:absolute;left:80px;top:0;width:40px;height:100%;
    background:linear-gradient(90deg,{bg},transparent);z-index:2;pointer-events:none;"></div>
  <!-- scrolling content -->
  <div style="overflow:hidden;flex:1;height:100%;display:flex;align-items:center;">
    <div class="risk-ticker-scroll">
      {content}
    </div>
  </div>
  <!-- right fade -->
  <div style="position:absolute;right:0;top:0;width:40px;height:100%;
    background:linear-gradient(-90deg,{bg},transparent);z-index:2;pointer-events:none;"></div>
</div>
"""

st.markdown(render_risk_ticker(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_qa, tab_predict, tab_cmms = st.tabs([
    ":material/menu_book: Manual Q&A",
    ":material/insights: Predictive Insights",
    ":material/assignment: Work Orders (CMMS)",
])


# ---------------------------------------------------------------------------
# Tab 1: Manual Q&A  (PDF RAG + streaming)
# ---------------------------------------------------------------------------
with tab_qa:
    title_col, sync_col = st.columns([3, 1])
    with title_col:
        st.subheader("Ask a technical maintenance question")
        st.caption(
            "Answers are grounded in PDF maintenance manuals retrieved semantically for your "
            "question and the selected asset type (retrieval-augmented generation)."
        )
    with sync_col:
        st.write("")  # top padding
        offline_now = is_offline()
        last_sync = st.session_state.get("last_manual_sync")
        sync_label = ":material/sync: Sync Manuals"
        sync_help = (
            "Cannot sync — app is offline." if offline_now
            else "Download latest manuals to local cache for offline use."
        )
        if st.button(sync_label, disabled=offline_now, help=sync_help, use_container_width=True):
            with st.spinner("Syncing manuals…"):
                time.sleep(2)
            st.session_state.last_manual_sync = dt.datetime.now().strftime("%d %b %Y, %H:%M")
            st.toast("Manuals synced — available offline.", icon=":material/check_circle:")
            st.rerun()
        if last_sync:
            st.caption(f"Last synced: {last_sync}")
        elif offline_now:
            st.caption("⚠ Not synced")

    # AI disclaimer — shown until acknowledged this session
    if not st.session_state.get("disclaimer_acknowledged"):
        st.markdown("""
<div style="
    border: 2px solid #e6a817;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 16px;
">
    <div style="
        background: #e6a817;
        padding: 6px 14px;
        color: #1a1a1a;
        font-weight: 800;
        font-size: 11px;
        letter-spacing: 2px;
        border-left: 6px solid #1a1a1a;
    ">⚠&nbsp;&nbsp;AI ADVISORY — MAINTENANCE SAFETY NOTICE</div>
    <div style="
        background: #1f1a0a;
        padding: 12px 16px;
        color: #e8d8a0;
        font-size: 13.5px;
        line-height: 1.6;
    ">
        Responses are generated by <strong>GPT-4o</strong> and grounded in retrieved manual excerpts.
        This assistant may produce errors or omissions. <strong>Do not act on AI output alone
        for any safety-critical maintenance decision</strong> — always verify against official
        Siemens documentation and consult a senior engineer when in doubt.
        <br><br>
        <span style="font-size:12px; color:#9a8a5a;">
            Treat this tool as a first-pass reference, not a replacement for certified procedures.
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
        if st.button("✓ Understood — don't show again this session", type="secondary"):
            st.session_state.disclaimer_acknowledged = True
            st.rerun()
    else:
        st.markdown("<span style='color:#ef4444;font-size:13px;'>⚠ AI-generated responses — verify against official documentation before acting.</span>", unsafe_allow_html=True)

    if st.session_state.get("disclaimer_acknowledged"):
        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []

        # Context badge + clear button
        ctx_col, clr_col = st.columns([5, 1])
        with ctx_col:
            risk_info = risk_by_id.get(selected_asset_id)
            zone_tag = ""
            if risk_info:
                zone = risk_info.get("classification", "")
                icon = {"Red": "🔴", "Yellow": "🟡", "Green": "🟢"}.get(zone, "")
                zone_tag = f"&nbsp;·&nbsp;{icon} {zone} Zone"
            st.markdown(
                f"<div style='font-size:12px;color:#6b7280;padding:4px 0 8px 0;'>"
                f":material/sensors: Answering for&nbsp;"
                f"<strong style='color:#d1d5db;'>{selected_asset_id}</strong>"
                f"&nbsp;·&nbsp;{selected_asset['name']}"
                f"&nbsp;·&nbsp;{selected_asset['asset_type']}"
                f"{zone_tag}"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;:material/translate: <strong style='color:#d1d5db;'>{language}</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with clr_col:
            if st.session_state.qa_history:
                if st.button("Clear chat", key="clear_chat", use_container_width=True):
                    st.session_state.qa_history = []
                    st.rerun()

        # Empty state
        if not st.session_state.qa_history:
            st.markdown("""
<div style="
    text-align:center; padding:40px 20px; color:#4b5563;
    border:1px dashed #1f2937; border-radius:12px; margin:12px 0 24px 0;
">
    <div style="font-size:36px; margin-bottom:12px;">🔧</div>
    <div style="font-size:16px; font-weight:600; color:#9ca3af; margin-bottom:6px;">
        Ask anything about this asset
    </div>
    <div style="font-size:13px; line-height:1.7; max-width:480px; margin:0 auto;">
        Answers are grounded in PDF maintenance manuals retrieved for the selected asset type.
        Try one of the example questions below, or type your own.
    </div>
</div>
""", unsafe_allow_html=True)

        for msg in st.session_state.qa_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        example_questions = [
            "What is the maximum allowed switch blade gap?",
            "What torque should I use on the sensor bracket bolts?",
            "Was ist die normale Betriebsspannung des Empfaengers?",
        ]
        cols = st.columns(len(example_questions))
        example_clicked = None
        for col, q in zip(cols, example_questions):
            if col.button(q, key=f"ex_{q}", use_container_width=True):
                example_clicked = q

        user_q = st.chat_input("Type your question...")
        final_q = user_q or example_clicked

        if final_q:
            st.session_state.qa_history.append({"role": "user", "content": final_q})
            with st.chat_message("user"):
                st.markdown(final_q)

            rag_hits = semantic_retrieve(final_q, selected_asset["asset_type"], language, k=4)
            logs = retrieve_repair_logs(asset_type=selected_asset["asset_type"])
            context = build_context_block(selected_asset, rag_hits, logs)

            history = st.session_state.qa_history[:-1]

            with st.chat_message("assistant"):
                answer = st.write_stream(stream_llm(final_q, context, history, asset_type=selected_asset["asset_type"], lang=language))
                if rag_hits:
                    with st.expander(":material/source: Retrieved PDF sources", expanded=False):
                        for hit in rag_hits:
                            st.markdown(f"**{hit['doc_id']}** — page {hit['page']}  `{hit['asset_type']}`  `{hit['language']}`")

            st.session_state.qa_history.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# Tab 3: Predictive Insights
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Fleet-wide predictive maintenance intelligence")
    st.caption("Asset health, risk classification, and remaining useful life across all monitored assets.")

    risk_df = pd.DataFrame(risk_scores)
    risk_df = risk_df.merge(
        pd.DataFrame(assets)[["asset_id", "name", "depot"]],
        on="asset_id", how="left",
    )
    risk_df = risk_df.sort_values("risk_score", ascending=False)

    # KPI row
    red_count = (risk_df["classification"] == "Red").sum()
    yellow_count = (risk_df["classification"] == "Yellow").sum()
    avg_remaining = risk_df["remaining_life_pct"].mean()
    min_rul = risk_df["rul_days"].min()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Assets monitored", len(risk_df))
    k2.metric("🔴 Red zone", int(red_count), help="Risk ≥ 70% — immediate action required")
    k3.metric("🟡 Yellow zone", int(yellow_count), help="Risk 40–69% — plan maintenance")
    k4.metric("Avg remaining life", f"{avg_remaining:.0f}%", help="Fleet-wide average remaining design lifespan")
    k5.metric("Nearest RUL", f"{min_rul} days", help="Shortest Remaining Useful Life across the fleet")

    st.divider()

    # Intelligence table
    st.markdown("### Asset risk intelligence")
    ZONE_LABEL = {"Red": "🔴 Red Zone", "Yellow": "🟡 Yellow Zone", "Green": "🟢 Green Zone"}
    display_df = risk_df[[
        "asset_id", "name", "asset_type", "depot",
        "classification", "risk_score", "rul_days",
        "remaining_life_pct", "predicted_days_to_failure", "assessment_date",
    ]].copy()
    display_df["risk_score"] = (display_df["risk_score"] * 100).round(0).astype(int)
    display_df["classification"] = display_df["classification"].map(ZONE_LABEL)
    display_df.columns = [
        "Asset ID", "Name", "Type", "Depot",
        "Zone", "Risk %", "RUL (days)",
        "Remaining Life %", "Days to Failure", "Assessment Date",
    ]
    st.dataframe(
        display_df, use_container_width=True, hide_index=True,
        column_config={
            "Risk %": st.column_config.ProgressColumn(
                "Risk %", min_value=0, max_value=100, format="%d%%"
            ),
            "Remaining Life %": st.column_config.ProgressColumn(
                "Remaining Life %", min_value=0, max_value=100, format="%d%%"
            ),
            "RUL (days)": st.column_config.NumberColumn("RUL (days)", help="Remaining Useful Life — estimated days before major overhaul or replacement is needed"),
        },
    )

    # Per-asset insight cards (Red + Yellow only)
    flagged = risk_df[risk_df["classification"].isin(["Red", "Yellow"])]
    if not flagged.empty:
        st.divider()
        st.markdown("### Asset insights")
        st.caption("Detailed observations for flagged assets. Red zone cards are expanded by default.")
        for _, row in flagged.iterrows():
            zone_icon = "🔴" if row["classification"] == "Red" else "🟡"
            with st.expander(
                f"{zone_icon} {row['asset_id']} — {row['name']}  ·  {row['asset_type']}",
                expanded=(row["classification"] == "Red"),
            ):
                ic1, ic2 = st.columns([3, 1])
                with ic1:
                    st.markdown(f"**Insight**")
                    st.markdown(row["insight"])
                    st.markdown(f"**Recommended action**")
                    st.markdown(row["recommended_action"])
                with ic2:
                    st.metric("Risk", f"{int(row['risk_score'] * 100)}%")
                    st.metric("Days to failure", row["predicted_days_to_failure"])
                    st.metric("RUL", f"{row['rul_days']} days")
                    st.metric("Remaining life", f"{row['remaining_life_pct']}%")

    # Work order creation from high-risk assets
    high_risk_rows = risk_df[risk_df["risk_score"] >= 0.6]
    if not high_risk_rows.empty:
        st.markdown("### Create work orders")
        st.caption("Assets exceeding 60% failure risk with a matched failure pattern.")
        for _, row in high_risk_rows.iterrows():
            pattern = patterns_by_id.get(row["matched_pattern"])
            asset = assets_by_id.get(row["asset_id"])
            if not pattern or not asset:
                continue
            already_created = any(
                wo["asset_id"] == row["asset_id"]
                for wo in st.session_state.get("work_orders", [])
            )
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 2, 1])
                c1.markdown(f"**{row['asset_id']}** — {row['asset_type']}")
                c1.caption(asset["location"])
                c2.markdown(f"**Pattern:** {pattern['name']}")
                c2.caption(f"Est. repair: {pattern['estimated_repair_minutes']} min · Parts: {', '.join(pattern['spare_parts'])}")
                with c3:
                    if already_created:
                        st.success("WO created", icon=":material/check:")
                    else:
                        if st.button(
                            ":material/note_add: Create WO",
                            key=f"wo_{row['asset_id']}",
                            use_container_width=True,
                            type="primary",
                        ):
                            created_ts = dt.datetime.now().strftime("%d %b %Y, %H:%M")
                            new_order = {
                                "order_id": f"WO-{1000 + len(st.session_state.get('work_orders', []))}",
                                "asset_id": row["asset_id"],
                                "asset_type": row["asset_type"],
                                "location": asset["location"],
                                "created": dt.date.today().isoformat(),
                                "diagnosis": pattern["name"],
                                "recommended_action": pattern["recommended_action"],
                                "spare_parts": pattern["spare_parts"],
                                "estimated_minutes": pattern["estimated_repair_minutes"],
                                "status": "Open",
                                "sap_stage": "Open",
                                "stage_history": [{"stage": "Open", "timestamp": created_ts}],
                            }
                            st.session_state.setdefault("work_orders", []).append(new_order)
                            st.rerun()

    st.divider()
    st.markdown("### Ask about fleet-wide patterns")

    fleet_examples = [
        "Which assets need attention this week and why?",
        "Which depot has the highest combined risk right now?",
        "What spare parts should I stock up on given current failure patterns?",
    ]
    fleet_cols = st.columns(len(fleet_examples))
    for col, q in zip(fleet_cols, fleet_examples):
        if col.button(q, key=f"fleet_ex_{q}", use_container_width=True):
            st.session_state.fleet_q_value = q
            st.session_state.fleet_auto_submit = True
            st.rerun()

    fleet_q = st.text_input("Or type your own question...", key="fleet_q_value")

    auto_submit = st.session_state.pop("fleet_auto_submit", False)
    if (st.button(":material/auto_awesome: Get recommendation", key="fleet_button") or auto_submit) and fleet_q:
        context = (
            "## Fleet-wide risk scores\n" + json.dumps(risk_scores, indent=2) +
            "\n## Known failure patterns\n" + json.dumps(failure_patterns, indent=2)
        )
        with st.spinner("Analysing fleet data..."):
            answer = call_llm(fleet_q, context, lang=language)
        st.markdown(answer)


# ---------------------------------------------------------------------------
# Tab 4: Work Orders (CMMS)
# ---------------------------------------------------------------------------
with tab_cmms:
    st.subheader("Maintenance work orders")
    st.caption("Track work orders through the SAP PM approval cycle — from creation to technical completion.")

    all_work_orders = st.session_state.get("work_orders", [])

    # Depot filter (consistent with sidebar)
    fc1, fc2 = st.columns([2, 3])
    with fc1:
        wo_depot_filter = st.selectbox(
            "Filter by depot",
            ["All depots"] + sorted({a["depot"] for a in assets}),
            key="wo_depot_filter",
        )

    filtered_wos = [
        wo for wo in all_work_orders
        if wo_depot_filter == "All depots"
        or assets_by_id.get(wo["asset_id"], {}).get("depot") == wo_depot_filter
    ]

    # KPI header
    stage_buckets = {"Open": 0, "In SAP": 0, "Active": 0, "Completed": 0}
    for wo in filtered_wos:
        s = wo.get("sap_stage", "Open")
        if s == "Open":
            stage_buckets["Open"] += 1
        elif s in ("Submitted to SAP", "Pending Approval"):
            stage_buckets["In SAP"] += 1
        elif s in ("Released", "In Progress"):
            stage_buckets["Active"] += 1
        elif s == "Completed":
            stage_buckets["Completed"] += 1

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Open", stage_buckets["Open"], help="Created, not yet submitted to SAP")
    k2.metric("In SAP", stage_buckets["In SAP"], help="Awaiting submission or supervisor approval")
    k3.metric("Active", stage_buckets["Active"], help="Released to technician or work underway")
    k4.metric("Completed", stage_buckets["Completed"], help="Technically complete (TECO)")

    st.divider()

    # Work order cards — active first, then completed (collapsed)
    if not filtered_wos:
        st.info(
            "No work orders yet. Go to **Predictive Insights**, find a flagged asset, "
            "and click **Create WO** to generate one."
        )
    else:
        active_wos  = [wo for wo in filtered_wos if wo.get("sap_stage") != "Completed"]
        done_wos    = [wo for wo in filtered_wos if wo.get("sap_stage") == "Completed"]

        for wo in list(reversed(active_wos)) + list(reversed(done_wos)):
            wo_idx = all_work_orders.index(wo)
            current_stage = wo.get("sap_stage", "Open")
            stage_idx = SAP_STAGES.index(current_stage) if current_stage in SAP_STAGES else 0
            is_done = current_stage == "Completed"
            depot = assets_by_id.get(wo["asset_id"], {}).get("depot", "—")

            with st.container(border=True):
                hc1, hc2 = st.columns([3, 1])
                with hc1:
                    st.markdown(f"**{wo['order_id']}** — {wo['asset_id']} · {wo['asset_type']}")
                    st.caption(f"{wo['location']}  ·  {depot}")
                with hc2:
                    if is_done:
                        st.success("Completed", icon=":material/task_alt:")
                    elif stage_idx >= 3:
                        st.info(current_stage, icon=":material/build:")
                    elif stage_idx >= 1:
                        st.warning(current_stage, icon=":material/pending:")
                    else:
                        st.warning("Open — not yet in SAP", icon=":material/circle:")

                st.markdown(render_sap_stepper(current_stage), unsafe_allow_html=True)

                # Button tracks the current stage node position
                if not is_done and current_stage in SAP_STAGE_ACTIONS:
                    btn_label, btn_help = SAP_STAGE_ACTIONS[current_stage]
                    stage_cols = st.columns(len(SAP_STAGES))
                    with stage_cols[stage_idx]:
                        if st.button(btn_label, key=f"sap_{wo['order_id']}", type="primary",
                                     help=btn_help, use_container_width=True):
                            next_stage = SAP_STAGES[stage_idx + 1]
                            st.session_state.work_orders[wo_idx]["sap_stage"] = next_stage
                            st.session_state.work_orders[wo_idx]["status"] = next_stage
                            st.session_state.work_orders[wo_idx].setdefault("stage_history", []).append({
                                "stage": next_stage,
                                "timestamp": dt.datetime.now().strftime("%d %b %Y, %H:%M"),
                            })
                            st.rerun()

                with st.expander("Details & audit trail", expanded=False):
                    dc1, dc2 = st.columns([3, 1])
                    with dc1:
                        st.markdown(f"**Diagnosis:** {wo['diagnosis']}")
                        st.markdown(f"**Recommended action:** {wo['recommended_action']}")
                        st.markdown("**Spare parts:** " + ", ".join(wo["spare_parts"]))
                    with dc2:
                        st.markdown(f"**Created:** {wo['created']}")
                        st.markdown(f"**Est. time:** {wo['estimated_minutes']} min")
                    if wo.get("stage_history"):
                        st.markdown("**Audit trail**")
                        for entry in wo["stage_history"]:
                            st.caption(f"→ {entry['stage']} — {entry['timestamp']}")

    # ---------------------------------------------------------------------------
    # Repair History — historical logs + completed WOs unified
    # ---------------------------------------------------------------------------
    st.divider()
    st.markdown("### Repair history")
    st.caption("Closed-loop record: historical repairs and completed work orders in one view.")

    # Completed WOs → repair log rows
    completed_as_logs = []
    for wo in all_work_orders:
        if wo.get("sap_stage") != "Completed":
            continue
        completion_entry = next(
            (e for e in reversed(wo.get("stage_history", [])) if e["stage"] == "Completed"),
            None,
        )
        completed_as_logs.append({
            "log_id": f"RL-{2035 + len(completed_as_logs)}",
            "wo_ref": wo["order_id"],
            "asset_id": wo["asset_id"],
            "asset_type": wo["asset_type"],
            "depot": assets_by_id.get(wo["asset_id"], {}).get("depot", "—"),
            "date": completion_entry["timestamp"][:10] if completion_entry else wo["created"],
            "issue": wo["diagnosis"],
            "action_taken": wo["recommended_action"],
            "technician": "SAP PM (auto)",
            "resolution_minutes": wo["estimated_minutes"],
            "source": "Work Order",
        })

    historical = [
        {**log, "wo_ref": "—", "depot": assets_by_id.get(log["asset_id"], {}).get("depot", "—"), "source": "Historical"}
        for log in repair_logs
    ]

    all_history = historical + completed_as_logs
    if all_history:
        hist_df = pd.DataFrame(all_history).sort_values("date", ascending=False)
        if wo_depot_filter != "All depots":
            hist_df = hist_df[hist_df["depot"] == wo_depot_filter]
        st.dataframe(
            hist_df[["log_id", "wo_ref", "asset_id", "asset_type", "depot", "date", "issue", "action_taken", "technician", "resolution_minutes", "source"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "log_id": st.column_config.TextColumn("Log ID"),
                "wo_ref": st.column_config.TextColumn("WO Ref", help="Work order that generated this repair entry. '—' for pre-system historical logs."),
                "resolution_minutes": st.column_config.NumberColumn("Resolved (min)"),
                "source": st.column_config.TextColumn("Source", help="'Historical' = pre-loaded log · 'Work Order' = completed via SAP cycle"),
            },
        )
