import streamlit as st
from styles import load_custom_css
from modules.database import init_db
import config

# Streamlit Page Configuration
st.set_page_config(
    page_title="LLM Hallucination Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS Styling
load_custom_css()

# Initialize Database Schema
init_db()

# Navigation Pages Mapping (Home Dashboard, Upload & Verify Workspace, Audit History)
PAGES = {
    "Workspace": "⚡ Upload & Verify Workspace",
    "Dashboard": "🏠 Home Dashboard",
    "History": "📜 Audit History"
}

# Session State Initialization
if "current_page" not in st.session_state or st.session_state["current_page"] not in PAGES:
    st.session_state["current_page"] = "Workspace"

if "active_document" not in st.session_state:
    st.session_state["active_document"] = None

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

# Ultra-Clean Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 14px 0;">
        <h2 style="margin: 0; font-size: 1.35rem; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
            🛡️ Hallucination Audit
        </h2>
        <span style="font-size: 0.8rem; color: #94A3B8;">RAG + NLI Verification Engine</span>
    </div>
    """, unsafe_allow_html=True)

    nav_options = list(PAGES.keys())

    # Match index if state changed programmatically
    current_idx = 0
    if st.session_state["current_page"] in nav_options:
        current_idx = nav_options.index(st.session_state["current_page"])

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        format_func=lambda x: PAGES.get(x, x),
        index=current_idx,
        label_visibility="collapsed"
    )

    st.session_state["current_page"] = selected_nav

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 18px 0 14px 0;'>", unsafe_allow_html=True)

    import html

    # Active Document Card
    st.markdown("**Active Document**")
    if st.session_state.get("active_document"):
        doc = st.session_state["active_document"]
        safe_fn = html.escape(str(doc.get('filename', '')))
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 12px; font-size: 0.85rem;">
            <div style="font-weight: 600; color: #10B981; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">📄 {safe_fn}</div>
            <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 4px;">{doc['page_count']} pages | {doc['chunk_count']} passages</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("No document loaded. Upload a PDF on the workspace to begin.")

    # Audit Counts in Sidebar
    if st.session_state.get("last_verification"):
        lv = st.session_state["last_verification"]
        claims = lv.get("claims", [])
        total_c = len(claims)
        supp_c = sum(1 for c in claims if c.get("verdict") == config.VERDICT_SUPPORTED)
        contra_c = sum(1 for c in claims if c.get("verdict") == config.VERDICT_CONTRADICTED)
        unsup_c = sum(1 for c in claims if c.get("verdict") == config.VERDICT_UNSUPPORTED)

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 14px 0 10px 0;'>", unsafe_allow_html=True)
        st.markdown("**Latest Audit Breakdown**")
        st.markdown(f"• **Supported:** :green[**{supp_c}/{total_c}**]")
        st.markdown(f"• **Contradicted:** :red[**{contra_c}/{total_c}**]")
        st.markdown(f"• **Unsupported:** :orange[**{unsup_c}/{total_c}**]")

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.caption("College Mini Project © 2026")

# Dynamic Page Router
if st.session_state["current_page"] == "Workspace":
    from pages.ask_question import render_ask_question
    render_ask_question()
elif st.session_state["current_page"] == "Dashboard":
    from pages.dashboard import render_dashboard
    render_dashboard()
elif st.session_state["current_page"] == "History":
    from pages.verification_history import render_verification_history
    render_verification_history()
