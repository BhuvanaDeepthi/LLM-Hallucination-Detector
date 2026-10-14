import streamlit as st
from modules.database import get_system_stats

def render_dashboard():
    """Renders the Home Dashboard page with real metrics and workflow architecture."""

    # Hero Section
    st.markdown("""
    <div class="app-header">
        <div class="app-title">LLM Hallucination Detector</div>
        <div class="app-subtitle">
            Verifying AI-Generated Answers Using Retrieval-Augmented Generation (RAG) and Natural Language Inference (NLI)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Live Database Statistics Cards
    stats = get_system_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Documents Processed</div>
            <div class="stat-value">{stats['documents_processed']}</div>
            <div class="stat-subtext">Active PDF Indexes</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Audit Sessions</div>
            <div class="stat-value">{stats['verification_sessions']}</div>
            <div class="stat-subtext">Q&A Audits Completed</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Claims Verified</div>
            <div class="stat-value">{stats['claims_checked']}</div>
            <div class="stat-subtext">Atomic Factual Units</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Claims Flagged</div>
            <div class="stat-value" style="color: #F87171;">{stats['claims_flagged']}</div>
            <div class="stat-subtext">Contradicted / Unsupported</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Action to Unified Workspace
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 18px 24px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <h4 style="margin: 0; color: #F8FAFC;">⚡ Unified All-in-One Audit Workspace</h4>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">Upload a PDF document, ask questions, and audit claim verdicts all on one single interactive screen.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Open Upload & Verification Workspace", type="primary", use_container_width=True):
        st.session_state["current_page"] = "Workspace"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # How It Works & Architecture Flow
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("""
        <div class="feature-card">
            <h3>🔬 How the Verification System Works</h3>
            <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.6;">
                Large Language Models frequently produce assertions that appear persuasive yet diverge from factual reality. 
                Rather than treating generative text as infallible, this application enforces a four-stage empirical verification pipeline:
            </p>
            <ol style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.8; padding-left: 20px;">
                <li><b>Page-Aware Document Ingestion:</b> Extracts text preserving exact page coordinates and builds dense MiniLM embeddings stored in a FAISS index.</li>
                <li><b>Grounded RAG Generation:</b> Answers user inquiries utilizing local <code>gemma3:latest</code> or fast Gemini Flash under strict factual constraints.</li>
                <li><b>Atomic Claim Decomposition:</b> Breaks answers down into discrete, self-contained factual statements.</li>
                <li><b>Batched NLI Verification:</b> Evaluates each claim against retrieved evidence passages using <code>roberta-large-mnli</code> to compute Entailment, Contradiction, and Neutral probabilities.</li>
                <li><b>LLM-as-a-Judge Audit:</b> Employs an optional second-stage evaluator for ambiguous probability distributions.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="feature-card">
            <h3>⚖️ Verification Verdict Criteria</h3>
            <div style="margin-bottom: 14px;">
                <span class="badge-supported">SUPPORTED</span>
                <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 6px; margin-bottom: 0;">
                    The retrieved source passage provides direct, sufficient textual entailment for the claim (Entailment &ge; 0.50).
                </p>
            </div>
            <div style="margin-bottom: 14px;">
                <span class="badge-contradicted">CONTRADICTED</span>
                <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 6px; margin-bottom: 0;">
                    The source document explicitly conflicts with or refutes the claim (Contradiction &ge; 0.50). This represents an active hallucination.
                </p>
            </div>
            <div>
                <span class="badge-unsupported">UNSUPPORTED</span>
                <p style="color: #94A3B8; font-size: 0.88rem; margin-top: 6px; margin-bottom: 0;">
                    The available document evidence is neutral or insufficient to substantiate the claim. <i>Note: Absence of evidence is not assumed to be proof of falsehood.</i>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
