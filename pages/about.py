import streamlit as st

def render_about():
    """Renders the About Project page with technical architecture, presentation notes, and research citations."""

    st.markdown("""
    <div class="app-header">
        <div class="app-title">About the Project</div>
        <div class="app-subtitle">
            College Mini-Project: Automated Detection and Natural Language Inference Verification of LLM Hallucinations in RAG Systems
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Overview Card
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Project Motivation & Problem Statement</h3>
        <p style="color: #CBD5E1; line-height: 1.6; font-size: 0.95rem;">
            Large Language Models (LLMs) frequently generate factual hallucinations—confidently asserting claims that are unsupported by 
            or in direct conflict with reference ground-truth data. In high-stakes domains such as legal discovery, medicine, and academic research, 
            verifying generated statements is critical.
        </p>
        <p style="color: #94A3B8; line-height: 1.6; font-size: 0.92rem;">
            This project introduces an end-to-end framework combining <b>Dense Retrieval-Augmented Generation (RAG)</b>, 
            <b>Atomic Claim Decomposition</b>, and <b>Natural Language Inference (NLI)</b> to provide auditable, transparent verification verdicts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Technical Pipeline Flow
    st.subheader("🏗️ System Architecture & Workflow")
    st.markdown("""
    ```
    ┌─────────────────────────┐
    │ User PDF Document       │
    └───────────┬─────────────┘
                │ PyMuPDF (Page-level text extraction)
                ▼
    ┌─────────────────────────┐
    │ Overlapping Chunker     │
    └───────────┬─────────────┘
                │ Sentence-Transformers (all-MiniLM-L6-v2)
                ▼
    ┌─────────────────────────┐
    │ FAISS Vector Store      │ ◄────── User Question
    └───────────┬─────────────┘
                │ Top-K Passage Retrieval (Cosine/Inner Product)
                ▼
    ┌─────────────────────────┐
    │ Gemma 3 (via Ollama)    │ ────── Grounded Answer Generation
    └───────────┬─────────────┘
                │ Structured Prompting
                ▼
    ┌─────────────────────────┐
    │ Atomic Claim Extractor  │ ────── Decomposed into Independent Claims
    └───────────┬─────────────┘
                │ Premise (Evidence) vs Hypothesis (Claim)
                ▼
    ┌─────────────────────────┐
    │ RoBERTa-large-MNLI      │ ────── Computes [Entailment, Contradiction, Neutral]
    └───────────┬─────────────┘
                │ Threshold Mapping (0.50) + Optional LLM Judge
                ▼
    ┌─────────────────────────┐
    │ Verdicts & Audit Report │ ────── [SUPPORTED / CONTRADICTED / UNSUPPORTED]
    └─────────────────────────┘
    ```
    """)

    # Technical Stack Breakdown
    st.subheader("🛠️ Technology Stack Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - **Frontend & Dashboard:** Streamlit (Custom Dark SaaS Theme)
        - **PDF Processing:** PyMuPDF (`pymupdf`) with page tracking
        - **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-d)
        - **Vector Similarity Search:** FAISS (`IndexFlatIP`)
        - **Local Generative LLM:** Ollama running `gemma3:latest`
        """)

    with col2:
        st.markdown("""
        - **Claim Decomposition:** Gemma with structured JSON formatting
        - **NLI Verifier:** Hugging Face `roberta-large-mnli`
        - **Optional LLM Judge:** Gemma / Gemini Cloud API
        - **Database:** SQLite 3 (`detector.db`)
        - **Testing:** Pytest (14 automated unit tests)
        - **Evaluation Architecture:** Scikit-Learn (Precision, Recall, F1, FPR)
        """)

    # College Presentation Talking Points
    st.subheader("🎓 College Presentation & Viva Highlights")
    with st.expander("Key Theoretical Distinctions for Viva Examination", expanded=True):
        st.markdown("""
        1. **Why Natural Language Inference (NLI) instead of pure LLM prompting for verification?**
           - Generative LLMs are themselves prone to self-confirmation bias and hallucination when evaluating their own answers. 
           - A dedicated NLI model like `roberta-large-mnli` is trained specifically on the classification task of comparing a Premise and a Hypothesis, providing deterministic cross-entropy softmax probabilities.

        2. **Why distinguish between 'Contradicted' and 'Unsupported'?**
           - A contradiction implies active empirical conflict (e.g., Document states 2020; Model asserts 1995).
           - An unsupported claim signifies that the document simply lacks information on the topic (Neutral). Treating absence of evidence as proof of contradiction is a logical fallacy (Argumentum ad ignorantiam).

        3. **Why atomic claim decomposition?**
           - Complex multi-sentence answers often contain a blend of true facts and fabricated details. Evaluating the answer as a single block leads to false positives or false negatives. Granular atomic extraction enables surgical hallucination tagging.
        """)

    # Future Roadmap
    st.subheader("🚀 Future Expansion: HaluEval Benchmark Ingestion")
    st.info(
        "The evaluation module (`modules/evaluation.py`) is architecturally structured to ingest standardized "
        "datasets (such as the HaluEval benchmark suite) to calculate formal Precision, Recall, F1-score, "
        "and Confusion Matrices for academic benchmarking."
    )
