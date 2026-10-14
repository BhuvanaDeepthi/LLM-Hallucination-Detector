import streamlit as st

def load_custom_css():
    """Injects modern, polished AI SaaS CSS styling into the Streamlit application."""
    css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Overall Background and Layout */
    .stApp {
        background-color: #0B0F19;
        color: #F1F5F9;
    }

    /* Top Header Bar */
    .app-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .app-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Stat Cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 20px 22px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 8px 28px rgba(99, 102, 241, 0.15);
    }

    .stat-label {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }

    .stat-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 4px;
    }

    .stat-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* Cards */
    .feature-card {
        background: #131B2E;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    .feature-card h3 {
        color: #E2E8F0;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0;
    }

    /* Verdict Badges */
    .badge-supported {
        display: inline-flex;
        align-items: center;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .badge-contradicted {
        display: inline-flex;
        align-items: center;
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .badge-unsupported {
        display: inline-flex;
        align-items: center;
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    /* Claim Item Container */
    .claim-container {
        background: #111827;
        border-left: 4px solid #4B5563;
        border-radius: 8px 12px 12px 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        border-right: 1px solid rgba(255, 255, 255, 0.04);
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }

    .claim-container.supported {
        border-left-color: #10B981;
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.04) 0%, #111827 100%);
    }

    .claim-container.contradicted {
        border-left-color: #EF4444;
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.04) 0%, #111827 100%);
    }

    .claim-container.unsupported {
        border-left-color: #F59E0B;
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.04) 0%, #111827 100%);
    }

    .claim-text {
        font-size: 1.02rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 8px;
    }

    .evidence-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.92rem;
        color: #CBD5E1;
        line-height: 1.5;
    }

    /* Streamlit widgets styling */
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 8px 20px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #2563EB 100%);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45);
        transform: translateY(-1px);
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #1E293B !important;
        border-radius: 8px !important;
        color: #F1F5F9 !important;
        font-weight: 600 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0A0E1A;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366F1, #8B5CF6);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
