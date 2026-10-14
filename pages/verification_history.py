import streamlit as st
import time
from modules.database import get_all_sessions, get_session_by_id, delete_session, clear_all_history
from modules.report_generator import ReportGenerator
import config

def render_verification_history():
    """Renders the historical audit sessions review page with full claim inspection and deletion controls."""

    st.markdown("""
    <div class="app-header">
        <div class="app-title">Verification History & Audit Log</div>
        <div class="app-subtitle">
            Review past verification sessions, examine claim breakdowns, audit retrieved evidence, and export historical reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

    sessions = get_all_sessions()

    # Clear All Button
    if sessions:
        col_hdr, col_btn = st.columns([3, 1])
        with col_btn:
            if st.button("🗑️ Clear All History", type="secondary", use_container_width=True):
                clear_all_history()
                st.success("All historical audit sessions have been deleted.")
                st.rerun()

    if not sessions:
        st.info("No verification sessions have been recorded yet. Process a PDF and run a query to populate the audit log.")
        return

    # List of Previous Sessions
    st.markdown(f"**Recorded Audit Sessions ({len(sessions)} total):**")

    for s in sessions:
        sess_id = s["id"]
        doc_name = s["document_name"]
        ts = s["timestamp"]
        total = s["total_claims"]
        supp = s["supported_claims"]
        contra = s["contradicted_claims"]
        unsupp = s["unsupported_claims"]
        overall = s["overall_verdict"]

        # Color-coded overall verdict with native Streamlit formatting
        if "Contradiction" in overall:
            badge_color = "#EF4444"
            verdict_badge = f":red[**{overall}**]"
        elif "Partially" in overall or "Unverified" in overall:
            badge_color = "#F59E0B"
            verdict_badge = f":orange[**{overall}**]"
        else:
            badge_color = "#10B981"
            verdict_badge = f":green[**{overall}**]"

        with st.expander(f"📁 Session #{sess_id} — {doc_name} ({ts}) | {overall}"):
            col_info1, col_info2 = st.columns([3, 1])
            with col_info1:
                st.markdown(f"**Input / Question:** {s['question']}")
                st.markdown(f"**Generated Output / Findings:**\n> {s['answer']}")
                st.caption(f"**Model:** `{s['model_used']}` | **Overall Verdict:** {verdict_badge}")

            with col_info2:
                st.markdown(f"• **Supported:** :green[**{supp}/{total}**]")
                st.markdown(f"• **Contradicted:** :red[**{contra}/{total}**]")
                st.markdown(f"• **Unsupported:** :orange[**{unsupp}/{total}**]")

                if st.button(f"🗑️ Delete Session", key=f"del_{sess_id}"):
                    delete_session(sess_id)
                    st.success(f"Deleted Session #{sess_id}")
                    st.rerun()

            # Retrieve session details
            session_full = get_session_by_id(sess_id)
            if session_full and session_full.get("claims"):
                st.markdown("<br>**Decomposed Claims Audit:**", unsafe_allow_html=True)
                for ci, claim in enumerate(session_full["claims"], 1):
                    verdict = claim.get("verdict", config.VERDICT_UNSUPPORTED)
                    v_color = "#10B981" if verdict == config.VERDICT_SUPPORTED else ("#EF4444" if verdict == config.VERDICT_CONTRADICTED else "#F59E0B")
                    v_badge = f":green[**{verdict.upper()}**]" if verdict == config.VERDICT_SUPPORTED else (f":red[**{verdict.upper()}**]" if verdict == config.VERDICT_CONTRADICTED else f":orange[**{verdict.upper()}**]")

                    score_details = []
                    if claim.get("nli_relation") and claim.get("nli_relation") != "None":
                        score_details.append(f"NLI: {claim.get('nli_relation')}")
                    if claim.get("entailment_score", 0) > 0 or claim.get("contradiction_score", 0) > 0:
                        score_details.append(f"Entailment: {claim.get('entailment_score', 0):.2f}")
                        score_details.append(f"Contradiction: {claim.get('contradiction_score', 0):.2f}")
                        score_details.append(f"Neutral: {claim.get('neutral_score', 0):.2f}")
                    if claim.get("evidence_page"):
                        score_details.append(f"Page: {claim.get('evidence_page')}")

                    score_line = " | ".join(score_details) if score_details else "Deterministic Verification"

                    st.markdown(f"**Claim {ci}:** \"{claim.get('claim_text')}\" — {v_badge}")
                    st.caption(score_line)

                    ev_text = (claim.get("evidence_text") or "").strip()
                    if ev_text and ev_text != "No relevant evidence found in the document to support or contradict this claim.":
                        with st.expander(f"View Evidence Excerpt for Claim {ci}"):
                            cite_label = f"Page {claim.get('evidence_page')}" if claim.get("evidence_page") else "Reference Evidence"
                            st.caption(f"Source passage ({cite_label}):")
                            st.markdown(f"> \"{ev_text}\"")

            # Export report for this session
            if session_full:
                report_md = ReportGenerator.generate_markdown_report(session_full)
                st.download_button(
                    label=f"📥 Download Audit Report for Session #{sess_id}",
                    data=report_md,
                    file_name=f"audit_session_{sess_id}.md",
                    mime="text/markdown",
                    key=f"dl_sess_{sess_id}"
                )
