import datetime
from pathlib import Path
from typing import Dict, Any, List
import config

class ReportGenerator:
    """Generates structured verification reports in Markdown and Text formats."""

    @staticmethod
    def generate_markdown_report(session_data: Dict[str, Any]) -> str:
        """Constructs an audit-ready Markdown verification report."""
        timestamp = session_data.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        doc_name = session_data.get("document_name", "Unknown Document")
        question = session_data.get("question", "")
        answer = session_data.get("answer", "")
        model_used = session_data.get("model_used", "N/A")
        overall_verdict = session_data.get("overall_verdict", "N/A")
        claims: List[Dict[str, Any]] = session_data.get("claims", [])

        total = len(claims)
        supported = sum(1 for c in claims if c.get("verdict") == config.VERDICT_SUPPORTED)
        contradicted = sum(1 for c in claims if c.get("verdict") == config.VERDICT_CONTRADICTED)
        unsupported = sum(1 for c in claims if c.get("verdict") == config.VERDICT_UNSUPPORTED)

        sup_pct = (supported / total * 100) if total > 0 else 0
        con_pct = (contradicted / total * 100) if total > 0 else 0
        uns_pct = (unsupported / total * 100) if total > 0 else 0

        md = []
        md.append(f"# AI Hallucination Verification Audit Report")
        md.append(f"**Generated:** {timestamp}")
        md.append(f"**Target Document:** `{doc_name}`")
        md.append(f"**Generation Model:** `{model_used}`")
        md.append(f"**Overall System Verdict:** **{overall_verdict}**\n")
        md.append("---")

        md.append("## 1. User Query & AI Answer")
        md.append(f"**User Question:**\n> {question}\n")
        md.append(f"**Generated Answer:**\n> {answer}\n")
        md.append("---")

        md.append("## 2. Verification Executive Summary")
        md.append("| Metric | Count | Percentage |")
        md.append("| :--- | :---: | :---: |")
        md.append(f"| **Total Claims Analyzed** | {total} | 100% |")
        md.append(f"| **Supported (Entailed)** | {supported} | {sup_pct:.1f}% |")
        md.append(f"| **Contradicted (Fact Conflict)** | {contradicted} | {con_pct:.1f}% |")
        md.append(f"| **Unsupported (Insufficient Evidence)** | {unsupported} | {uns_pct:.1f}% |\n")
        md.append("---")

        md.append("## 3. Atomic Claim Verification Breakdown\n")

        for i, c in enumerate(claims, 1):
            claim_text = c.get("claim_text", "")
            verdict = c.get("verdict", config.VERDICT_UNSUPPORTED)
            nli_rel = c.get("nli_relation", "Neutral")
            e_score = c.get("entailment_score", 0.0)
            c_score = c.get("contradiction_score", 0.0)
            n_score = c.get("neutral_score", 0.0)
            evidence = c.get("evidence_text", "").strip()
            page = c.get("evidence_page")
            judge_used = c.get("judge_used", False)
            judge_note = c.get("judge_assessment")

            # Icon tag
            if verdict == config.VERDICT_SUPPORTED:
                badge = "[SUPPORTED]"
            elif verdict == config.VERDICT_CONTRADICTED:
                badge = "[CONTRADICTED]"
            else:
                badge = "[UNSUPPORTED]"

            md.append(f"### Claim #{i}: {badge}")
            md.append(f"**Claim Text:** \"{claim_text}\"")
            md.append(f"- **Final Verdict:** `{verdict}`")

            if c.get("explanation"):
                md.append(f"- **Verification Explanation:** {c['explanation']}")

            if c.get("nli_executed", True):
                md.append(f"- **NLI Relationship:** `{nli_rel}` (Entailment: {e_score:.3f}, Contradiction: {c_score:.3f}, Neutral: {n_score:.3f})")
            else:
                md.append(f"- **Verification Method:** `{c.get('verification_method', 'Deterministic Grounding')}`")

            if page:
                md.append(f"- **Source Reference:** Page {page}")
            elif c.get("example_id"):
                md.append(f"- **Evaluation Example ID:** `{c['example_id']}`")

            ev_type = c.get("evidence_type", "Evidence")
            if evidence and evidence != "No relevant evidence found in the document to support or contradict this claim.":
                md.append(f"- **{ev_type}:**\n  > \"{evidence}\"")
                if c.get("full_passage") and c.get("full_passage") != evidence:
                    md.append(f"- **Full Passage Context:**\n  > \"{c.get('full_passage')}\"")
            else:
                md.append(f"- **Evidence Status:** No relevant evidence found in the document to support or contradict this claim.")

            if judge_used and judge_note:
                md.append(f"- **LLM-Judge Evaluation:** {judge_note}")

            md.append("")

        md.append("---")
        md.append("## 4. Verification Disclaimer")
        md.append(
            "_Automated natural language inference models and semantic retrieval provide probabilistic verification "
            "based on retrieved passages. Users and evaluators should always review underlying citations and source pages._"
        )

        return "\n".join(md)

    @staticmethod
    def generate_plain_text_report(session_data: Dict[str, Any]) -> str:
        """Generates a plain-text version for quick export."""
        md = ReportGenerator.generate_markdown_report(session_data)
        # Convert simple markdown headers and quotes to clean text
        lines = []
        for line in md.split("\n"):
            clean = line.replace("#", "").replace("**", "").replace("`", "").replace(">", "  ")
            lines.append(clean)
        return "\n".join(lines)
