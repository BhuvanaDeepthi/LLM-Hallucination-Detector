import streamlit as st
import pandas as pd
import json
import io
from typing import List, Dict, Any
from modules.nli_verifier import NLIVerifier
import config

SAMPLE_CSV_DATA = """claim,evidence,ground_truth
"The model uses a transformer architecture.","The proposed system is based on the multi-head self-attention transformer architecture.",Supported
"The experiment was conducted in 1995.","All model experiments and training runs took place between June and August 2023.",Contradicted
"The software supports real-time streaming audio.","The application provides document processing and vector database indexing.",Unsupported
"The dataset contains 50000 images.","The training benchmark consists of exactly 50,000 labeled high-resolution images.",Supported
"""

def render_evaluation_page():
    """Renders the Dataset Benchmark & Evaluation interface for testing verification accuracy."""

    st.markdown("""
    <div class="app-header">
        <div class="app-title">Dataset Benchmark & Evaluation</div>
        <div class="app-subtitle">
            Upload an evaluation dataset (CSV) to benchmark the detector's Precision, Recall, F1-Score, and Confusion Matrix.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Explanation Card
    st.markdown("""
    <div class="feature-card">
        <h3>🧪 Benchmark Your Custom Evaluation Dataset</h3>
        <p style="color: #94A3B8; font-size: 0.92rem; line-height: 1.6; margin-bottom: 8px;">
            You can test the detector against standardized benchmark datasets (like HaluEval) or your own custom college test dataset.
            The evaluation engine tests each claim against its paired evidence passage and compares the predicted verdict against your reference ground-truth labels.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Download Template Button
    col_dl, col_space = st.columns([1.5, 2.5])
    with col_dl:
        st.download_button(
            label="📥 Download Sample Evaluation Dataset (CSV)",
            data=SAMPLE_CSV_DATA,
            file_name="evaluation_sample.csv",
            mime="text/csv",
            help="Download a pre-formatted CSV template with example claims and evidence."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # File Uploader
    uploaded_eval_file = st.file_uploader(
        "Upload your Evaluation Dataset (CSV format):",
        type=["csv"],
        key="eval_dataset_uploader"
    )

    if uploaded_eval_file is not None:
        try:
            df = pd.read_csv(uploaded_eval_file)
            st.markdown(f"**Loaded Dataset:** `{uploaded_eval_file.name}` ({len(df)} test rows)")
            
            # Preview loaded data
            with st.expander("Preview Uploaded Dataset (First 5 rows)", expanded=False):
                st.dataframe(df.head(5), use_container_width=True)

            # Detect required columns
            claim_col = next((c for c in df.columns if c.lower() in ["claim", "statement", "hypothesis", "question"]), None)
            evidence_col = next((c for c in df.columns if c.lower() in ["evidence", "context", "passage", "premise", "source"]), None)
            label_col = next((c for c in df.columns if c.lower() in ["ground_truth", "label", "target", "verdict", "hallucination"]), None)

            if not claim_col or not evidence_col or not label_col:
                st.error(
                    f"Dataset must contain columns for Claim, Evidence, and Ground Truth. "
                    f"Found columns: {list(df.columns)}. Please rename your columns to `claim`, `evidence`, and `ground_truth`."
                )
                return

            if st.button("🚀 Run Dataset Benchmark Evaluation", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Loading NLI verifier and running batched evaluation...")

                verifier = NLIVerifier()
                results = []
                y_true = []
                y_pred = []

                total_rows = len(df)
                pairs = []
                for _, row in df.iterrows():
                    premise = str(row[evidence_col])
                    hypothesis = str(row[claim_col])
                    pairs.append((premise, hypothesis))

                # Fast batched prediction
                status_text.text(f"Classifying {total_rows} benchmark pairs with RoBERTa-large-MNLI...")
                progress_bar.progress(40)
                scores = verifier.predict_batch(pairs, batch_size=16)

                progress_bar.progress(80)
                status_text.text("Computing Precision, Recall, and Confusion Matrix...")

                for idx, (_, row) in enumerate(df.iterrows()):
                    sc = scores[idx]
                    n = sc["neutral"]
                    if c >= config.CONTRADICTION_THRESHOLD and c > e and c > n:
                        pred_verdict = config.VERDICT_CONTRADICTED
                    elif e >= config.ENTAILMENT_THRESHOLD and e > c and e > n:
                        pred_verdict = config.VERDICT_SUPPORTED
                    else:
                        pred_verdict = config.VERDICT_UNSUPPORTED

                    true_raw = str(row[label_col]).strip()
                    # Normalize true label
                    if true_raw.lower() in ["supported", "true", "0", "faithful", "entailment"]:
                        true_verdict = config.VERDICT_SUPPORTED
                    elif true_raw.lower() in ["contradicted", "false", "1", "hallucinated", "contradiction"]:
                        true_verdict = config.VERDICT_CONTRADICTED
                    else:
                        true_verdict = config.VERDICT_UNSUPPORTED

                    is_correct = (pred_verdict.lower() == true_verdict.lower())

                    # Binary mapping for standard classification metrics:
                    # 1 = Hallucinated (Contradicted/Unsupported), 0 = Faithful (Supported)
                    y_true.append(0 if true_verdict == config.VERDICT_SUPPORTED else 1)
                    y_pred.append(0 if pred_verdict == config.VERDICT_SUPPORTED else 1)

                    rec = {}
                    if "example_id" in row:
                        rec["Example ID"] = str(row["example_id"])
                    rec["Claim"] = str(row[claim_col])
                    rec["Reference Evidence"] = str(row[evidence_col])
                    rec["Ground Truth"] = true_verdict
                    rec["Predicted"] = pred_verdict
                    rec["Match"] = "✅ Correct" if is_correct else "❌ Misclassified"
                    rec["Entailment"] = round(e, 3)
                    rec["Contradiction"] = round(c, 3)
                    rec["Neutral"] = round(n, 3)
                    rec["Dataset"] = uploaded_eval_file.name

                    results.append(rec)

                progress_bar.progress(100)
                status_text.empty()

                # Compute formal metrics using scikit-learn
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

                st.success("✅ Evaluation Benchmark Complete!")

                # Metric Cards
                st.markdown("### 📈 Overall Performance Metrics")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("Accuracy", f"{acc * 100:.1f}%")
                with m_col2:
                    st.metric("Precision", f"{prec * 100:.1f}%")
                with m_col3:
                    st.metric("Recall", f"{rec * 100:.1f}%")
                with m_col4:
                    st.metric("F1-Score", f"{f1 * 100:.1f}%")

                # Confusion Matrix Display
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🧮 Confusion Matrix Breakdown")
                cm_col1, cm_col2 = st.columns(2)
                with cm_col1:
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 16px;">
                        <div style="font-weight: 600; color: #94A3B8; margin-bottom: 8px;">Classification Counts:</div>
                        <ul style="color: #CBD5E1; line-height: 1.8; margin: 0; padding-left: 20px;">
                            <li><b>True Positives (Correctly Flagged Hallucinations):</b> {tp}</li>
                            <li><b>True Negatives (Correctly Verified Faithful Claims):</b> {tn}</li>
                            <li><b>False Positives (Faithful Claims Mistakenly Flagged):</b> {fp}</li>
                            <li><b>False Negatives (Missed Hallucinations):</b> {fn}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                with cm_col2:
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    st.metric("False Positive Rate (FPR)", f"{fpr * 100:.1f}%", help="Proportion of faithful claims falsely tagged as hallucinations.")
                    st.metric("Total Tested Samples", total_rows)

                # Results Table
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📋 Itemized Verification Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing evaluation dataset: {str(e)}")
