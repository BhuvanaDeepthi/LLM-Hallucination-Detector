"""
Comprehensive Regression Test Suite for LLM Hallucination Detector
Verifies all 6 required workflow cases:
1. A question whose answer is explicitly available in the PDF.
2. A directly supported factual claim.
3. A directly contradicted factual claim.
4. An unsupported claim for which the PDF provides no relevant information.
5. A question whose answer is not available in the document.
6. A custom CSV evaluation example using its own reference evidence.
"""

import sys
import time
import os
import pandas as pd
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import config
from modules.vector_store import VectorStore
from modules.embeddings import EmbeddingManager
from modules.rag_pipeline import RAGPipeline
from modules.claim_extractor import ClaimExtractor
from modules.nli_verifier import NLIVerifier
from modules.database import save_verification_session, get_session_by_id, init_db
from modules.report_generator import ReportGenerator

def run_regression_suite():
    print("=================================================================")
    print("RUNNING COMPREHENSIVE REGRESSION WORKFLOW SUITE (6 CASES)")
    print("=================================================================\n")

    init_db()

    # Load FAISS index from the verified PDF index
    index_id = "fe0cb464dcb83a328f1133257a58b15d821e6dc1074e7d4473a32c33cdbd0ff1"
    index_path = config.INDEXES_DIR / index_id
    if not (index_path.with_suffix(".faiss")).exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    vs = VectorStore(dimension=config.EMBEDDING_DIMENSION)
    vs.load(index_path)
    doc_name = "BI AssignmentII.pdf"
    print(f"[Init] Loaded FAISS vector index for '{doc_name}' ({vs.index.ntotal} passages).\n")

    emb_mgr = EmbeddingManager()
    rag = RAGPipeline()
    extractor = ClaimExtractor()
    verifier = NLIVerifier()

    results_summary = []

    # =========================================================================
    # Case 1: Question whose answer is explicitly available in the PDF
    # =========================================================================
    print("-----------------------------------------------------------------")
    print("CASE 1: Question with answer explicitly available in PDF")
    query_c1 = "What are the report layout requirements for the assignment?"
    print(f"Query: \"{query_c1}\"")
    t0 = time.perf_counter()

    is_q1 = ClaimExtractor.is_question(query_c1)
    assert is_q1 is True, "Case 1 must be classified as a question"

    q_vec = emb_mgr.encode(query_c1)
    retrieved_c1 = vs.search(q_vec, query_text=query_c1, top_k=3)
    t_ret1 = time.perf_counter() - t0

    t_gen_0 = time.perf_counter()
    ans_c1, model_c1, _ = rag.generate_answer(query_c1, retrieved_c1, is_query_question=True)
    t_gen1 = time.perf_counter() - t_gen_0

    t_ext_0 = time.perf_counter()
    claims_c1 = extractor.extract_claims(ans_c1, user_query=query_c1, is_query_question=True)
    t_ext1 = time.perf_counter() - t_ext_0

    t_nli_0 = time.perf_counter()
    verified_c1 = verifier.verify_all_claims(claims_c1, retrieved_c1)
    t_nli1 = time.perf_counter() - t_nli_0
    t_tot1 = time.perf_counter() - t0

    print(f"  Classification: Question ({is_q1})")
    print(f"  Generated Answer: \"{ans_c1}\"")
    print(f"  Extracted Claims: {[c['claim_text'] for c in verified_c1]}")
    assert len(verified_c1) > 0, "Should have extracted claims from answer"
    top_v1 = verified_c1[0]
    print(f"  Verdict: {top_v1['verdict']} ({top_v1['nli_relation']})")
    print(f"  Evidence Cited: Page {top_v1['evidence_page']} | Snippet: \"{top_v1['evidence_excerpt'][:100]}...\"")
    print(f"  Timings: Retrieval={t_ret1:.3f}s, Gen={t_gen1:.3f}s, Ext={t_ext1:.3f}s, NLI={t_nli1:.3f}s, Total={t_tot1:.3f}s")

    assert top_v1["verdict"] == config.VERDICT_SUPPORTED, f"Expected Supported, got {top_v1['verdict']}"
    assert top_v1["evidence_page"] == 2, f"Expected Page 2, got {top_v1['evidence_page']}"
    assert "Overview and Detailed Analysis" in top_v1["evidence_excerpt"] or "two-page" in top_v1["evidence_excerpt"]
    print("  >>> CASE 1 PASSED!\n")

    results_summary.append({
        "case": "Case 1 (Question with answer in PDF)",
        "input": query_c1,
        "input_type": "Question",
        "verdict": top_v1["verdict"],
        "evidence_page": top_v1["evidence_page"],
        "provenance": f"{doc_name} (Page {top_v1['evidence_page']})",
        "latency_sec": round(t_tot1, 3)
    })

    # =========================================================================
    # Case 2: Directly supported factual claim
    # =========================================================================
    print("-----------------------------------------------------------------")
    print("CASE 2: Directly supported factual claim")
    query_c2 = "The report requires an Overview and Detailed Analysis two-page layout."
    print(f"Query: \"{query_c2}\"")
    t0 = time.perf_counter()

    is_q2 = ClaimExtractor.is_question(query_c2)
    assert is_q2 is False, "Case 2 must be classified as a factual statement"

    q_vec = emb_mgr.encode(query_c2)
    retrieved_c2 = vs.search(q_vec, query_text=query_c2, top_k=3)
    t_ret2 = time.perf_counter() - t0

    t_gen_0 = time.perf_counter()
    ans_c2, model_c2, _ = rag.generate_answer(query_c2, retrieved_c2, is_query_question=False)
    t_gen2 = time.perf_counter() - t_gen_0

    t_ext_0 = time.perf_counter()
    claims_c2 = extractor.extract_claims(ans_c2, user_query=query_c2, is_query_question=False)
    t_ext2 = time.perf_counter() - t_ext_0

    t_nli_0 = time.perf_counter()
    verified_c2 = verifier.verify_all_claims(claims_c2, retrieved_c2)
    t_nli2 = time.perf_counter() - t_nli_0
    t_tot2 = time.perf_counter() - t0

    print(f"  Classification: Factual Statement ({not is_q2})")
    print(f"  Verification Explanation: \"{ans_c2}\"")
    top_v2 = verified_c2[0]
    print(f"  Claim: \"{top_v2['claim_text']}\"")
    print(f"  Verdict: {top_v2['verdict']} ({top_v2['nli_relation']}) | E={top_v2['entailment_score']}")
    print(f"  Evidence Cited: Page {top_v2['evidence_page']} | Snippet: \"{top_v2['evidence_excerpt'][:100]}...\"")
    print(f"  Timings: Retrieval={t_ret2:.3f}s, Gen={t_gen2:.3f}s, Ext={t_ext2:.3f}s, NLI={t_nli2:.3f}s, Total={t_tot2:.3f}s")

    assert top_v2["verdict"] == config.VERDICT_SUPPORTED, f"Expected Supported, got {top_v2['verdict']}"
    assert top_v2["evidence_page"] == 2
    assert "Overview and Detailed Analysis" in top_v2["evidence_excerpt"] or "two-page" in top_v2["evidence_excerpt"]
    print("  >>> CASE 2 PASSED!\n")

    results_summary.append({
        "case": "Case 2 (Directly supported claim)",
        "input": query_c2,
        "input_type": "Factual Statement",
        "verdict": top_v2["verdict"],
        "evidence_page": top_v2["evidence_page"],
        "provenance": f"{doc_name} (Page {top_v2['evidence_page']})",
        "latency_sec": round(t_tot2, 3)
    })

    # =========================================================================
    # Case 3: Directly contradicted factual claim
    # =========================================================================
    print("-----------------------------------------------------------------")
    print("CASE 3: Directly contradicted factual claim")
    query_c3 = "The assignment requires DAX measures for every scenario."
    print(f"Query: \"{query_c3}\"")
    t0 = time.perf_counter()

    is_q3 = ClaimExtractor.is_question(query_c3)
    assert is_q3 is False, "Case 3 must be classified as a factual statement"

    q_vec = emb_mgr.encode(query_c3)
    retrieved_c3 = vs.search(q_vec, query_text=query_c3, top_k=3)
    t_ret3 = time.perf_counter() - t0

    t_gen_0 = time.perf_counter()
    ans_c3, model_c3, _ = rag.generate_answer(query_c3, retrieved_c3, is_query_question=False)
    t_gen3 = time.perf_counter() - t_gen_0

    t_ext_0 = time.perf_counter()
    claims_c3 = extractor.extract_claims(ans_c3, user_query=query_c3, is_query_question=False)
    t_ext3 = time.perf_counter() - t_ext_0

    t_nli_0 = time.perf_counter()
    verified_c3 = verifier.verify_all_claims(claims_c3, retrieved_c3)
    t_nli3 = time.perf_counter() - t_nli_0
    t_tot3 = time.perf_counter() - t0

    print(f"  Classification: Factual Statement ({not is_q3})")
    print(f"  Verification Explanation: \"{ans_c3}\"")
    top_v3 = verified_c3[0]
    print(f"  Claim: \"{top_v3['claim_text']}\"")
    print(f"  Verdict: {top_v3['verdict']} ({top_v3['nli_relation']}) | C={top_v3['contradiction_score']}")
    print(f"  Evidence Cited: Page {top_v3['evidence_page']} | Snippet: \"{top_v3['evidence_excerpt'][:100]}...\"")
    print(f"  Timings: Retrieval={t_ret3:.3f}s, Gen={t_gen3:.3f}s, Ext={t_ext3:.3f}s, NLI={t_nli3:.3f}s, Total={t_tot3:.3f}s")

    assert top_v3["verdict"] == config.VERDICT_CONTRADICTED, f"Expected Contradicted, got {top_v3['verdict']}"
    assert top_v3["evidence_page"] == 2
    assert "DAX is not required" in top_v3["evidence_excerpt"]
    print("  >>> CASE 3 PASSED!\n")

    results_summary.append({
        "case": "Case 3 (Directly contradicted claim)",
        "input": query_c3,
        "input_type": "Factual Statement",
        "verdict": top_v3["verdict"],
        "evidence_page": top_v3["evidence_page"],
        "provenance": f"{doc_name} (Page {top_v3['evidence_page']})",
        "latency_sec": round(t_tot3, 3)
    })

    # =========================================================================
    # Case 4: Unsupported claim for which the PDF provides no relevant info
    # =========================================================================
    print("-----------------------------------------------------------------")
    print("CASE 4: Unsupported claim (unmentioned topic in PDF)")
    query_c4 = "The database uses PostgreSQL version 15 with connection pooling."
    print(f"Query: \"{query_c4}\"")
    t0 = time.perf_counter()

    is_q4 = ClaimExtractor.is_question(query_c4)
    assert is_q4 is False, "Case 4 must be classified as a factual statement"

    q_vec = emb_mgr.encode(query_c4)
    retrieved_c4 = vs.search(q_vec, query_text=query_c4, top_k=3)
    t_ret4 = time.perf_counter() - t0

    t_gen_0 = time.perf_counter()
    ans_c4, model_c4, _ = rag.generate_answer(query_c4, retrieved_c4, is_query_question=False)
    t_gen4 = time.perf_counter() - t_gen_0

    t_ext_0 = time.perf_counter()
    claims_c4 = extractor.extract_claims(ans_c4, user_query=query_c4, is_query_question=False)
    t_ext4 = time.perf_counter() - t_ext_0

    t_nli_0 = time.perf_counter()
    verified_c4 = verifier.verify_all_claims(claims_c4, retrieved_c4)
    t_nli4 = time.perf_counter() - t_nli_0
    t_tot4 = time.perf_counter() - t0

    print(f"  Classification: Factual Statement ({not is_q4})")
    print(f"  Verification Explanation: \"{ans_c4}\"")
    top_v4 = verified_c4[0]
    print(f"  Claim: \"{top_v4['claim_text']}\"")
    print(f"  Verdict: {top_v4['verdict']} ({top_v4['nli_relation']})")
    print(f"  Evidence Status: {top_v4['evidence_type']} | Page: {top_v4['evidence_page']}")
    print(f"  Evidence Text: \"{top_v4['evidence_excerpt']}\"")
    print(f"  Timings: Retrieval={t_ret4:.3f}s, Gen={t_gen4:.3f}s, Ext={t_ext4:.3f}s, NLI={t_nli4:.3f}s, Total={t_tot4:.3f}s")

    assert top_v4["verdict"] == config.VERDICT_UNSUPPORTED, f"Expected Unsupported, got {top_v4['verdict']}"
    assert top_v4["evidence_page"] is None, "Unmentioned claim must NOT cite a page number"
    assert "No relevant evidence" in top_v4["evidence_excerpt"] or "does not contain" in top_v4["evidence_excerpt"]
    print("  >>> CASE 4 PASSED!\n")

    results_summary.append({
        "case": "Case 4 (Unsupported unmentioned claim)",
        "input": query_c4,
        "input_type": "Factual Statement",
        "verdict": top_v4["verdict"],
        "evidence_page": "None (Unmentioned)",
        "provenance": f"{doc_name} (No citation fabricated)",
        "latency_sec": round(t_tot4, 3)
    })

    # =========================================================================
    # Case 5: Question whose answer is not available in the document
    # =========================================================================
    print("-----------------------------------------------------------------")
    print("CASE 5: Question whose answer is not available in document")
    query_c5 = "What is the recommended RAM configuration for running PostgreSQL 15?"
    print(f"Query: \"{query_c5}\"")
    t0 = time.perf_counter()

    is_q5 = ClaimExtractor.is_question(query_c5)
    assert is_q5 is True, "Case 5 must be classified as a question"

    q_vec = emb_mgr.encode(query_c5)
    retrieved_c5 = vs.search(q_vec, query_text=query_c5, top_k=3)
    t_ret5 = time.perf_counter() - t0

    t_gen_0 = time.perf_counter()
    ans_c5, model_c5, _ = rag.generate_answer(query_c5, retrieved_c5, is_query_question=True)
    t_gen5 = time.perf_counter() - t_gen_0

    t_ext_0 = time.perf_counter()
    claims_c5 = extractor.extract_claims(ans_c5, user_query=query_c5, is_query_question=True)
    t_ext5 = time.perf_counter() - t_ext_0

    t_nli_0 = time.perf_counter()
    verified_c5 = verifier.verify_all_claims(claims_c5, retrieved_c5)
    t_nli5 = time.perf_counter() - t_nli_0
    t_tot5 = time.perf_counter() - t0

    print(f"  Classification: Question ({is_q5})")
    print(f"  Generated Answer: \"{ans_c5}\"")
    top_v5 = verified_c5[0]
    print(f"  Claim / Refusal: \"{top_v5['claim_text']}\"")
    print(f"  Verdict: {top_v5['verdict']} ({top_v5['nli_relation']}) | Method: {top_v5['verification_method']}")
    print(f"  Evidence Status: {top_v5['evidence_type']}")
    print(f"  Timings: Retrieval={t_ret5:.3f}s, Gen={t_gen5:.3f}s, Ext={t_ext5:.3f}s, NLI={t_nli5:.3f}s, Total={t_tot5:.3f}s")

    assert top_v5["verdict"] == config.VERDICT_UNSUPPORTED, f"Expected Unsupported, got {top_v5['verdict']}"
    assert "does not contain sufficient information" in ans_c5.lower() or "not enough information" in ans_c5.lower()
    print("  >>> CASE 5 PASSED!\n")

    results_summary.append({
        "case": "Case 5 (Question not in document)",
        "input": query_c5,
        "input_type": "Question",
        "verdict": top_v5["verdict"],
        "evidence_page": "None (Insufficient Info)",
        "provenance": f"{doc_name} (Insufficient Info Refusal)",
        "latency_sec": round(t_tot5, 3)
    })

    # =========================================================================
    # Case 6: Custom CSV evaluation example using its own reference evidence
    # =========================================================================
    print("-----------------------------------------------------------------")
    print("CASE 6: Custom CSV evaluation example with its own reference evidence")
    csv_file = "hallucination_detector_custom_eval.csv"
    assert os.path.exists(csv_file), f"CSV file not found: {csv_file}"
    df = pd.read_csv(csv_file)
    row_c6 = df.iloc[0]  # SYN001-S: "The final report must be submitted as a PDF."

    ex_id = row_c6["example_id"]
    domain = row_c6["domain"]
    ref_evidence = row_c6["evidence_passage"]
    claim_c6 = row_c6["claim"]
    expected_v = row_c6["expected_verdict"].capitalize()

    print(f"Example ID: {ex_id} | Domain: {domain}")
    print(f"Claim: \"{claim_c6}\"")
    print(f"Reference Evidence Passage: \"{ref_evidence}\"")
    t0 = time.perf_counter()

    # Pass strictly the row's reference evidence passage without querying uploaded PDF
    csv_passage = [
        {
            "text": ref_evidence,
            "page_number": None,
            "example_id": ex_id,
            "dataset_name": "Custom Controlled Evaluation",
            "similarity_score": 1.0
        }
    ]

    verified_c6 = verifier.verify_all_claims([claim_c6], csv_passage)
    t_tot6 = time.perf_counter() - t0
    top_v6 = verified_c6[0]

    print(f"  Prediction: {top_v6['verdict']} ({top_v6['nli_relation']}) | Expected: {expected_v}")
    print(f"  Provenance: Dataset '{top_v6['dataset_name']}' | Example ID '{top_v6['example_id']}'")
    print(f"  Cited Page: {top_v6['evidence_page']} (None as expected for CSV)")
    print(f"  Evidence Excerpt: \"{top_v6['evidence_excerpt']}\"")
    print(f"  Latency: {t_tot6:.4f}s")

    assert top_v6["verdict"] == expected_v, f"Expected {expected_v}, got {top_v6['verdict']}"
    assert top_v6["example_id"] == ex_id, f"Expected {ex_id}, got {top_v6['example_id']}"
    assert top_v6["evidence_page"] is None, "CSV evaluation must NOT invent a page number"
    assert top_v6["source_filename"] is None, "CSV evaluation must NOT invent a PDF filename"
    print("  >>> CASE 6 PASSED!\n")

    results_summary.append({
        "case": "Case 6 (Custom CSV Evaluation Example)",
        "input": claim_c6,
        "input_type": "CSV Benchmark Pair",
        "verdict": top_v6["verdict"],
        "evidence_page": f"None (ID: {ex_id})",
        "provenance": f"Dataset: Custom Controlled Evaluation (Example: {ex_id})",
        "latency_sec": round(t_tot6, 3)
    })

    # =========================================================================
    # Additional Verification: SQLite Database & Report Export
    # =========================================================================
    print("-----------------------------------------------------------------")
    print("CHECKING DATABASE AUDIT PERSISTENCE & REPORT EXPORT...")
    sess_id = save_verification_session(
        document_name=doc_name,
        question=query_c1,
        answer=ans_c1,
        model_used=model_c1,
        claims=verified_c1,
        doc_id=1
    )
    assert sess_id > 0, "Session ID must be positive"
    sess_retrieved = get_session_by_id(sess_id)
    assert sess_retrieved is not None, "Saved session must be retrievable from SQLite"
    assert len(sess_retrieved["claims"]) == len(verified_c1)
    print(f"  SQLite Audit Save: Session #{sess_id} verified with {len(sess_retrieved['claims'])} claim(s).")

    report_md = ReportGenerator.generate_markdown_report({
        "document_name": doc_name,
        "question": query_c1,
        "answer": ans_c1,
        "model_used": model_c1,
        "claims": verified_c1
    })
    assert "# AI Hallucination Verification Audit Report" in report_md
    assert "### Claim #1" in report_md
    # Verify no raw HTML span strings in generated report or UI formatting
    assert "<span style=" not in report_md
    print(f"  Markdown Report Export: Successfully generated ({len(report_md)} characters). No raw HTML detected.")

    print("\n=================================================================")
    print("ALL 6 REGRESSION CASES & WORKFLOW CHECKS PASSED PERFECTLY!")
    print("=================================================================\n")

    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_regression_suite()
