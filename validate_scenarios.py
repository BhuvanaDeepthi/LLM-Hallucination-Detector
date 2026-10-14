import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
import config
from modules.embeddings import EmbeddingManager
from modules.vector_store import VectorStore
from modules.rag_pipeline import RAGPipeline
from modules.claim_extractor import ClaimExtractor
from modules.nli_verifier import NLIVerifier

print("=================================================================")
print("COMPREHENSIVE END-TO-END PIPELINE VALIDATION")
print("=================================================================\n")

# Load index and initialize models once (reused across requests)
t_init_0 = time.perf_counter()
vs = VectorStore(dimension=config.EMBEDDING_DIMENSION)
vs.load(config.INDEXES_DIR / "fe0cb464dcb83a328f1133257a58b15d821e6dc1074e7d4473a32c33cdbd0ff1")
emb_mgr = EmbeddingManager()
rag = RAGPipeline()
extractor = ClaimExtractor()
verifier = NLIVerifier()
t_init = time.perf_counter() - t_init_0
print(f"[Init] Loaded FAISS index ({vs.index.ntotal} chunks) & models in {t_init:.2f}s\n")

test_cases = [
    {
        "id": "Case 1 (Report 5 Pages - Conflict)",
        "query": "The report requires at least five pages.",
        "expected_verdict": config.VERDICT_CONTRADICTED,
        "expected_page": 2
    },
    {
        "id": "Case 2 (DAX Required - Conflict)",
        "query": "The assignment requires DAX measures for every scenario.",
        "expected_verdict": config.VERDICT_CONTRADICTED,
        "expected_page": 2
    },
    {
        "id": "Case 3 (PostgreSQL - Unsupported/Unmentioned)",
        "query": "The database uses PostgreSQL version 15 with connection pooling.",
        "expected_verdict": config.VERDICT_UNSUPPORTED,
        "expected_page": None
    },
    {
        "id": "Case 4 (Two-page report - Supported)",
        "query": "The report requires an Overview and Detailed Analysis two-page layout.",
        "expected_verdict": config.VERDICT_SUPPORTED,
        "expected_page": 2
    }
]

for tc in test_cases:
    print(f"-----------------------------------------------------------------")
    print(f"RUNNING: {tc['id']}")
    print(f"Query: \"{tc['query']}\"")
    print(f"-----------------------------------------------------------------")

    t_case_start = time.perf_counter()

    # Stage 1: Retrieval
    t0 = time.perf_counter()
    q_vec = emb_mgr.encode(tc["query"])
    retrieved = vs.search(q_vec, query_text=tc["query"], top_k=3)
    t_retrieval = time.perf_counter() - t0

    # Stage 2: RAG Generation
    t0 = time.perf_counter()
    answer, model_used, _ = rag.generate_answer(tc["query"], retrieved)
    t_gen = time.perf_counter() - t0

    # Stage 3: Claim Extraction
    t0 = time.perf_counter()
    claims = extractor.extract_claims(answer, user_query=tc["query"])
    t_extract = time.perf_counter() - t0

    # Stage 4: NLI Verification
    t0 = time.perf_counter()
    verified = verifier.verify_all_claims(claims, retrieved)
    t_nli = time.perf_counter() - t0

    t_total = time.perf_counter() - t_case_start

    print(f"Timing Breakdown:")
    print(f"  Retrieval:       {t_retrieval:.4f}s (Top {len(retrieved)} passages, Pages {[p.get('page_number') for p in retrieved]})")
    print(f"  Answer Gen:      {t_gen:.4f}s ({model_used})")
    print(f"  Extraction:      {t_extract:.4f}s ({len(claims)} claim(s))")
    print(f"  NLI Verify:      {t_nli:.4f}s")
    print(f"  Total Latency:   {t_total:.4f}s")

    print(f"Answer: \"{answer}\"")
    print(f"Verification Results:")
    for i, v in enumerate(verified, 1):
        print(f"  [{v.get('source', 'unknown')}] \"{v['claim_text']}\"")
        print(f"    -> Verdict: {v['verdict'].upper()} ({v['nli_relation']}) | E={v['entailment_score']}, C={v['contradiction_score']}, N={v['neutral_score']}")
        if v.get('evidence_page'):
            print(f"    -> Cited: Page {v['evidence_page']}")
        snippet = v.get('evidence_text', '').replace('\n', ' ')[:100]
        if snippet:
            print(f"    -> Evidence: \"{snippet}...\"")

    # Validate primary user claim verdict
    user_claim_res = verified[0] if verified else {}
    actual_verdict = user_claim_res.get("verdict")
    assert actual_verdict == tc["expected_verdict"], f"Expected {tc['expected_verdict']}, got {actual_verdict}"
    print(f"  >>> PASSED: Verdict matches expected {tc['expected_verdict']}\n")

print("=================================================================")
print("ALL VALIDATION SCENARIOS COMPLETED SUCCESSFULLY!")
print("=================================================================")
