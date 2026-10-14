import time
import os
from pathlib import Path
from modules.pdf_processor import PDFProcessor
from modules.text_chunker import TextChunker
from modules.embeddings import EmbeddingManager
from modules.vector_store import VectorStore
from modules.rag_pipeline import RAGPipeline
from modules.claim_extractor import ClaimExtractor
from modules.nli_verifier import NLIVerifier
import config

def run_benchmark():
    pdf_path = r"c:\Users\reeth\Downloads\BI AssignmentII.pdf"
    query = "The assignment requires DAX measures for every scenario."
    filename = os.path.basename(pdf_path)

    print("=================================================================")
    print("LIVE PIPELINE BENCHMARK: 'BI AssignmentII.pdf'")
    print(f"Query: \"{query}\"")
    print("=================================================================\n")

    t_overall_start = time.perf_counter()

    # Stage 1: PDF Extraction
    t0 = time.perf_counter()
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    pages, metadata = PDFProcessor.extract_text(pdf_bytes, filename)
    t_pdf_extract = time.perf_counter() - t0
    print(f"[1] PDF Extraction: {t_pdf_extract:.4f}s ({len(pages)} pages, {metadata['total_words']} words)")

    # Stage 2: Chunking & Embedding
    t0 = time.perf_counter()
    chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = chunker.chunk_pages(pages, filename)
    
    emb_mgr = EmbeddingManager()
    embeddings = emb_mgr.encode([c["text"] for c in chunks])
    
    vector_store = VectorStore(dimension=config.EMBEDDING_DIMENSION)
    vector_store.clear()
    vector_store.add_chunks(chunks, embeddings)
    t_chunk_embed = time.perf_counter() - t0
    print(f"[2] Chunking & Embedding + FAISS Indexing: {t_chunk_embed:.4f}s ({len(chunks)} chunks)")

    # Save to disk cache for index reuse verification
    file_hash = metadata["file_hash"]
    idx_prefix = config.INDEXES_DIR / file_hash
    vector_store.save(idx_prefix)
    print(f"    Saved FAISS index to disk: {idx_prefix}.faiss")

    # Stage 3: FAISS Retrieval
    t0 = time.perf_counter()
    query_vector = emb_mgr.encode(query)
    retrieved = vector_store.search(
        query_vector,
        query_text=query,
        top_k=config.TOP_K_RETRIEVAL
    )
    t_retrieval = time.perf_counter() - t0
    print(f"\n[3] FAISS Hybrid Retrieval: {t_retrieval:.4f}s (Top {len(retrieved)} passages)")
    for i, p in enumerate(retrieved, 1):
        print(f"    Passage {i} (Page {p['page_number']}, Score {p['similarity_score']}):")
        preview = p['text'].replace('\n', ' ')[:140]
        print(f"      \"{preview}...\"")

    # Stage 4: LLM Answer Generation
    t0 = time.perf_counter()
    rag = RAGPipeline()
    answer, model_used, prompt_used = rag.generate_answer(query, retrieved)
    t_generation = time.perf_counter() - t0
    print(f"\n[4] LLM Answer Generation ({model_used}): {t_generation:.4f}s")
    print(f"    Answer: \"{answer}\"")

    # Stage 5: Claim Extraction
    t0 = time.perf_counter()
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(answer, user_query=query)
    t_extraction = time.perf_counter() - t0
    print(f"\n[5] Claim Extraction: {t_extraction:.4f}s ({len(claims)} claim(s) extracted)")
    for i, c in enumerate(claims, 1):
        print(f"    Claim {i} [{c['source']}]: \"{c['claim_text']}\"")

    # Stage 6: NLI Inference
    t0 = time.perf_counter()
    verifier = NLIVerifier()
    verified_results = verifier.verify_all_claims(claims, retrieved)
    t_nli = time.perf_counter() - t0
    print(f"\n[6] RoBERTa-large-MNLI Verification: {t_nli:.4f}s")
    for i, v in enumerate(verified_results, 1):
        print(f"    Claim {i}: \"{v['claim_text']}\"")
        print(f"      -> Verdict: {v['verdict'].upper()} ({v['nli_relation']})")
        print(f"      -> Entailment: {v['entailment_score']}, Contradiction: {v['contradiction_score']}, Neutral: {v['neutral_score']}")
        print(f"      -> Evidence Cited: Page {v['evidence_page']}")
        print(f"      -> Evidence Snippet: \"{v['evidence_text'].replace(chr(10), ' ')[:120]}...\"")

    t_total_query_pipeline = t_retrieval + t_generation + t_extraction + t_nli
    t_total_including_indexing = time.perf_counter() - t_overall_start

    print("\n=================================================================")
    print("STAGE LATENCY BREAKDOWN (BENCHMARK RESULTS):")
    print("=================================================================")
    print(f"1. PDF Extraction:          {t_pdf_extract:>8.3f}s")
    print(f"2. Chunking & Embedding:    {t_chunk_embed:>8.3f}s")
    print(f"3. FAISS Hybrid Retrieval:  {t_retrieval:>8.3f}s")
    print(f"4. LLM Answer Generation:   {t_generation:>8.3f}s")
    print(f"5. Claim Extraction:        {t_extraction:>8.3f}s")
    print(f"6. NLI Inference:           {t_nli:>8.3f}s")
    print("-----------------------------------------------------------------")
    print(f"Query Pipeline Time (3..6): {t_total_query_pipeline:>8.3f}s")
    print(f"Total Full Processing:      {t_total_including_indexing:>8.3f}s")
    print("=================================================================")

if __name__ == "__main__":
    run_benchmark()
