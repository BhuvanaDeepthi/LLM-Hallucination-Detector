import os
import pymupdf
from modules.pdf_processor import PDFProcessor
from modules.text_chunker import TextChunker
from modules.embeddings import EmbeddingManager
from modules.vector_store import VectorStore
from modules.rag_pipeline import RAGPipeline
from modules.claim_extractor import ClaimExtractor
from modules.nli_verifier import NLIVerifier
from modules.database import save_document, save_verification_session, get_system_stats
from modules.report_generator import ReportGenerator

def main():
    print("=== STARTING FULL PIPELINE INTEGRATION TEST ===")
    
    # 1. Create a sample PDF
    doc = pymupdf.open()
    page = doc.new_page()
    sample_text = (
        "The Transformer architecture was introduced by Vaswani et al. in 2017 in the paper Attention Is All You Need. "
        "It replaced recurrent layers entirely with multi-head self-attention mechanisms. "
        "On the WMT 2014 English-to-German translation task, the big model achieved a BLEU score of 28.4. "
        "Training was conducted using eight NVIDIA P100 GPUs for 3.5 days."
    )
    page.insert_text((50, 72), sample_text)
    pdf_bytes = doc.tobytes()
    doc.close()
    print("[OK] Step 1: In-memory PDF created successfully.")

    # 2. Extract
    pages, metadata = PDFProcessor.extract_text(pdf_bytes, "transformer_eval.pdf")
    print(f"[OK] Step 2: Extracted {len(pages)} page(s) ({metadata['total_words']} words).")

    # 3. Chunk
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_pages(pages, "transformer_eval.pdf")
    print(f"[OK] Step 3: Created {len(chunks)} overlapping passage chunk(s).")

    # 4. Embed & Index
    emb_mgr = EmbeddingManager()
    embs = emb_mgr.encode([c["text"] for c in chunks])
    vs = VectorStore(dimension=384)
    vs.add_chunks(chunks, embs)
    print(f"[OK] Step 4: Built FAISS vector index with {vs.index.ntotal} passages.")

    # 5. Search
    query = "What BLEU score did the transformer achieve and which GPUs were used for training?"
    q_emb = emb_mgr.encode(query)
    retrieved = vs.search(q_emb, top_k=2)
    print(f"[OK] Step 5: Retrieved {len(retrieved)} relevant passage(s).")

    # 6. RAG Answer
    rag = RAGPipeline()
    answer, model_used, prompt = rag.generate_answer(query, retrieved)
    print(f"[OK] Step 6: Generated answer via {model_used}:")
    print(f"    --> \"{answer}\"")

    # 7. Claim Extraction
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(answer)
    print(f"[OK] Step 7: Extracted {len(claims)} atomic claim(s):")
    for i, c in enumerate(claims, 1):
        print(f"    Claim {i}: \"{c}\"")

    # 8. NLI Verification
    verifier = NLIVerifier()
    verified = verifier.verify_all_claims(claims, retrieved)
    print(f"[OK] Step 8: Verified {len(verified)} claim(s) using RoBERTa-large-MNLI:")
    for v in verified:
        print(f"    [{v['verdict'].upper()}] \"{v['claim_text']}\" (Entailment: {v['entailment_score']}, Contradiction: {v['contradiction_score']})")

    # 9. Database Save
    doc_id = save_document("transformer_eval.pdf", metadata["page_count"], len(chunks), metadata["file_hash"])
    sess_id = save_verification_session("transformer_eval.pdf", query, answer, model_used, verified, doc_id)
    stats = get_system_stats()
    print(f"[OK] Step 9: Saved session #{sess_id} to SQLite. Real Database Stats: {stats}")

    # 10. Report Generation
    report = ReportGenerator.generate_markdown_report({
        "document_name": "transformer_eval.pdf",
        "question": query,
        "answer": answer,
        "model_used": model_used,
        "claims": verified
    })
    print(f"[OK] Step 10: Generated markdown audit report ({len(report)} characters).")
    print("=== FULL END-TO-END PIPELINE VERIFIED SUCCESSFULLY! ===")

if __name__ == "__main__":
    main()
