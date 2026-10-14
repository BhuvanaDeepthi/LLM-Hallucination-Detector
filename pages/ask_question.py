import streamlit as st
import time
import traceback
from modules.pdf_processor import PDFProcessor, PDFProcessingError
from modules.text_chunker import TextChunker
from modules.embeddings import EmbeddingManager
from modules.vector_store import VectorStore
from modules.rag_pipeline import RAGPipeline
from modules.claim_extractor import ClaimExtractor
from modules.nli_verifier import NLIVerifier
from modules.llm_judge import LLMJudge
from modules.report_generator import ReportGenerator
from modules.database import save_document, save_verification_session
import config

def render_ask_question():
    """Renders the Unified Document Upload, Question Answering, and Verification Workspace."""

    st.markdown("""
    <div class="app-header">
        <div class="app-title">Document Audit & Verification Workspace</div>
        <div class="app-subtitle">
            Upload your document, ask questions, and audit AI-generated answers for hallucinations all in one place.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # STEP 1: PDF DOCUMENT INGESTION & MANAGEMENT (SAME PAGE)
    # =========================================================================
    has_active_doc = bool(st.session_state.get("active_document") and st.session_state.get("vector_store"))
    
    with st.expander("📄 Step 1: Document Upload & Indexing", expanded=not has_active_doc):
        if has_active_doc:
            doc = st.session_state["active_document"]
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #10B981; font-size: 1.05rem;">Active Document: {doc['filename']}</span>
                    <span style="font-size: 0.85rem; color: #94A3B8;">{doc['page_count']} Pages | {doc['chunk_count']} Passage Chunks</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload a text-based PDF document:",
            type=["pdf"],
            key="workspace_pdf_uploader",
            help="Upload research papers, legal documents, reports, or textbooks."
        )

        col_proc, col_rst = st.columns([2, 1])
        with col_proc:
            if uploaded_file is not None:
                if st.button("⚡ Process & Index Document", type="primary", use_container_width=True, key="btn_index_pdf"):
                    prog = st.progress(0)
                    stat = st.empty()
                    file_bytes = uploaded_file.read()
                    filename = uploaded_file.name

                    try:
                        file_hash = PDFProcessor.compute_file_hash(file_bytes)
                        idx_prefix = config.INDEXES_DIR / file_hash

                        # Check if previously cached index exists on disk
                        if (idx_prefix.with_suffix(".faiss")).exists() and (idx_prefix.with_suffix(".meta")).exists():
                            stat.text("Reusing cached FAISS index from disk...")
                            prog.progress(70)
                            vector_store = VectorStore(dimension=config.EMBEDDING_DIMENSION)
                            vector_store.load(idx_prefix)
                            pages, metadata = PDFProcessor.extract_text(file_bytes, filename)
                            st.session_state["extracted_pages"] = pages
                            doc_id = save_document(filename, metadata["page_count"], len(vector_store.chunks_metadata), metadata["file_hash"])
                        else:
                            stat.text("Validating PDF and extracting page text...")
                            prog.progress(25)
                            pages, metadata = PDFProcessor.extract_text(file_bytes, filename)
                            st.session_state["extracted_pages"] = pages

                            stat.text("Segmenting into overlapping passages...")
                            prog.progress(50)
                            chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                            chunks = chunker.chunk_pages(pages, filename)

                            stat.text("Generating MiniLM passage embeddings...")
                            prog.progress(75)
                            emb_mgr = EmbeddingManager()
                            embeddings = emb_mgr.encode([c["text"] for c in chunks])

                            stat.text("Building FAISS vector search index...")
                            prog.progress(90)
                            vector_store = VectorStore(dimension=config.EMBEDDING_DIMENSION)
                            vector_store.clear()
                            vector_store.add_chunks(chunks, embeddings)
                            # Persist index to disk for fast reuse
                            vector_store.save(idx_prefix)

                            # Save to database
                            doc_id = save_document(filename, metadata["page_count"], len(chunks), metadata["file_hash"])

                        st.session_state["vector_store"] = vector_store
                        st.session_state["active_document"] = {
                            "doc_id": doc_id,
                            "filename": filename,
                            "page_count": metadata["page_count"],
                            "chunk_count": len(vector_store.chunks_metadata),
                            "total_words": metadata["total_words"]
                        }

                        prog.progress(100)
                        stat.empty()
                        st.success(f"Successfully indexed **{filename}**! Ready for queries below.")
                        time.sleep(0.4)
                        st.rerun()

                    except PDFProcessingError as e:
                        prog.empty()
                        stat.empty()
                        st.error(f"PDF Validation Error: {str(e)}")
                    except Exception as e:
                        prog.empty()
                        stat.empty()
                        st.error(f"Processing Error: {str(e)}")

        with col_rst:
            if has_active_doc:
                if st.button("🔄 Clear Active Document", use_container_width=True):
                    st.session_state["active_document"] = None
                    st.session_state["vector_store"] = None
                    st.session_state["last_verification"] = None
                    st.session_state["extracted_pages"] = None
                    st.rerun()

        # Document Text Preview
        if st.session_state.get("extracted_pages"):
            pages = st.session_state["extracted_pages"]
            with st.expander(f"View Document Text ({len(pages)} pages total)"):
                p_num = st.selectbox("Page", range(1, len(pages) + 1), index=0, key="preview_page_select")
                st.text_area("Content", pages[p_num - 1]["text"], height=180, disabled=True)

    # Prompt user if no document yet
    if not st.session_state.get("active_document") or not st.session_state.get("vector_store"):
        st.info("👆 Please upload and index a PDF document above to enable question answering and verification.")
        return

    doc_info = st.session_state["active_document"]
    vector_store: VectorStore = st.session_state["vector_store"]

    # =========================================================================
    # STEP 2: QUESTION ANSWERING & VERIFICATION CONTROLS
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔍 Step 2: Ask Question or Enter Factual Claim to Verify")

    # Clean, intuitive input accepting questions or factual statements
    question = st.text_area(
        f"Enter your question or factual claim regarding '{doc_info['filename']}':",
        placeholder="Enter a Question: e.g. 'What is the required report layout?'\nOR a Factual Claim: e.g. 'The report requires at least five pages.'",
        height=110,
        key="workspace_question_input"
    )

    submit_btn = st.button("🚀 Process & Audit Factual Grounding", type="primary", use_container_width=True, key="btn_run_verification")

    # =========================================================================
    # STEP 3: EXECUTION PIPELINE
    # =========================================================================
    if submit_btn:
        if not question.strip():
            st.error("Please enter a question or factual claim.")
            return

        verification_placeholder = st.empty()
        with verification_placeholder.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
            t_start = time.time()

            try:
                top_k = config.TOP_K_RETRIEVAL
                is_q = ClaimExtractor.is_question(question.strip())

                # 1. Retrieval from active uploaded PDF
                t0_retrieval = time.time()
                status_text.text("Retrieving relevant passages from active PDF index...")
                progress_bar.progress(25)

                emb_mgr = EmbeddingManager()
                query_vector = emb_mgr.encode(question.strip())
                retrieved_passages = vector_store.search(
                    query_vector,
                    query_text=question.strip(),
                    top_k=top_k
                )
                t_retrieval = round(time.time() - t0_retrieval, 3)

                # 2. RAG Generation / Document Explanation
                t0_generation = time.time()
                if is_q:
                    status_text.text("Generating grounded answer from document evidence...")
                else:
                    status_text.text("Evaluating source document evidence regarding the statement...")
                progress_bar.progress(50)

                rag = RAGPipeline()
                answer, model_used, prompt_used = rag.generate_answer(
                    question=question.strip(),
                    retrieved_passages=retrieved_passages,
                    is_query_question=is_q
                )
                t_generation = round(time.time() - t0_generation, 3)

                # 3. Claim Extraction
                t0_extraction = time.time()
                status_text.text("Extracting and identifying verifiable factual claims...")
                progress_bar.progress(70)

                extractor = ClaimExtractor()
                claims_list = extractor.extract_claims(
                    answer,
                    user_query=question.strip(),
                    is_query_question=is_q
                )
                t_extraction = round(time.time() - t0_extraction, 3)

                # 4. Batched NLI Verification
                t0_nli = time.time()
                status_text.text(f"Verifying {len(claims_list)} claim(s) against retrieved document passages...")
                progress_bar.progress(90)

                verifier = NLIVerifier()
                verified_claims = verifier.verify_all_claims(claims_list, retrieved_passages)
                t_nli = round(time.time() - t0_nli, 3)

                # Check if NLI actually executed
                nli_used = any(c.get("nli_executed", False) for c in verified_claims)

                # 5. Database Storage
                status_text.text("Recording audit session in SQLite...")
                progress_bar.progress(98)

                session_id = save_verification_session(
                    document_name=doc_info["filename"],
                    question=question.strip(),
                    answer=answer,
                    model_used=model_used,
                    claims=verified_claims,
                    doc_id=doc_info.get("doc_id")
                )

                elapsed = round(time.time() - t_start, 2)
                progress_bar.progress(100)
                time.sleep(0.2)
                verification_placeholder.empty()

                stage_timings = {
                    "retrieval": t_retrieval,
                    "generation": t_generation,
                    "extraction": t_extraction,
                    "nli": t_nli if nli_used else None,
                    "nli_used": nli_used,
                    "total": elapsed
                }

                # Store result
                st.session_state["last_verification"] = {
                    "session_id": session_id,
                    "document_name": doc_info["filename"],
                    "question": question.strip(),
                    "answer": answer,
                    "is_question": is_q,
                    "model_used": model_used,
                    "retrieved_passages": retrieved_passages,
                    "claims": verified_claims,
                    "elapsed_time": elapsed,
                    "stage_timings": stage_timings,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

            except Exception as e:
                verification_placeholder.empty()
                st.error(f"**Verification Failed:** {str(e)}")
                with st.expander("View error details"):
                    st.code(traceback.format_exc())
                return

    # =========================================================================
    # STEP 4: VERIFICATION RESULTS DASHBOARD (SAME PAGE)
    # =========================================================================
    if st.session_state.get("last_verification"):
        import html
        res = st.session_state["last_verification"]
        claims = res["claims"]
        total_claims = len(claims)
        is_q = res.get("is_question", False)

        supported_cnt = sum(1 for c in claims if c.get("verdict") == config.VERDICT_SUPPORTED)
        contradicted_cnt = sum(1 for c in claims if c.get("verdict") == config.VERDICT_CONTRADICTED)
        unsupported_cnt = sum(1 for c in claims if c.get("verdict") == config.VERDICT_UNSUPPORTED)

        # Overall Status
        if contradicted_cnt > 0:
            overall_status = "⚠️ Contains Contradictions (Hallucinations Detected)"
            overall_color = "#EF4444"
            overall_desc = "One or more assertions directly conflict with the source document evidence."
        elif unsupported_cnt > 0:
            overall_status = "⚡ Partially Grounded / Requires Review"
            overall_color = "#F59E0B"
            overall_desc = "Some assertions lack direct verifiable evidence in the retrieved source text."
        elif total_claims > 0:
            overall_status = "✅ Fully Supported by Evidence"
            overall_color = "#10B981"
            overall_desc = "All factual claims are confirmed by the source document evidence."
        else:
            overall_status = "Inconclusive"
            overall_color = "#94A3B8"
            overall_desc = "No factual claims could be verified."

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 24px 0;'>", unsafe_allow_html=True)
        st.subheader("📊 Step 3: Verification Results & Audit Trail")

        # Results Banner with stage latency breakdown
        timings = res.get("stage_timings", {})
        nli_timing_str = f"NLI: <b>{timings.get('nli', '?')}s</b>" if timings.get("nli_used") else "Verification: <b>Deterministic Grounding</b>"
        timing_line = f"Retrieval: <b>{timings.get('retrieval', '?')}s</b> | Generation: <b>{timings.get('generation', '?')}s</b> | Extraction: <b>{timings.get('extraction', '?')}s</b> | {nli_timing_str} | Total: <b>{res.get('elapsed_time', '?')}s</b>"

        safe_model = html.escape(str(res['model_used']))
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid {overall_color}66; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.25rem; font-weight: 700; color: {overall_color};">{overall_status}</span>
                <span style="font-size: 0.85rem; color: #94A3B8;">Model: <b>{safe_model}</b></span>
            </div>
            <p style="color: #CBD5E1; font-size: 0.92rem; margin-top: 6px; margin-bottom: 6px;">{overall_desc}</p>
            <div style="font-size: 0.82rem; color: #94A3B8; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.07);">
                ⏱️ <b>Processing Latency:</b> {timing_line}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Summary Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Claims", total_claims)
        with col_m2:
            st.metric("Supported", supported_cnt, delta=f"{supported_cnt/total_claims*100:.0f}%" if total_claims else "0%")
        with col_m3:
            st.metric("Contradicted", contradicted_cnt, delta=f"{contradicted_cnt/total_claims*100:.0f}%" if total_claims else "0%", delta_color="inverse")
        with col_m4:
            st.metric("Unsupported", unsupported_cnt, delta=f"{unsupported_cnt/total_claims*100:.0f}%" if total_claims else "0%", delta_color="off")

        # Answer & Retrieved Evidence Expander
        card_title = "📝 AI Generated Answer & Source Context" if is_q else "📋 Source Document Findings & Context"
        with st.expander(card_title, expanded=True):
            if is_q:
                st.markdown(f"**Question Asked:**\n> {res['question']}")
                st.markdown(f"**Generated Answer:**\n> {res['answer']}")
            else:
                st.markdown(f"**Statement to Verify:**\n> {res['question']}")
                st.markdown(f"**Source Document Explanation:**\n> {res['answer']}")

            if res.get("retrieved_passages"):
                st.markdown("<br>**Retrieved Document Context Passages:**", unsafe_allow_html=True)
                for i, p in enumerate(res["retrieved_passages"], 1):
                    p_page = p.get('page_number', '?')
                    st.caption(f"**Passage {i} — 📄 {doc_info['filename']} (Page {p_page})** | Similarity: {p.get('similarity_score', 0.0):.3f}")
                    st.text(p.get("text", "")[:280] + ("..." if len(p.get("text", "")) > 280 else ""))

        # Separation of User Statement Claims vs Generated Answer Claims
        user_query_claims = [c for c in claims if c.get("source") in ("user_query", "user_statement", "query_assertion")]
        answer_claims = [c for c in claims if c.get("source") not in ("user_query", "user_statement", "query_assertion")]

        # 1. Statement Verification Card (when user enters a factual statement directly)
        if user_query_claims:
            st.markdown("#### 🎯 Original Statement Verification")
            for idx, c in enumerate(user_query_claims, 1):
                verdict = c.get("verdict", config.VERDICT_UNSUPPORTED)
                v_class = "supported" if verdict == config.VERDICT_SUPPORTED else ("contradicted" if verdict == config.VERDICT_CONTRADICTED else "unsupported")
                badge_class = f"badge-{v_class}"

                safe_claim = html.escape(str(c.get('claim_text', '')))
                st.markdown(f"""
                <div class="claim-container {v_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: #94A3B8; font-weight: 600;">VERIFIED STATEMENT #{idx}</span>
                        <span class="{badge_class}">{verdict.upper()}</span>
                    </div>
                    <div class="claim-text">"{safe_claim}"</div>
                </div>
                """, unsafe_allow_html=True)

                # Explanation & Evidence Box
                if c.get("explanation"):
                    st.markdown(f"💡 **Verification Explanation:** {c.get('explanation')}")

                ev_type = c.get("evidence_type", "Evidence")
                ev_excerpt = (c.get("evidence_excerpt") or c.get("evidence_text") or "").strip()
                page_cited = c.get("evidence_page")

                if verdict == config.VERDICT_SUPPORTED and ev_excerpt:
                    st.success(f"**Supporting Evidence (Page {page_cited if page_cited else '?'}):**\n\n> \"{ev_excerpt}\"")
                elif verdict == config.VERDICT_CONTRADICTED and ev_excerpt:
                    st.error(f"**Conflicting Evidence (Page {page_cited if page_cited else '?'}):**\n\n> \"{ev_excerpt}\"")
                else:
                    if ev_excerpt and page_cited:
                        st.warning(f"**Insufficient Evidence Context (Page {page_cited}):**\n\n> \"{ev_excerpt}\"")
                    else:
                        st.info("ℹ️ **Insufficient Evidence:** No relevant evidence found in the document to support or contradict this claim.")

                # Scores & Method Details
                details = []
                if c.get("nli_executed"):
                    details.append(f"**Method:** RoBERTa-large-MNLI ({c.get('nli_relation', 'N/A')})")
                    details.append(f"Entailment: `{c.get('entailment_score', 0):.3f}`")
                    details.append(f"Contradiction: `{c.get('contradiction_score', 0):.3f}`")
                    details.append(f"Neutral: `{c.get('neutral_score', 0):.3f}`")
                else:
                    details.append(f"**Method:** `{c.get('verification_method', 'Deterministic Grounding')}`")

                st.caption(" | ".join(details))

                if c.get("full_passage") and c.get("full_passage") != ev_excerpt:
                    with st.expander(f"View Full Context Passage (Page {page_cited if page_cited else '?'})"):
                        st.text(c.get("full_passage"))

        # 2. Additional / Generated Answer Claims
        if answer_claims:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header = "#### 🔎 Decomposed Answer Claims" if is_q else "#### 🔎 Additional Findings Claims"
            st.markdown(section_header)

            filter_col1, _ = st.columns([1.5, 2.5])
            with filter_col1:
                verdict_filter = st.selectbox(
                    "Filter claims:",
                    options=["All Claims", "Contradicted Only (Hallucinations)", "Unsupported Only", "Supported Only"],
                    key="workspace_verdict_filter"
                )

            filtered_claims = []
            for c in answer_claims:
                v = c.get("verdict")
                if verdict_filter == "Contradicted Only (Hallucinations)" and v != config.VERDICT_CONTRADICTED:
                    continue
                if verdict_filter == "Unsupported Only" and v != config.VERDICT_UNSUPPORTED:
                    continue
                if verdict_filter == "Supported Only" and v != config.VERDICT_SUPPORTED:
                    continue
                filtered_claims.append(c)

            if not filtered_claims:
                st.info(f"No claims match the filter: **{verdict_filter}**.")

            for idx, c in enumerate(filtered_claims, 1):
                verdict = c.get("verdict", config.VERDICT_UNSUPPORTED)
                v_class = "supported" if verdict == config.VERDICT_SUPPORTED else ("contradicted" if verdict == config.VERDICT_CONTRADICTED else "unsupported")
                badge_class = f"badge-{v_class}"

                safe_claim = html.escape(str(c.get('claim_text', '')))
                st.markdown(f"""
                <div class="claim-container {v_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 0.82rem; color: #94A3B8; font-weight: 600;">CLAIM #{idx}</span>
                        <span class="{badge_class}">{verdict.upper()}</span>
                    </div>
                    <div class="claim-text">"{safe_claim}"</div>
                </div>
                """, unsafe_allow_html=True)

                if c.get("explanation"):
                    st.markdown(f"💡 **Verification Explanation:** {c.get('explanation')}")

                ev_excerpt = (c.get("evidence_excerpt") or c.get("evidence_text") or "").strip()
                page_cited = c.get("evidence_page")

                if verdict == config.VERDICT_SUPPORTED and ev_excerpt:
                    st.success(f"**Supporting Evidence (Page {page_cited if page_cited else '?'}):**\n\n> \"{ev_excerpt}\"")
                elif verdict == config.VERDICT_CONTRADICTED and ev_excerpt:
                    st.error(f"**Conflicting Evidence (Page {page_cited if page_cited else '?'}):**\n\n> \"{ev_excerpt}\"")
                else:
                    if ev_excerpt and page_cited:
                        st.warning(f"**Insufficient Evidence Context (Page {page_cited}):**\n\n> \"{ev_excerpt}\"")
                    else:
                        st.info("ℹ️ **Insufficient Evidence:** No relevant evidence found in the document to support or contradict this claim.")

                # Scores & Method Details
                details = []
                if c.get("nli_executed"):
                    details.append(f"**Method:** RoBERTa-large-MNLI ({c.get('nli_relation', 'N/A')})")
                    details.append(f"Entailment: `{c.get('entailment_score', 0):.3f}`")
                    details.append(f"Contradiction: `{c.get('contradiction_score', 0):.3f}`")
                    details.append(f"Neutral: `{c.get('neutral_score', 0):.3f}`")
                else:
                    details.append(f"**Method:** `{c.get('verification_method', 'Deterministic Grounding')}`")

                st.caption(" | ".join(details))

                if c.get("full_passage") and c.get("full_passage") != ev_excerpt:
                    with st.expander(f"View Full Context Passage (Page {page_cited if page_cited else '?'})"):
                        st.text(c.get("full_passage"))
        st.markdown("#### 📥 Export Verification Audit Report")

        report_md = ReportGenerator.generate_markdown_report(res)
        report_txt = ReportGenerator.generate_plain_text_report(res)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📄 Download Report (Markdown .md)",
                data=report_md,
                file_name=f"audit_report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
                key="workspace_dl_md"
            )
        with dl_col2:
            st.download_button(
                label="📝 Download Report (Plain Text .txt)",
                data=report_txt,
                file_name=f"audit_report_{int(time.time())}.txt",
                mime="text/plain",
                use_container_width=True,
                key="workspace_dl_txt"
            )
