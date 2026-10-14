import streamlit as st
import traceback
from modules.pdf_processor import PDFProcessor, PDFProcessingError
from modules.text_chunker import TextChunker
from modules.embeddings import EmbeddingManager
from modules.vector_store import VectorStore
from modules.database import save_document
import config

def render_upload_document():
    """Renders the PDF Document Upload and Indexing interface."""
    
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Document Processing & Indexing</div>
        <div class="app-subtitle">
            Upload text-based PDF documents to extract page-level text, generate MiniLM embeddings, and build FAISS vector index.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Current Document Session Status Banner
    if st.session_state.get("active_document"):
        doc_info = st.session_state["active_document"]
        st.info(
            f"Active Document Session: **{doc_info['filename']}** "
            f"({doc_info['page_count']} pages, {doc_info['chunk_count']} index chunks)"
        )
        if st.button("🔄 Reset / Start New Document Session", type="secondary"):
            st.session_state["active_document"] = None
            st.session_state["vector_store"] = None
            st.session_state["retrieved_passages"] = None
            st.session_state["last_verification"] = None
            st.rerun()

    # File Uploader
    uploaded_file = st.file_uploader(
        "Select a PDF document (text-based)",
        type=["pdf"],
        help="Upload academic papers, reports, or documentation in PDF format."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        col_meta1, col_meta2 = st.columns([1, 1])
        with col_meta1:
            st.caption(f"**Filename:** {filename}")
        with col_meta2:
            st.caption(f"**Filesize:** {len(file_bytes) / 1024:.1f} KB")

        # Process Button
        if st.button("⚡ Process & Index Document", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Stage 1: Validation and Extraction
                status_text.text("Validating PDF structure and extracting page-level text...")
                progress_bar.progress(20)
                
                pages, metadata = PDFProcessor.extract_text(file_bytes, filename)
                st.session_state["extracted_pages"] = pages
                st.session_state["doc_metadata"] = metadata

                # Stage 2: Chunking
                status_text.text(f"Chunking {metadata['page_count']} pages into overlapping passages...")
                progress_bar.progress(45)
                
                chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                chunks = chunker.chunk_pages(pages, filename)
                st.session_state["chunks"] = chunks

                # Stage 3: Embedding Generation
                status_text.text(f"Computing MiniLM embeddings for {len(chunks)} passages...")
                progress_bar.progress(70)
                
                emb_mgr = EmbeddingManager()
                texts = [c["text"] for c in chunks]
                embeddings = emb_mgr.encode(texts)

                # Stage 4: FAISS Vector Indexing
                status_text.text("Building FAISS index with inner-product similarity...")
                progress_bar.progress(90)
                
                vector_store = VectorStore(dimension=config.EMBEDDING_DIMENSION)
                vector_store.clear()
                vector_store.add_chunks(chunks, embeddings)
                
                # Persist to database
                doc_id = save_document(
                    filename=filename,
                    page_count=metadata["page_count"],
                    chunk_count=len(chunks),
                    file_hash=metadata["file_hash"]
                )

                st.session_state["vector_store"] = vector_store
                st.session_state["active_document"] = {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_count": metadata["page_count"],
                    "chunk_count": len(chunks),
                    "total_words": metadata["total_words"]
                }

                progress_bar.progress(100)
                status_text.empty()
                st.success(
                    f"Successfully processed **{filename}**! "
                    f"Generated {len(chunks)} searchable passage vectors across {metadata['page_count']} page(s)."
                )

            except PDFProcessingError as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"**PDF Processing Warning:** {str(e)}")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"**Unexpected Processing Error:** {str(e)}")
                with st.expander("Show detailed error log"):
                    st.code(traceback.format_exc())

    # Document Inspection & Preview
    if st.session_state.get("extracted_pages"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📄 Document Inspection & Text Preview")
        
        pages = st.session_state["extracted_pages"]
        total_p = len(pages)

        selected_page = st.selectbox(
            f"Select page to inspect (1 to {total_p})",
            options=range(1, total_p + 1),
            index=0
        )

        page_data = pages[selected_page - 1]
        st.caption(f"Page {selected_page} contains {page_data['word_count']} words ({page_data['char_count']} characters)")
        st.text_area(
            "Extracted Page Content",
            value=page_data["text"],
            height=260,
            disabled=True
        )

        if st.session_state.get("chunks"):
            with st.expander(f"View Chunking Details ({len(st.session_state['chunks'])} passages generated)"):
                for i, chk in enumerate(st.session_state["chunks"][:6], 1):
                    st.markdown(f"**Passage {i} (Page {chk['page_number']}) — {chk['word_count']} words**")
                    st.code(chk["text"][:300] + ("..." if len(chk["text"]) > 300 else ""), language="text")
                if len(st.session_state["chunks"]) > 6:
                    st.caption(f"... and {len(st.session_state['chunks']) - 6} additional passages indexed.")
