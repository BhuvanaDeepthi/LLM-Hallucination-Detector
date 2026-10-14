import pytest
import numpy as np
from pathlib import Path
from modules.vector_store import VectorStore
from modules.embeddings import EmbeddingManager
import config

def test_vector_store_empty():
    vs = VectorStore(dimension=384)
    res = vs.search(np.zeros((384,)), query_text="test query")
    assert res == []

def test_vector_store_save_and_load(tmp_path):
    vs = VectorStore(dimension=384)
    chunks = [
        {"text": "Scenario 1 describes sales transactions.", "page_number": 1, "document_name": "test.pdf"},
        {"text": "Important Restrictions: DAX is not required for these assignments.", "page_number": 2, "document_name": "test.pdf"}
    ]
    # Synthetic normalized embeddings
    embs = np.random.randn(2, 384).astype(np.float32)
    vs.add_chunks(chunks, embs)
    assert vs.index.ntotal == 2

    prefix = tmp_path / "test_idx"
    vs.save(prefix)

    vs_loaded = VectorStore(dimension=384)
    success = vs_loaded.load(prefix)
    assert success is True
    assert vs_loaded.index.ntotal == 2
    assert len(vs_loaded.chunks_metadata) == 2
    assert vs_loaded.chunks_metadata[1]["page_number"] == 2
    assert "DAX is not required" in vs_loaded.chunks_metadata[1]["text"]

def test_hybrid_search_boosts_rare_keyword_over_generic_similarity():
    """
    Test that BM25-IDF keyword boosting elevates a chunk containing a rare acronym ('DAX')
    even if another generic passage had slightly higher initial dense similarity.
    """
    vs = VectorStore(dimension=4)
    chunks = [
        {"text": "This is a general overview of business intelligence and report design principles.", "page_number": 1},
        {"text": "Important Restrictions: DAX is not required for these assignments.", "page_number": 2},
        {"text": "Detailed descriptions of dataset formatting, data cleansing, and bar chart styling.", "page_number": 18},
    ]

    # Chunk 0 has slightly higher dense similarity to query
    # Chunk 1 has lower dense similarity, but contains the rare keyword "DAX"
    # Chunk 2 has low similarity
    embeddings = np.array([
        [0.8, 0.2, 0.1, 0.0],   # Chunk 0: high dense alignment with query [1, 0, 0, 0]
        [0.4, 0.5, 0.1, 0.0],   # Chunk 1: moderate dense alignment
        [0.1, 0.1, 0.1, 0.1],   # Chunk 2: low dense alignment
    ], dtype=np.float32)

    vs.add_chunks(chunks, embeddings)

    # Query with direction [1, 0, 0, 0] and text mentioning DAX
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    query_text = "The assignment requires DAX measures for every scenario."

    # Pure dense search (without query_text) -> Chunk 0 is first
    dense_only = vs.search(query_vec, query_text=None, top_k=2)
    assert dense_only[0]["page_number"] == 1

    # Hybrid search (with query_text) -> Chunk 1 with "DAX" must be boosted to #1
    hybrid_results = vs.search(query_vec, query_text=query_text, top_k=2)
    assert hybrid_results[0]["page_number"] == 2
    assert "DAX is not required" in hybrid_results[0]["text"]
    assert hybrid_results[0]["similarity_score"] > hybrid_results[1]["similarity_score"]
