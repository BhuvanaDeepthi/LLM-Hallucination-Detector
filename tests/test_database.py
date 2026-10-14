import pytest
import sqlite3
import config
from modules.database import (
    init_db,
    save_document,
    get_all_documents,
    save_verification_session,
    get_all_sessions,
    get_session_by_id,
    delete_session,
    clear_all_history,
    get_system_stats
)

@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Overrides DB_PATH to use an isolated temporary database for each test."""
    test_db = tmp_path / "test_detector.db"
    monkeypatch.setattr(config, "DB_PATH", test_db)
    init_db()

def test_document_crud():
    doc_id = save_document("research_paper.pdf", page_count=5, chunk_count=12, file_hash="hash123")
    assert doc_id > 0

    docs = get_all_documents()
    assert len(docs) == 1
    assert docs[0]["filename"] == "research_paper.pdf"
    assert docs[0]["page_count"] == 5

def test_session_and_claims_flow():
    doc_id = save_document("test.pdf", 2, 4, "hashabc")

    sample_claims = [
        {
            "claim_text": "Model achieved 95% accuracy.",
            "verdict": config.VERDICT_SUPPORTED,
            "nli_relation": "Entailment",
            "entailment_score": 0.92,
            "contradiction_score": 0.02,
            "neutral_score": 0.06,
            "evidence_text": "The model reached 95% accuracy on benchmark test.",
            "evidence_page": 1,
            "evidence_score": 0.88,
            "judge_used": False,
            "judge_assessment": None
        },
        {
            "claim_text": "Model was released in 1990.",
            "verdict": config.VERDICT_CONTRADICTED,
            "nli_relation": "Contradiction",
            "entailment_score": 0.01,
            "contradiction_score": 0.94,
            "neutral_score": 0.05,
            "evidence_text": "The model was released in 2023.",
            "evidence_page": 2,
            "evidence_score": 0.75,
            "judge_used": True,
            "judge_assessment": "Direct date conflict."
        }
    ]

    session_id = save_verification_session(
        document_name="test.pdf",
        question="What were the model results and year?",
        answer="The model achieved 95% accuracy and was released in 1990.",
        model_used="Ollama (gemma3:latest)",
        claims=sample_claims,
        doc_id=doc_id
    )

    assert session_id > 0

    session = get_session_by_id(session_id)
    assert session is not None
    assert session["total_claims"] == 2
    assert session["supported_claims"] == 1
    assert session["contradicted_claims"] == 1
    assert session["overall_verdict"] == "Contains Contradictions"
    assert len(session["claims"]) == 2

    # Check live stats
    stats = get_system_stats()
    assert stats["documents_processed"] == 1
    assert stats["verification_sessions"] == 1
    assert stats["claims_checked"] == 2
    assert stats["claims_flagged"] == 1

    # Delete session
    deleted = delete_session(session_id)
    assert deleted is True
    assert get_session_by_id(session_id) is None
