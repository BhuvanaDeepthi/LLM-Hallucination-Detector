import pytest
import config
from modules.nli_verifier import NLIVerifier

def test_verdict_decision_mapping(monkeypatch):
    verifier = NLIVerifier()

    # Case 1: High entailment -> Supported
    monkeypatch.setattr(verifier, "predict_pair", lambda premise, hypothesis: {
        "contradiction": 0.05, "neutral": 0.15, "entailment": 0.80
    })
    res = verifier.verify_claim(
        claim_text="The sky is blue.",
        evidence_passages=[{"text": "The sky is clear and blue.", "page_number": 1, "similarity_score": 0.89}]
    )
    assert res["verdict"] == config.VERDICT_SUPPORTED
    assert res["nli_relation"] == "Entailment"
    assert res["evidence_page"] == 1

    # Case 2: High contradiction -> Contradicted
    monkeypatch.setattr(verifier, "predict_pair", lambda premise, hypothesis: {
        "contradiction": 0.88, "neutral": 0.10, "entailment": 0.02
    })
    res2 = verifier.verify_claim(
        claim_text="The sky is green.",
        evidence_passages=[{"text": "The sky is blue and clear.", "page_number": 1, "similarity_score": 0.85}]
    )
    assert res2["verdict"] == config.VERDICT_CONTRADICTED
    assert res2["nli_relation"] == "Contradiction"

    # Case 3: High neutral -> Unsupported (Not Contradicted!)
    monkeypatch.setattr(verifier, "predict_pair", lambda premise, hypothesis: {
        "contradiction": 0.10, "neutral": 0.80, "entailment": 0.10
    })
    res3 = verifier.verify_claim(
        claim_text="The airplane was manufactured by Boeing in Seattle.",
        evidence_passages=[{"text": "The flight was delayed due to heavy rain.", "page_number": 3, "similarity_score": 0.45}]
    )
    assert res3["verdict"] == config.VERDICT_UNSUPPORTED
    assert res3["nli_relation"] == "Neutral"

def test_empty_evidence_handling():
    verifier = NLIVerifier()
    res = verifier.verify_claim("Some claim.", [])
    assert res["verdict"] == config.VERDICT_UNSUPPORTED
    assert res["nli_relation"] == "Neutral"

def test_insufficient_information_refusal_not_contradicted():
    """Refusal and uncertainty statements should never be marked as contradicted."""
    verifier = NLIVerifier()
    res = verifier.verify_claim(
        claim_text="The document does not provide enough information to establish this claim.",
        evidence_passages=[{"text": "Scenario 1 describes retail analytics and data modeling.", "page_number": 1, "similarity_score": 0.85}]
    )
    assert res["verdict"] == config.VERDICT_UNSUPPORTED
    assert res["nli_relation"] == "Neutral"
    assert res["contradiction_score"] <= 0.15

def test_dax_unmentioned_claim_is_unsupported_not_contradicted(monkeypatch):
    """When the document does not mention DAX, the claim must be UNSUPPORTED, not CONTRADICTED."""
    verifier = NLIVerifier()
    monkeypatch.setattr(verifier, "predict_pair", lambda premise, hypothesis: {
        "contradiction": 0.07, "neutral": 0.78, "entailment": 0.15
    })
    res = verifier.verify_claim(
        claim_text="The assignment requires DAX measures for every scenario.",
        evidence_passages=[{"text": "The assignment involves dataset loading, preparation, and report development.", "page_number": 2, "similarity_score": 0.82}]
    )
    assert res["verdict"] == config.VERDICT_UNSUPPORTED
    assert res["nli_relation"] == "Neutral"
    assert res["contradiction_score"] == 0.07
    assert res["neutral_score"] == 0.78
    assert res["evidence_page"] == 2

def test_dax_explicit_conflict_is_contradicted(monkeypatch):
    """When the document explicitly states DAX is not required, the claim must be CONTRADICTED."""
    verifier = NLIVerifier()
    monkeypatch.setattr(verifier, "predict_pair", lambda premise, hypothesis: {
        "contradiction": 0.94, "neutral": 0.04, "entailment": 0.02
    })
    res = verifier.verify_claim(
        claim_text="The assignment requires DAX measures for every scenario.",
        evidence_passages=[{"text": "DAX measures are not required for any scenario in this assignment.", "page_number": 3, "similarity_score": 0.95}]
    )
    assert res["verdict"] == config.VERDICT_CONTRADICTED
    assert res["nli_relation"] == "Contradiction"
    assert res["contradiction_score"] == 0.94
    assert res["evidence_page"] == 3

def test_dax_explicit_requirement_is_supported(monkeypatch):
    """When the document explicitly states DAX is required, the claim must be SUPPORTED."""
    verifier = NLIVerifier()
    monkeypatch.setattr(verifier, "predict_pair", lambda premise, hypothesis: {
        "contradiction": 0.01, "neutral": 0.04, "entailment": 0.95
    })
    res = verifier.verify_claim(
        claim_text="The assignment requires DAX measures for every scenario.",
        evidence_passages=[{"text": "Students must create DAX measures for every scenario.", "page_number": 4, "similarity_score": 0.96}]
    )
    assert res["verdict"] == config.VERDICT_SUPPORTED
    assert res["nli_relation"] == "Entailment"
    assert res["entailment_score"] == 0.95
    assert res["evidence_page"] == 4

def test_salient_sentences_preserves_surrounding_context():
    verifier = NLIVerifier()
    passage = "Introduction to Power BI. Each scenario focuses on distinct business domains. Dataset must have at least 50 records. DAX measures are optional. Reports must be exported as PDF."
    claim = "DAX measures are required for every scenario."
    salient = verifier.extract_salient_sentences(passage, claim, max_sentences=3)
    assert "DAX measures" in salient
    # Surrounding context is included
    assert len(salient.split()) > 10
