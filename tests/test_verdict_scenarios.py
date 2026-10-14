import pytest
import config
from modules.nli_verifier import NLIVerifier
from modules.claim_extractor import ClaimExtractor

def test_verdict_contradicted_dax_scenario(monkeypatch):
    """
    Scenario: User asks 'The assignment requires DAX measures for every scenario.'
    Document passage: 'Important Restrictions: DAX is not required for these assignments.' (Page 2)
    Verdict must be CONTRADICTED, citing Page 2.
    """
    verifier = NLIVerifier()
    monkeypatch.setattr(verifier, "predict_pair", lambda premise, hypothesis: {
        "contradiction": 0.99, "neutral": 0.01, "entailment": 0.00
    })

    claim = "The assignment requires DAX measures for every scenario."
    evidence = [
        {
            "text": "Important Restrictions: DAX is not required for these assignments. Visual calculations or standard visuals must be used.",
            "page_number": 2,
            "similarity_score": 1.42
        },
        {
            "text": "Scenario 1 describes retail analytics and report design.",
            "page_number": 1,
            "similarity_score": 0.45
        }
    ]

    result = verifier.verify_claim(claim, evidence)
    assert result["verdict"] == config.VERDICT_CONTRADICTED
    assert result["nli_relation"] == "Contradiction"
    assert result["contradiction_score"] >= 0.90
    assert result["evidence_page"] == 2
    assert "DAX is not required" in result["evidence_text"]

def test_verdict_supported_scenario(monkeypatch):
    """
    Scenario: Claim is directly affirmed by the evidence.
    Verdict must be SUPPORTED.
    """
    verifier = NLIVerifier()
    monkeypatch.setattr(verifier, "predict_pair", lambda premise, hypothesis: {
        "contradiction": 0.01, "neutral": 0.03, "entailment": 0.96
    })

    claim = "Reports must be submitted as PDF files."
    evidence = [
        {
            "text": "Submission Guidelines: Reports must be submitted as PDF files before the deadline.",
            "page_number": 5,
            "similarity_score": 1.10
        }
    ]

    result = verifier.verify_claim(claim, evidence)
    assert result["verdict"] == config.VERDICT_SUPPORTED
    assert result["nli_relation"] == "Entailment"
    assert result["entailment_score"] >= 0.90
    assert result["evidence_page"] == 5

def test_verdict_unsupported_unmentioned_scenario(monkeypatch):
    """
    Scenario: Claim asserts something completely unmentioned in the source document.
    Verdict must be UNSUPPORTED (neutral), NOT contradicted.
    """
    verifier = NLIVerifier()
    monkeypatch.setattr(verifier, "predict_pair", lambda premise, hypothesis: {
        "contradiction": 0.08, "neutral": 0.82, "entailment": 0.10
    })

    claim = "The database uses PostgreSQL version 15 with connection pooling."
    evidence = [
        {
            "text": "Scenario 3 focuses on hospital operational metrics and bed occupancy rates.",
            "page_number": 12,
            "similarity_score": 0.38
        }
    ]

    result = verifier.verify_claim(claim, evidence)
    assert result["verdict"] == config.VERDICT_UNSUPPORTED
    assert result["nli_relation"] == "Neutral"
    assert result["neutral_score"] >= 0.70
    assert result["contradiction_score"] <= 0.15

def test_verdict_insufficient_information_scenario():
    """
    Scenario: LLM answer expresses refusal or uncertainty:
    'The provided document does not contain sufficient information to establish this claim.'
    The system must NOT classify this uncertainty sentence as CONTRADICTED.
    Instead, it should recognize it as an uncertainty statement and yield UNSUPPORTED.
    """
    extractor = ClaimExtractor()
    uncertainty_answer = "The provided document does not contain sufficient information to determine whether DAX is required."
    user_query = "The assignment requires DAX measures for every scenario."

    # Verify claim extractor identifies uncertainty and preserves query assertion
    claims = extractor.extract_claims(uncertainty_answer, user_query=user_query)
    assert len(claims) >= 1
    # Check that the raw uncertainty sentence is NOT emitted as a standalone verifiable claim
    claim_texts = [c["claim_text"] for c in claims]
    for ct in claim_texts:
        assert not extractor.is_meta_or_uncertainty_phrase(ct)

    # When verified, even if tested against evidence with neutral score
    verifier = NLIVerifier()
    # Test directly with an uncertainty statement passed to verify_claim
    res = verifier.verify_claim(
        claim_text=uncertainty_answer,
        evidence_passages=[{"text": "General course policies and grading rubric.", "page_number": 1, "similarity_score": 0.40}]
    )
    assert res["verdict"] == config.VERDICT_UNSUPPORTED
    assert res["nli_relation"] == "Neutral"
    assert res["contradiction_score"] <= 0.15
