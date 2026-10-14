import pytest
from modules.claim_extractor import ClaimExtractor

def test_clean_json_response_with_markdown():
    extractor = ClaimExtractor()
    markdown_raw = """
    Here are the extracted claims:
    ```json
    [
      {"claim": "The model was released in 2023."},
      {"claim": "It has 7 billion parameters."}
    ]
    ```
    I hope this helps!
    """
    cleaned = extractor._clean_json_response(markdown_raw)
    assert cleaned.startswith("[")
    assert cleaned.endswith("]")
    assert "7 billion parameters" in cleaned

def test_rule_based_fallback():
    extractor = ClaimExtractor()
    text = "The system achieved 92% accuracy on the test benchmark. It was trained using Adam optimizer with cosine decay."
    claims = extractor._rule_based_fallback(text)
    assert len(claims) == 2
    assert "92% accuracy" in claims[0]
    assert "Adam optimizer" in claims[1]

def test_empty_input():
    extractor = ClaimExtractor()
    claims = extractor.extract_claims("")
    assert claims == []

def test_is_meta_or_uncertainty_phrase():
    extractor = ClaimExtractor()
    assert extractor.is_meta_or_uncertainty_phrase("The document does not provide enough information to establish this claim.")
    assert extractor.is_meta_or_uncertainty_phrase("The provided document does not contain sufficient information to answer this question.")
    assert extractor.is_meta_or_uncertainty_phrase("It is not specified in the document.")
    assert extractor.is_meta_or_uncertainty_phrase("The text does not mention DAX measures.")
    # Factual assertions should NOT be flagged as uncertainty
    assert not extractor.is_meta_or_uncertainty_phrase("The dataset must have at least 50 records.")
    assert not extractor.is_meta_or_uncertainty_phrase("The assignment requires DAX measures for every scenario.")

def test_extract_core_assertion_from_query():
    extractor = ClaimExtractor()
    assert extractor.extract_core_assertion_from_query(
        "The assignment requires DAX measures for every scenario."
    ) == "The assignment requires DAX measures for every scenario."

    assert extractor.extract_core_assertion_from_query(
        "The assignment requires DAX measures for every scenario. Is this statement correct according to the document?"
    ) == "The assignment requires DAX measures for every scenario."

    assert extractor.extract_core_assertion_from_query(
        "Does the assignment require DAX measures for every scenario?"
    ) == "The assignment requires DAX measures for every scenario."

    assert extractor.extract_core_assertion_from_query(
        "Is DAX required for every scenario?"
    ) == "DAX is required for every scenario."

    # Open-ended WH questions should return None for declarative assertion
    assert extractor.extract_core_assertion_from_query("What is the minimum dataset size?") is None
    assert extractor.extract_core_assertion_from_query("How many records are required?") is None

def test_extract_claims_filters_uncertainty_and_preserves_query_assertion():
    extractor = ClaimExtractor()
    answer = "The document does not provide enough information to establish this claim. The document outlines scenario-based individual assignments."
    query = "The assignment requires DAX measures for every scenario."
    extracted = extractor.extract_claims(answer, user_query=query)

    claim_texts = [c["claim_text"] for c in extracted]

    # Must preserve the tested user assertion
    assert "The assignment requires DAX measures for every scenario." in claim_texts
    # Must preserve substantive factual claim
    assert "The document outlines scenario-based individual assignments." in claim_texts
    # Must NEVER include the meta refusal statement
    for text in claim_texts:
        assert "does not provide enough information" not in text.lower()
