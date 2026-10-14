import pytest
from unittest.mock import MagicMock
from modules.evaluation import CustomDatasetEvaluator, HaluEvalEvaluator


def test_custom_evaluator_metrics_computation(tmp_path):
    # Mock verifier to return deterministic predictions
    mock_verifier = MagicMock()
    mock_verifier.predict_batch.return_value = [
        {"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03},  # SUPPORTED
        {"entailment": 0.01, "contradiction": 0.98, "neutral": 0.01},  # CONTRADICTED
        {"entailment": 0.10, "contradiction": 0.10, "neutral": 0.80},  # UNSUPPORTED
    ]

    # Create dummy custom eval CSV
    csv_file = tmp_path / "custom_test.csv"
    csv_file.write_text(
        "example_id,domain,evidence_passage,claim,expected_verdict,label_rationale,label_origin,split\n"
        "T01,Software,Report is PDF.,Must be PDF.,SUPPORTED,Explicit format.,rule,test\n"
        "T02,Software,Report is PDF.,Must be PPT.,CONTRADICTED,Conflict.,rule,test\n"
        "T03,Software,Report is PDF.,Has 50 pages.,UNSUPPORTED,Not mentioned.,rule,test\n",
        encoding="utf-8"
    )

    evaluator = CustomDatasetEvaluator(verifier=mock_verifier, output_dir=tmp_path)
    res = evaluator.evaluate(csv_path=str(csv_file), batch_size=4)

    summary = res["summary"]
    assert summary["total_examples"] == 3
    assert summary["total_correct"] == 3
    assert summary["overall_accuracy"] == 1.0
    assert summary["per_class_metrics"]["SUPPORTED"]["f1_score"] == 1.0
    assert summary["per_class_metrics"]["CONTRADICTED"]["f1_score"] == 1.0
    assert summary["per_class_metrics"]["UNSUPPORTED"]["f1_score"] == 1.0


def test_halueval_evaluator_binary_metrics_computation(tmp_path):
    mock_verifier = MagicMock()
    # 1 record -> 2 candidate answers:
    # 1. Correct Answer -> Predicts SUPPORTED (entailment=0.90) -> Faithful (0)
    # 2. Hallucinated Answer -> Predicts CONTRADICTED (contradiction=0.92) -> Hallucinated (1)
    mock_verifier.predict_batch.return_value = [
        {"entailment": 0.90, "contradiction": 0.05, "neutral": 0.05},
        {"entailment": 0.02, "contradiction": 0.92, "neutral": 0.06}
    ]

    csv_file = tmp_path / "halueval_test.csv"
    csv_file.write_text(
        "knowledge,raw_question,correct_answer,hallucinated_answer\n"
        "Arthur Magazine was published in Philadelphia in the 19th century.,Which city?,Philadelphia,New York\n",
        encoding="utf-8"
    )

    evaluator = HaluEvalEvaluator(verifier=mock_verifier, output_dir=tmp_path)
    res = evaluator.evaluate(csv_path=str(csv_file), sample_size=1, batch_size=2)

    bm = res["summary"]["binary_metrics"]
    assert bm["accuracy"] == 1.0
    assert bm["precision"] == 1.0
    assert bm["recall"] == 1.0
    assert bm["f1_score"] == 1.0
    assert bm["true_positives"] == 1
    assert bm["true_negatives"] == 1
    assert bm["false_positives"] == 0
    assert bm["false_negatives"] == 0
