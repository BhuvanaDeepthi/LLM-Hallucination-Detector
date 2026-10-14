"""
Dual-Dataset Evaluation Module for LLM Hallucination Detector

Provides standalone, offline evaluation for:
1. Custom Controlled Dataset (hallucination_detector_custom_eval.csv)
   - Evaluates multi-domain claims against paired evidence passages.
   - 3-Class classification: SUPPORTED, CONTRADICTED, UNSUPPORTED.
   - Calculates per-class precision, recall, F1, confusion matrix, domain breakdowns, and latency.

2. Standardized Benchmark Dataset (halueval.csv)
   - Evaluates QA knowledge-answer pairs from HaluEval.
   - Evaluates faithful (correct) vs hallucinated answers against reference knowledge.
   - Computes binary hallucination detection metrics (Precision, Recall, F1, FPR, Confusion Matrix)
     and 3-class attribution distributions.
   - Strictly preserves data separation without data leakage or hardcoded verdicts.
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import config
from modules.nli_verifier import NLIVerifier


class CustomDatasetEvaluator:
    """
    Offline evaluation harness for the custom controlled evaluation dataset (60 examples).
    Tests claims against provided evidence passages without retrieval bias or data leakage.
    """

    def __init__(self, verifier: Optional[NLIVerifier] = None, output_dir: Optional[Path] = None):
        self.verifier = verifier or NLIVerifier()
        self.output_dir = output_dir or (config.DATA_DIR / "reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        csv_path: str = "hallucination_detector_custom_eval.csv",
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Executes evaluation on the 60 custom examples.
        Compares actual predicted verdict with expected_verdict.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Custom evaluation dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = [
            "example_id", "domain", "evidence_passage", "claim",
            "expected_verdict", "label_rationale", "label_origin", "split"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in custom eval CSV: {col}")

        total_examples = len(df)
        print(f"\n=================================================================")
        print(f"STARTING CUSTOM DATASET EVALUATION ({total_examples} examples)")
        print(f"=================================================================")

        # Build inference pairs: (premise, hypothesis)
        pairs = []
        for _, row in df.iterrows():
            premise = str(row["evidence_passage"]).strip()
            hypothesis = str(row["claim"]).strip()
            pairs.append((premise, hypothesis))

        # Batched NLI inference with per-batch progress logging
        t_start = time.perf_counter()
        raw_scores = []
        n_batches = int(np.ceil(len(pairs) / batch_size))

        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end = min(b_start + batch_size, len(pairs))
            batch = pairs[b_start:b_end]

            t_b0 = time.perf_counter()
            b_res = self.verifier.predict_batch(batch, batch_size=batch_size)
            raw_scores.extend(b_res)
            t_batch = time.perf_counter() - t_b0

            print(f"  Processed batch {b_idx + 1}/{n_batches} ({b_end}/{len(pairs)} pairs) in {t_batch:.2f}s")

        total_inference_time = time.perf_counter() - t_start
        avg_latency = total_inference_time / total_examples if total_examples > 0 else 0.0

        # Classify verdicts and build prediction records
        records = []
        y_true = []
        y_pred = []
        classes = ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]

        for idx, row in df.iterrows():
            scores = raw_scores[idx]
            e = scores["entailment"]
            c = scores["contradiction"]
            n = scores["neutral"]

            # Standard calibrated decision logic
            if e >= config.ENTAILMENT_THRESHOLD and e > c and e > n:
                pred_verdict = "SUPPORTED"
            elif c >= config.CONTRADICTION_THRESHOLD and c > e and c > n:
                pred_verdict = "CONTRADICTED"
            else:
                pred_verdict = "UNSUPPORTED"

            exp_verdict = str(row["expected_verdict"]).strip().upper()
            is_correct = (pred_verdict == exp_verdict)

            y_true.append(exp_verdict)
            y_pred.append(pred_verdict)

            records.append({
                "example_id": row["example_id"],
                "domain": row["domain"],
                "evidence_passage": row["evidence_passage"],
                "claim": row["claim"],
                "expected_verdict": exp_verdict,
                "predicted_verdict": pred_verdict,
                "is_correct": is_correct,
                "entailment_score": round(float(e), 4),
                "contradiction_score": round(float(c), 4),
                "neutral_score": round(float(n), 4),
                "latency_sec": round(avg_latency, 4),
                "label_rationale": row["label_rationale"],
                "label_origin": row["label_origin"],
                "split": row["split"],
                "verification_engine": "roberta-large-mnli (calibrated)"
            })

        # Calculate metrics using scikit-learn
        overall_accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        per_class_metrics = {}
        for c in classes:
            prec = precision_score(y_true, y_pred, labels=[c], average="micro", zero_division=0)
            rec = recall_score(y_true, y_pred, labels=[c], average="micro", zero_division=0)
            f1 = f1_score(y_true, y_pred, labels=[c], average="micro", zero_division=0)
            support = sum(1 for y in y_true if y == c)
            per_class_metrics[c] = {
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "support": int(support)
            }

        macro_p = precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        macro_r = recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

        weighted_p = precision_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)
        weighted_r = recall_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)

        # Domain breakdown
        domain_breakdown = {}
        for d in sorted(df["domain"].unique()):
            d_records = [r for r in records if r["domain"] == d]
            d_correct = sum(1 for r in d_records if r["is_correct"])
            domain_breakdown[d] = {
                "total": len(d_records),
                "correct": d_correct,
                "accuracy": round(d_correct / len(d_records), 4) if d_records else 0.0
            }

        summary = {
            "dataset_name": "Custom Controlled Evaluation Dataset (60 examples)",
            "dataset_path": csv_path,
            "total_examples": total_examples,
            "total_correct": int(sum(1 for r in records if r["is_correct"])),
            "overall_accuracy": round(float(overall_accuracy), 4),
            "macro_avg": {
                "precision": round(float(macro_p), 4),
                "recall": round(float(macro_r), 4),
                "f1_score": round(float(macro_f1), 4)
            },
            "weighted_avg": {
                "precision": round(float(weighted_p), 4),
                "recall": round(float(weighted_r), 4),
                "f1_score": round(float(weighted_f1), 4)
            },
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": {
                "labels": classes,
                "matrix": cm.tolist()
            },
            "domain_breakdown": domain_breakdown,
            "timing": {
                "total_inference_time_sec": round(float(total_inference_time), 4),
                "avg_latency_per_example_sec": round(float(avg_latency), 4),
                "throughput_pairs_per_sec": round(float(total_examples / total_inference_time), 2) if total_inference_time > 0 else 0.0
            }
        }

        # Save outputs
        pred_df = pd.DataFrame(records)
        pred_csv = self.output_dir / "custom_eval_predictions.csv"
        pred_json = self.output_dir / "custom_eval_predictions.json"
        metrics_json = self.output_dir / "custom_eval_metrics.json"
        report_md = self.output_dir / "custom_eval_report.md"

        pred_df.to_csv(pred_csv, index=False)
        with open(pred_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._generate_custom_markdown_report(report_md, summary, records)

        print(f"\n[Custom Dataset Evaluation Complete]")
        print(f"  Overall Accuracy: {summary['overall_accuracy']*100:.2f}% ({summary['total_correct']}/{total_examples})")
        print(f"  Macro F1-Score:   {summary['macro_avg']['f1_score']:.4f}")
        print(f"  Total Duration:   {total_inference_time:.2f}s ({summary['timing']['throughput_pairs_per_sec']} ex/sec)")
        print(f"  Saved outputs to: {self.output_dir}\n")

        return {
            "summary": summary,
            "records": records,
            "saved_files": {
                "predictions_csv": str(pred_csv),
                "predictions_json": str(pred_json),
                "metrics_json": str(metrics_json),
                "report_md": str(report_md)
            }
        }

    def _generate_custom_markdown_report(self, path: Path, summary: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
        """Generates a comprehensive Markdown evaluation report."""
        cm = summary["confusion_matrix"]["matrix"]
        classes = summary["confusion_matrix"]["labels"]

        md = [
            f"# Custom Controlled Dataset Evaluation Report",
            f"",
            f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Dataset:** `{summary['dataset_name']}` ({summary['total_examples']} examples)",
            f"**Verification Engine:** `roberta-large-mnli` (Calibrated Thresholds)",
            f"",
            f"---",
            f"",
            f"## 1. Executive Summary",
            f"",
            f"| Metric | Value |",
            f"| :--- | :--- |",
            f"| **Overall Accuracy** | **{summary['overall_accuracy']*100:.2f}%** ({summary['total_correct']}/{summary['total_examples']}) |",
            f"| **Macro Precision** | {summary['macro_avg']['precision']:.4f} |",
            f"| **Macro Recall** | {summary['macro_avg']['recall']:.4f} |",
            f"| **Macro F1-Score** | **{summary['macro_avg']['f1_score']:.4f}** |",
            f"| **Weighted F1-Score** | {summary['weighted_avg']['f1_score']:.4f} |",
            f"| **Total Latency** | {summary['timing']['total_inference_time_sec']:.2f} seconds |",
            f"| **Throughput** | {summary['timing']['throughput_pairs_per_sec']} examples/sec |",
            f"",
            f"---",
            f"",
            f"## 2. Per-Class Performance",
            f"",
            f"| Class | Support | Precision | Recall | F1-Score |",
            f"| :--- | :--- | :--- | :--- | :--- |"
        ]

        for c in classes:
            m = summary["per_class_metrics"][c]
            md.append(f"| **{c}** | {m['support']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} |")

        md.extend([
            f"",
            f"---",
            f"",
            f"## 3. Confusion Matrix",
            f"",
            f"| True \\ Predicted | {classes[0]} | {classes[1]} | {classes[2]} |",
            f"| :--- | :--- | :--- | :--- |",
            f"| **{classes[0]}** | **{cm[0][0]}** | {cm[0][1]} | {cm[0][2]} |",
            f"| **{classes[1]}** | {cm[1][0]} | **{cm[1][1]}** | {cm[1][2]} |",
            f"| **{classes[2]}** | {cm[2][0]} | {cm[2][1]} | **{cm[2][2]}** |",
            f"",
            f"---",
            f"",
            f"## 4. Performance by Domain",
            f"",
            f"| Domain | Total | Correct | Accuracy |",
            f"| :--- | :--- | :--- | :--- |"
        ])

        for d, dm in summary["domain_breakdown"].items():
            md.append(f"| **{d}** | {dm['total']} | {dm['correct']} | {dm['accuracy']*100:.1f}% |")

        # Discrepancy analysis
        mismatches = [r for r in records if not r["is_correct"]]
        md.extend([
            f"",
            f"---",
            f"",
            f"## 5. Discrepancy & Boundary Case Analysis ({len(mismatches)} instances)",
            f"",
            f"The following instances exhibited boundaries where zero-shot linguistic entailment differed from strict real-world mathematical or temporal domain rules:",
            f""
        ])

        if mismatches:
            for m in mismatches:
                md.append(f"- **ID `{m['example_id']}` ({m['domain']}):**")
                md.append(f"  - *Evidence:* \"{m['evidence_passage']}\"")
                md.append(f"  - *Claim:* \"{m['claim']}\"")
                md.append(f"  - *Expected:* `{m['expected_verdict']}` | *Predicted:* `{m['predicted_verdict']}`")
                md.append(f"  - *Scores:* Entailment={m['entailment_score']}, Contradiction={m['contradiction_score']}, Neutral={m['neutral_score']}")
                md.append(f"  - *Label Rationale:* {m['label_rationale']}")
                md.append(f"")
        else:
            md.append("*(No discrepancies; all 60 examples were correctly classified!)*\n")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))


class HaluEvalEvaluator:
    """
    Offline evaluation harness for HaluEval (QA subset: 2,000 records).
    Tests both faithful (correct_answer) and unfaithful (hallucinated_answer) candidate answers
    against reference knowledge.
    Computes binary hallucination detection metrics alongside fine-grained 3-class distribution analysis.
    """

    def __init__(self, verifier: Optional[NLIVerifier] = None, output_dir: Optional[Path] = None):
        self.verifier = verifier or NLIVerifier()
        self.output_dir = output_dir or (config.DATA_DIR / "reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        csv_path: str = "halueval.csv",
        sample_size: Optional[int] = None,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Executes evaluation on HaluEval records.
        Evaluates correct_answer and hallucinated_answer pairs separately against reference knowledge.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"HaluEval dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = ["knowledge", "raw_question", "correct_answer", "hallucinated_answer"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in HaluEval CSV: {col}")

        if sample_size and sample_size < len(df):
            df = df.iloc[:sample_size].copy()
            run_mode = f"Sample ({sample_size} records, {sample_size * 2} candidate answers)"
            file_prefix = f"halueval_sample_{sample_size}"
        else:
            run_mode = f"Full Dataset ({len(df)} records, {len(df) * 2} candidate answers)"
            file_prefix = "halueval_full"

        n_records = len(df)
        total_instances = n_records * 2
        print(f"\n=================================================================")
        print(f"STARTING HALUEVAL BENCHMARK EVALUATION: {run_mode}")
        print(f"=================================================================")

        # Build candidate instances
        # Each record generates 2 instances:
        # 1. correct_answer: Ground truth = Faithful (0)
        # 2. hallucinated_answer: Ground truth = Hallucinated (1)
        instances = []
        pairs = []

        for row_idx, row in df.iterrows():
            k = str(row["knowledge"]).strip()
            q = str(row["raw_question"]).strip()
            ca = str(row["correct_answer"]).strip()
            ha = str(row["hallucinated_answer"]).strip()

            # 1. Faithful candidate
            instances.append({
                "record_id": int(row_idx),
                "instance_id": f"HALU_{row_idx:04d}_FAITHFUL",
                "candidate_type": "correct_answer",
                "candidate_text": ca,
                "knowledge": k,
                "raw_question": q,
                "ground_truth_binary": 0,  # 0 = Faithful
                "expected_verdict_3class": "SUPPORTED"
            })
            pairs.append((k, ca))

            # 2. Hallucinated candidate
            instances.append({
                "record_id": int(row_idx),
                "instance_id": f"HALU_{row_idx:04d}_HALLUCINATED",
                "candidate_type": "hallucinated_answer",
                "candidate_text": ha,
                "knowledge": k,
                "raw_question": q,
                "ground_truth_binary": 1,  # 1 = Hallucinated
                "expected_verdict_3class": "HALLUCINATED"  # (Contradicted or Unsupported)
            })
            pairs.append((k, ha))

        # Batched inference with ETA reporting
        t_start = time.perf_counter()
        raw_scores = []
        n_batches = int(np.ceil(len(pairs) / batch_size))

        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end = min(b_start + batch_size, len(pairs))
            batch = pairs[b_start:b_end]

            t_b0 = time.perf_counter()
            b_res = self.verifier.predict_batch(batch, batch_size=batch_size)
            raw_scores.extend(b_res)
            t_batch = time.perf_counter() - t_b0

            elapsed = time.perf_counter() - t_start
            rate = len(raw_scores) / elapsed if elapsed > 0 else 0.0
            remaining_pairs = len(pairs) - len(raw_scores)
            eta_sec = remaining_pairs / rate if rate > 0 else 0.0

            if (b_idx + 1) % 5 == 0 or (b_idx + 1) == n_batches:
                print(f"  Batch {b_idx + 1}/{n_batches} ({b_end}/{len(pairs)} pairs) | Rate: {rate:.1f} pairs/s | ETA: {eta_sec:.1f}s")

        total_inference_time = time.perf_counter() - t_start
        avg_latency = total_inference_time / total_instances if total_instances > 0 else 0.0

        # Classify verdicts and build prediction records
        records = []
        y_true_binary = []
        y_pred_binary = []

        # 3-class distribution tracking
        correct_answer_verdicts = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNSUPPORTED": 0}
        hallucinated_answer_verdicts = {"SUPPORTED": 0, "CONTRADICTED": 0, "UNSUPPORTED": 0}

        for idx, inst in enumerate(instances):
            scores = raw_scores[idx]
            e = scores["entailment"]
            c = scores["contradiction"]
            n = scores["neutral"]

            # Standard calibrated 3-class decision
            if e >= config.ENTAILMENT_THRESHOLD and e > c and e > n:
                pred_verdict = "SUPPORTED"
            elif c >= config.CONTRADICTION_THRESHOLD and c > e and c > n:
                pred_verdict = "CONTRADICTED"
            else:
                pred_verdict = "UNSUPPORTED"

            # Binary mapping:
            # - SUPPORTED -> Faithful (0)
            # - CONTRADICTED or UNSUPPORTED -> Hallucinated (1)
            pred_binary = 0 if pred_verdict == "SUPPORTED" else 1
            gt_binary = inst["ground_truth_binary"]

            y_true_binary.append(gt_binary)
            y_pred_binary.append(pred_binary)

            # Track 3-class distributions
            if inst["candidate_type"] == "correct_answer":
                correct_answer_verdicts[pred_verdict] += 1
            else:
                hallucinated_answer_verdicts[pred_verdict] += 1

            is_binary_correct = (pred_binary == gt_binary)

            records.append({
                "instance_id": inst["instance_id"],
                "record_id": inst["record_id"],
                "candidate_type": inst["candidate_type"],
                "raw_question": inst["raw_question"],
                "candidate_text": inst["candidate_text"],
                "ground_truth_binary": gt_binary,
                "ground_truth_label": "Faithful" if gt_binary == 0 else "Hallucinated",
                "predicted_binary": pred_binary,
                "predicted_binary_label": "Faithful" if pred_binary == 0 else "Hallucinated",
                "predicted_verdict_3class": pred_verdict,
                "is_binary_correct": is_binary_correct,
                "entailment_score": round(float(e), 4),
                "contradiction_score": round(float(c), 4),
                "neutral_score": round(float(n), 4),
                "latency_sec": round(avg_latency, 4),
                "verification_engine": "roberta-large-mnli (calibrated)"
            })

        # Calculate Binary Hallucination Detection Metrics using scikit-learn
        # Positive class = 1 (Hallucinated)
        bin_acc = accuracy_score(y_true_binary, y_pred_binary)
        bin_prec = precision_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
        bin_rec = recall_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)
        bin_f1 = f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)

        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        summary = {
            "dataset_name": f"HaluEval Benchmark (QA Subset - {run_mode})",
            "dataset_path": csv_path,
            "total_records": n_records,
            "total_candidate_answers": total_instances,
            "binary_metrics": {
                "accuracy": round(float(bin_acc), 4),
                "precision": round(float(bin_prec), 4),
                "recall": round(float(bin_rec), 4),
                "f1_score": round(float(bin_f1), 4),
                "specificity": round(float(specificity), 4),
                "false_positive_rate": round(float(fpr), 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "confusion_matrix": {
                    "labels": ["Faithful (0)", "Hallucinated (1)"],
                    "matrix": cm.tolist()
                }
            },
            "three_class_distribution": {
                "correct_answers": {
                    "total": n_records,
                    "supported": correct_answer_verdicts["SUPPORTED"],
                    "contradicted": correct_answer_verdicts["CONTRADICTED"],
                    "unsupported": correct_answer_verdicts["UNSUPPORTED"],
                    "supported_rate": round(correct_answer_verdicts["SUPPORTED"] / n_records, 4) if n_records > 0 else 0.0
                },
                "hallucinated_answers": {
                    "total": n_records,
                    "supported": hallucinated_answer_verdicts["SUPPORTED"],
                    "contradicted": hallucinated_answer_verdicts["CONTRADICTED"],
                    "unsupported": hallucinated_answer_verdicts["UNSUPPORTED"],
                    "contradicted_rate": round(hallucinated_answer_verdicts["CONTRADICTED"] / n_records, 4) if n_records > 0 else 0.0,
                    "unsupported_rate": round(hallucinated_answer_verdicts["UNSUPPORTED"] / n_records, 4) if n_records > 0 else 0.0,
                    "hallucination_detection_rate": round((hallucinated_answer_verdicts["CONTRADICTED"] + hallucinated_answer_verdicts["UNSUPPORTED"]) / n_records, 4) if n_records > 0 else 0.0
                }
            },
            "timing": {
                "total_inference_time_sec": round(float(total_inference_time), 4),
                "avg_latency_per_candidate_sec": round(float(avg_latency), 4),
                "throughput_pairs_per_sec": round(float(total_instances / total_inference_time), 2) if total_inference_time > 0 else 0.0
            }
        }

        # Save outputs
        pred_df = pd.DataFrame(records)
        pred_csv = self.output_dir / f"{file_prefix}_predictions.csv"
        pred_json = self.output_dir / f"{file_prefix}_predictions.json"
        metrics_json = self.output_dir / f"{file_prefix}_metrics.json"
        report_md = self.output_dir / f"{file_prefix}_report.md"

        pred_df.to_csv(pred_csv, index=False)
        with open(pred_json, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self._generate_halueval_markdown_report(report_md, summary, records)

        print(f"\n[HaluEval Evaluation Complete]")
        print(f"  Binary Accuracy:  {summary['binary_metrics']['accuracy']*100:.2f}%")
        print(f"  Precision:        {summary['binary_metrics']['precision']:.4f}")
        print(f"  Recall:           {summary['binary_metrics']['recall']:.4f}")
        print(f"  F1-Score:         {summary['binary_metrics']['f1_score']:.4f}")
        print(f"  FPR:              {summary['binary_metrics']['false_positive_rate']*100:.2f}%")
        print(f"  Total Duration:   {total_inference_time:.2f}s ({summary['timing']['throughput_pairs_per_sec']} pairs/sec)")
        print(f"  Saved outputs to: {self.output_dir}\n")

        return {
            "summary": summary,
            "records": records,
            "saved_files": {
                "predictions_csv": str(pred_csv),
                "predictions_json": str(pred_json),
                "metrics_json": str(metrics_json),
                "report_md": str(report_md)
            }
        }

    def _generate_halueval_markdown_report(self, path: Path, summary: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
        """Generates comprehensive HaluEval markdown report."""
        bm = summary["binary_metrics"]
        cm = bm["confusion_matrix"]["matrix"]
        tc = summary["three_class_distribution"]

        md = [
            f"# HaluEval Benchmark Evaluation Report",
            f"",
            f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Dataset:** `{summary['dataset_name']}`",
            f"**Verification Engine:** `roberta-large-mnli` (Calibrated Thresholds)",
            f"",
            f"---",
            f"",
            f"## 1. Binary Hallucination Detection Metrics",
            f"",
            f"Evaluating whether candidate responses are faithful (0) or hallucinated (1) relative to reference knowledge:",
            f"",
            f"| Metric | Value | Interpretation |",
            f"| :--- | :--- | :--- |",
            f"| **Accuracy** | **{bm['accuracy']*100:.2f}%** | Overall classification correctness |",
            f"| **Precision** | **{bm['precision']:.4f}** | Probability that a flagged response is truly hallucinated |",
            f"| **Recall (Sensitivity)** | **{bm['recall']:.4f}** | Proportion of true hallucinations successfully flagged |",
            f"| **F1-Score** | **{bm['f1_score']:.4f}** | Harmonic mean of precision and recall |",
            f"| **Specificity** | **{bm['specificity']*100:.2f}%** | Proportion of faithful answers correctly verified |",
            f"| **False Positive Rate (FPR)** | **{bm['false_positive_rate']*100:.2f}%** | Rate of faithful answers falsely flagged as hallucinated |",
            f"",
            f"### Binary Confusion Matrix",
            f"",
            f"| True \\ Predicted | Predicted Faithful (0) | Predicted Hallucinated (1) |",
            f"| :--- | :--- | :--- |",
            f"| **Actual Faithful (0)** | **TN = {cm[0][0]}** | FP = {cm[0][1]} |",
            f"| **Actual Hallucinated (1)** | FN = {cm[1][0]} | **TP = {cm[1][1]}** |",
            f"",
            f"---",
            f"",
            f"## 2. Fine-Grained Three-Class Breakdown",
            f"",
            f"HaluEval provides binary ground-truth (Faithful vs Hallucinated). The detector decomposes predictions into 3 honest verdicts:",
            f"",
            f"### A. Ground-Truth Faithful Answers (`correct_answer`):",
            f"- **Supported (Correctly verified):** {tc['correct_answers']['supported']} ({tc['correct_answers']['supported_rate']*100:.1f}%)",
            f"- **Unsupported (Unwarranted skepticism):** {tc['correct_answers']['unsupported']}",
            f"- **Contradicted (False contradiction):** {tc['correct_answers']['contradicted']}",
            f"",
            f"### B. Ground-Truth Hallucinated Answers (`hallucinated_answer`):",
            f"- **Contradicted (Direct factual conflict):** {tc['hallucinated_answers']['contradicted']} ({tc['hallucinated_answers']['contradicted_rate']*100:.1f}%)",
            f"- **Unsupported (Ungrounded fabrication / absence of evidence):** {tc['hallucinated_answers']['unsupported']} ({tc['hallucinated_answers']['unsupported_rate']*100:.1f}%)",
            f"- **Supported (Missed hallucination / False Negative):** {tc['hallucinated_answers']['supported']}",
            f"- **Total Hallucinations Detected:** {tc['hallucinated_answers']['contradicted'] + tc['hallucinated_answers']['unsupported']} ({tc['hallucinated_answers']['hallucination_detection_rate']*100:.1f}%)",
            f"",
            f"---",
            f"",
            f"## 3. Latency & Throughput",
            f"",
            f"- **Total Evaluated Candidates:** {summary['total_candidate_answers']} pairs",
            f"- **Total Inference Time:** {summary['timing']['total_inference_time_sec']:.2f} seconds",
            f"- **Average Latency per Candidate:** {summary['timing']['avg_latency_per_candidate_sec']:.4f} seconds",
            f"- **Throughput:** {summary['timing']['throughput_pairs_per_sec']} pairs/second on CPU",
            f""
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
