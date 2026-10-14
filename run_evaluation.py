"""
CLI Runner for Dual-Dataset Evaluation
Supports running:
1. Custom Controlled Evaluation (60 examples)
2. HaluEval Sample (e.g. 50 records / 100 candidate answers)
3. HaluEval Full (2,000 records / 4,000 candidate answers)
4. All benchmarks sequentially reusing the loaded NLI model.
"""

import sys
import argparse
import time
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from modules.nli_verifier import NLIVerifier
from modules.evaluation import CustomDatasetEvaluator, HaluEvalEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run Dual-Dataset Evaluation for LLM Hallucination Detector")
    parser.add_argument(
        "--mode",
        choices=["custom", "halueval-sample", "halueval-full", "all-sample", "all-full"],
        default="all-sample",
        help="Evaluation mode to execute"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of HaluEval records to sample for sample runs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for PyTorch inference (default: 32)"
    )
    parser.add_argument(
        "--custom-csv",
        type=str,
        default="hallucination_detector_custom_eval.csv",
        help="Path to custom evaluation CSV"
    )
    parser.add_argument(
        "--halueval-csv",
        type=str,
        default="halueval.csv",
        help="Path to HaluEval CSV"
    )

    args = parser.parse_args()

    print("=================================================================")
    print("DUAL-DATASET EVALUATION RUNNER")
    print("LLM Hallucination Detector Offline Verification Benchmarking")
    print("=================================================================")
    print(f"Selected Mode:        {args.mode}")
    print(f"Custom CSV:           {args.custom_csv}")
    print(f"HaluEval CSV:         {args.halueval_csv}")
    print(f"HaluEval Sample Size: {args.sample_size}")
    print(f"Batch Size:           {args.batch_size}")
    print("-----------------------------------------------------------------\n")

    # Initialize NLI model once and reuse across datasets
    t_load_0 = time.perf_counter()
    print("[1/3] Initializing NLI Verifier (roberta-large-mnli)...")
    verifier = NLIVerifier()
    t_load = time.perf_counter() - t_load_0
    print(f"  Model loaded into memory in {t_load:.2f}s.\n")

    # 1. Run Custom Dataset Evaluation if requested
    if args.mode in ["custom", "all-sample", "all-full"]:
        print("[2/3] Executing Custom Dataset Evaluation (60 examples)...")
        custom_evaluator = CustomDatasetEvaluator(verifier=verifier)
        custom_res = custom_evaluator.evaluate(
            csv_path=args.custom_csv,
            batch_size=args.batch_size
        )
        print("  Custom evaluation reports saved successfully.\n")

    # 2. Run HaluEval Evaluation if requested
    if args.mode in ["halueval-sample", "all-sample"]:
        print(f"[3/3] Executing HaluEval Sample Evaluation ({args.sample_size} records)...")
        halu_evaluator = HaluEvalEvaluator(verifier=verifier)
        halu_res = halu_evaluator.evaluate(
            csv_path=args.halueval_csv,
            sample_size=args.sample_size,
            batch_size=args.batch_size
        )
        print("  HaluEval sample reports saved successfully.\n")
    elif args.mode in ["halueval-full", "all-full"]:
        print(f"[3/3] Executing Full HaluEval Evaluation (all records)...")
        halu_evaluator = HaluEvalEvaluator(verifier=verifier)
        halu_res = halu_evaluator.evaluate(
            csv_path=args.halueval_csv,
            sample_size=None,
            batch_size=args.batch_size
        )
        print("  Full HaluEval reports saved successfully.\n")

    print("=================================================================")
    print("ALL EVALUATION RUNS COMPLETED SUCCESSFULLY!")
    print("All output files saved to data/reports/")
    print("=================================================================")


if __name__ == "__main__":
    main()
