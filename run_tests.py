"""Phase 1 Test Runner Script.

Executes all unit, integration, and golden-set benchmark tests across the platform.
"""

import sys
import time
from tests.test_pipeline import (
    test_1_valid_pdf_upload_success,
    test_2_docling_pdf_processing,
    test_3_multiple_pages_recognized,
    test_4_headings_and_sections_preserved,
    test_5_pii_redacted_from_pdf,
    test_6_entities_extracted,
    test_7_chunks_contain_full_metadata,
    test_8_invalid_file_type_rejected,
    test_9_empty_file_rejected,
    test_10_corrupted_pdf_rejected_gracefully,
    test_11_distinct_pdfs_receive_different_ids,
    test_security_prompt_injection,
    test_evaluator_metrics,
)
from eval.evaluator import evaluate_golden_set, EvaluationBenchmarkReport

TEST_SUITE = [
    ("TEST 1: Valid PDF Upload Success", test_1_valid_pdf_upload_success),
    ("TEST 2: Docling PDF Parser Processing", test_2_docling_pdf_processing),
    ("TEST 3: Multiple Pages Recognition", test_3_multiple_pages_recognized),
    ("TEST 4: Headings and Section Preservation", test_4_headings_and_sections_preserved),
    ("TEST 5: PII Redaction From PDF", test_5_pii_redacted_from_pdf),
    ("TEST 6: Entity Extraction", test_6_entities_extracted),
    ("TEST 7: Chunk Metadata Lineage Retention", test_7_chunks_contain_full_metadata),
    ("TEST 8: Invalid File Type Rejection", test_8_invalid_file_type_rejected),
    ("TEST 9: Empty File Rejection", test_9_empty_file_rejected),
    ("TEST 10: Corrupted PDF Graceful Rejection", test_10_corrupted_pdf_rejected_gracefully),
    ("TEST 11: Distinct PDF Isolation", test_11_distinct_pdfs_receive_different_ids),
    ("Prompt Injection Security Defense", test_security_prompt_injection),
    ("Evaluator Metrics Calculator", test_evaluator_metrics),
]


def run_all_tests() -> None:
    """Run all automated test cases and golden-set benchmark suite."""
    print("\n=======================================================")
    print("  Phase 1 — Real PDF Document Ingestion Test Suite")
    print("=======================================================\n")

    passed_count: int = 0
    start_time: float = time.perf_counter()

    for name, test_func in TEST_SUITE:
        try:
            test_func()
            print(f"  [PASS] - {name}")
            passed_count += 1
        except Exception as exc:
            print(f"  [FAIL] - {name}: {exc}")

    # Golden Set Benchmark Evaluation
    print("\n-------------------------------------------------------")
    print("  Running Golden-Set Quality Evaluation Benchmark...")
    print("-------------------------------------------------------")

    golden_cases = [
        {
            "question": "What is the notice period for Contract #123?",
            "expected_chunks": [],
        },
        {
            "question": "What is the termination notice period for Contract #456?",
            "expected_chunks": [],
        },
    ]

    report: EvaluationBenchmarkReport = evaluate_golden_set(golden_cases)

    duration: float = round(time.perf_counter() - start_time, 2)
    print("\n=======================================================")
    print(f"  RESULTS: {passed_count}/{len(TEST_SUITE)} Tests Passed ({duration}s)")
    print(f"  Golden-Set Evaluated: {report.total_test_cases} Queries")
    print(f"  Citation Correctness: {report.mean_citation_correctness * 100}%")
    print(f"  Average Query Latency: {report.avg_latency_ms}ms")
    print("=======================================================\n")

    if passed_count < len(TEST_SUITE):
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
