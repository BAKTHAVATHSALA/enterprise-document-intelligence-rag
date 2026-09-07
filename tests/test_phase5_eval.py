"""Unit and Integration Tests for Phase 5 Step 5.2 Evaluation Benchmark Engine.

Verifies:
- Structure and content of golden_dataset.json (20 test cases, 5 per PDF).
- Math correctness of NDCG@K calculation.
- Percentile latency calculation (p50, p90, p99).
- Channel-specific MRR calculation (Vector, BM25, Graph, RRF).
- End-to-end golden dataset benchmarking.
"""

import json
import os
import pytest
from eval.evaluator import (
    calculate_ndcg_at_k,
    calculate_percentile_latencies,
    calculate_retrieval_metrics,
    evaluate_channel_mrr,
    evaluate_golden_set,
    EvaluationBenchmarkReport,
    RetrievalMetricsResult,
)


def test_golden_dataset_integrity():
    """Verify that golden_dataset.json exists, contains 20 items, 5 per PDF, with required fields."""
    dataset_path = os.path.join("eval", "golden_dataset.json")
    assert os.path.exists(dataset_path), f"Missing golden dataset file at {dataset_path}"

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert isinstance(cases, list)
    assert len(cases) == 20, f"Expected 20 test cases, got {len(cases)}"

    # Group by source PDF
    pdf_counts = {}
    required_keys = {"id", "question", "source_pdf", "page_number", "question_type", "expected_answer", "document_ids"}

    for idx, case in enumerate(cases, start=1):
        missing = required_keys - set(case.keys())
        assert not missing, f"Test case {idx} ({case.get('id')}) missing keys: {missing}"

        pdf = case["source_pdf"]
        pdf_counts[pdf] = pdf_counts.get(pdf, 0) + 1

    expected_pdfs = [
        "NIST.CSWP.01162020.pdf",
        "NIST.CSWP.29.pdf",
        "NIST.SP.1299.pdf.pdf",
        "NIST.SP.1302.pdf.pdf",
    ]

    for pdf in expected_pdfs:
        assert pdf in pdf_counts, f"PDF {pdf} not found in golden dataset"
        assert pdf_counts[pdf] == 5, f"Expected 5 test cases for {pdf}, found {pdf_counts[pdf]}"


def test_ndcg_at_k_perfect_ranking():
    """NDCG@K should be 1.0 when ground truth items match retrieved items in rank order."""
    gt = ["chunk_a", "chunk_b"]
    retrieved = ["chunk_a", "chunk_b", "chunk_c"]
    score = calculate_ndcg_at_k(ground_truth_chunk_ids=gt, retrieved_chunk_ids=retrieved, top_k=5)
    assert score == 1.0


def test_ndcg_at_k_partial_and_zero_ranking():
    """NDCG@K should produce expected discounted values for partial matches and 0.0 for zero matches."""
    gt = ["chunk_a"]
    
    # Relevant item at rank 2 vs rank 1
    retrieved_rank2 = ["chunk_x", "chunk_a", "chunk_y"]
    score_rank2 = calculate_ndcg_at_k(ground_truth_chunk_ids=gt, retrieved_chunk_ids=retrieved_rank2, top_k=5)
    assert 0.0 < score_rank2 < 1.0

    # Zero hits
    retrieved_zero = ["chunk_x", "chunk_y"]
    score_zero = calculate_ndcg_at_k(ground_truth_chunk_ids=gt, retrieved_chunk_ids=retrieved_zero, top_k=5)
    assert score_zero == 0.0

    # Empty inputs
    assert calculate_ndcg_at_k([], ["chunk_a"]) == 0.0
    assert calculate_ndcg_at_k(["chunk_a"], []) == 0.0


def test_percentile_latencies():
    """Verify p50, p90, and p99 latency calculations."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    p_dict = calculate_percentile_latencies(latencies)

    assert "p50" in p_dict
    assert "p90" in p_dict
    assert "p99" in p_dict

    assert p_dict["p50"] == 55.0
    assert p_dict["p90"] == 91.0
    assert p_dict["p99"] == 99.1

    # Single latency edge case
    single_p = calculate_percentile_latencies([42.5])
    assert single_p == {"p50": 42.5, "p90": 42.5, "p99": 42.5}

    # Empty latency list edge case
    empty_p = calculate_percentile_latencies([])
    assert empty_p == {"p50": 0.0, "p90": 0.0, "p99": 0.0}


def test_channel_mrr():
    """Verify channel-specific MRR calculation."""
    gt = ["chunk_target"]
    
    mrr_rank1 = evaluate_channel_mrr(gt, ["chunk_target", "chunk_other"])
    assert mrr_rank1 == 1.0

    mrr_rank3 = evaluate_channel_mrr(gt, ["c1", "c2", "chunk_target"])
    assert mrr_rank3 == 0.3333

    mrr_none = evaluate_channel_mrr(gt, ["c1", "c2"])
    assert mrr_none == 0.0


def test_calculate_retrieval_metrics_extended():
    """Verify extended RetrievalMetricsResult container returns precision, recall, MRR, and NDCG@K."""
    gt = ["c1", "c2"]
    retrieved = ["c1", "x1", "c2", "x2"]
    res = calculate_retrieval_metrics(ground_truth_chunk_ids=gt, retrieved_chunk_ids=retrieved, top_k=5)

    assert isinstance(res, RetrievalMetricsResult)
    assert res.precision_at_k == 0.5
    assert res.recall_at_k == 1.0
    assert res.mrr == 1.0
    assert res.ndcg_at_k > 0.0


def test_evaluate_golden_set_execution(monkeypatch):
    """Integration test for evaluate_golden_set using mock workflow output."""
    from interfaces.query_interface import QueryResponseInterface, CitationInterface, CitationValidationSummaryInterface

    def mock_workflow(req):
        return QueryResponseInterface(
            question=req.question,
            answer="Mocked grounded answer.",
            citations=[
                CitationInterface(
                    document_id="NIST.CSWP.01162020.pdf",
                    page=10,
                    section="1.0",
                    chunk_id="chunk_gold_1",
                    is_valid=True
                )
            ],
            validation_summary=CitationValidationSummaryInterface(
                total_citations=1, valid_count=1, invalid_count=0, fabricated_count=0, missing_count=0, is_fully_validated=True
            ),
            confidence_score=0.95,
            processing_time_ms=120.0,
        )

    monkeypatch.setattr("eval.evaluator.execute_query_workflow", mock_workflow)

    test_cases = [
        {
            "question": "Sample question 1?",
            "document_ids": ["NIST.CSWP.01162020.pdf"],
            "expected_chunks": ["chunk_gold_1"],
            "vector_chunks": ["chunk_gold_1"],
            "bm25_chunks": ["chunk_other"],
            "graph_chunks": [],
            "rrf_chunks": ["chunk_gold_1"],
        }
    ]

    report = evaluate_golden_set(test_cases)
    assert isinstance(report, EvaluationBenchmarkReport)
    assert report.total_test_cases == 1
    assert report.mean_precision_at_k == 1.0
    assert report.mean_recall_at_k == 1.0
    assert report.mean_mrr == 1.0
    assert report.mean_ndcg_at_k == 1.0
    assert report.vector_mrr == 1.0
    assert report.bm25_mrr == 0.0
    assert report.avg_latency_ms == 120.0
    assert report.latency_p50_ms == 120.0
