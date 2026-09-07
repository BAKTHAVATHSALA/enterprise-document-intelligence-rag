"""Focused Unit and Integration Tests for Phase 5 Step 5.3 RAGAS Integration.

Verifies:
- Mock/offline mode toggling and safety (zero OpenAI API calls in mock mode).
- RagasMetricsResult container and answer_relevancy alias.
- Deterministic heuristic metric calculations (Faithfulness, Answer Relevance, Context Precision, Context Recall).
- Resilience to empty and edge case inputs (insufficient evidence, empty contexts).
- Golden dataset schema compatibility with RAGAS SingleTurnSample / EvaluationDataset.
- End-to-end integration with evaluate_golden_set and evaluate_golden_dataset_file.
- No hardcoded API keys in evaluation code.
"""

import json
import os
from unittest.mock import MagicMock
import pytest

from eval.ragas_evaluator import (
    RagasMetricsResult,
    evaluate_ragas_batch,
    evaluate_ragas_mock,
    evaluate_ragas_sample,
    init_ragas_judge,
    is_mock_ragas_mode,
    set_mock_ragas_mode,
)
from eval.evaluator import (
    EvaluationBenchmarkReport,
    evaluate_golden_dataset_file,
    evaluate_golden_set,
)


@pytest.fixture(autouse=True)
def ensure_mock_mode():
    """Ensure mock mode is active by default for all test runs."""
    set_mock_ragas_mode(True)
    yield
    set_mock_ragas_mode(True)


def test_ragas_mock_mode_toggling():
    """Verify mock mode toggling behavior."""
    set_mock_ragas_mode(True)
    assert is_mock_ragas_mode() is True

    set_mock_ragas_mode(False)
    # If OPENAI_API_KEY is not in env, is_mock_ragas_mode defaults to True
    # If OPENAI_API_KEY is present, it will be False
    set_mock_ragas_mode(True)
    assert is_mock_ragas_mode() is True


def test_ragas_metrics_result_container():
    """Verify RagasMetricsResult container and property alias."""
    res = RagasMetricsResult(
        faithfulness=0.85,
        answer_relevance=0.92,
        context_precision=0.78,
        context_recall=0.88,
    )
    assert res.faithfulness == 0.85
    assert res.answer_relevance == 0.92
    assert res.answer_relevancy == 0.92  # Alias
    assert res.context_precision == 0.78
    assert res.context_recall == 0.88


def test_ragas_mock_metrics_calculation():
    """Verify mock metric calculation produces bounded valid scores."""
    question = "What are the core functions of the NIST Privacy Framework?"
    contexts = [
        "The NIST Privacy Framework defines five Core Functions: Identify-P, Govern-P, Control-P, Communicate-P, and Protect-P.",
        "Implementation Tiers describe how an organization manages privacy risk.",
    ]
    answer = "The five Core Functions of the NIST Privacy Framework are Identify-P, Govern-P, Control-P, Communicate-P, and Protect-P."
    ground_truth = "Identify-P, Govern-P, Control-P, Communicate-P, and Protect-P."

    res = evaluate_ragas_mock(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )

    assert isinstance(res, RagasMetricsResult)
    assert 0.0 <= res.faithfulness <= 1.0
    assert 0.0 <= res.answer_relevance <= 1.0
    assert 0.0 <= res.context_precision <= 1.0
    assert 0.0 <= res.context_recall <= 1.0

    # Strong grounding should yield high scores
    assert res.faithfulness > 0.5
    assert res.answer_relevance > 0.5
    assert res.context_precision > 0.5
    assert res.context_recall > 0.5


def test_ragas_mock_edge_cases():
    """Verify handling of empty inputs and insufficient evidence."""
    # 1. Empty answer
    res_empty_ans = evaluate_ragas_mock(
        question="What is X?",
        answer="",
        contexts=["Some context"],
        ground_truth="Some truth",
    )
    assert res_empty_ans.faithfulness == 0.0
    assert res_empty_ans.answer_relevance == 0.0

    # 2. Empty contexts with insufficient evidence answer
    res_no_ctx = evaluate_ragas_mock(
        question="What is X?",
        answer="Insufficient evidence in ingested documents to answer the question.",
        contexts=[],
        ground_truth="Some truth",
    )
    assert res_no_ctx.faithfulness == 1.0  # Faithful refusal
    assert res_no_ctx.context_precision == 0.0
    assert res_no_ctx.context_recall == 0.0

    # 3. Completely empty sample batch
    res_empty_batch = evaluate_ragas_batch([], use_mock=True)
    assert res_empty_batch.faithfulness == 0.0
    assert res_empty_batch.answer_relevance == 0.0


def test_mock_mode_makes_no_openai_calls(monkeypatch):
    """Verify that in mock mode, OpenAI client is never called."""
    mock_openai = MagicMock()
    monkeypatch.setattr("eval.ragas_evaluator.OpenAI", mock_openai)

    set_mock_ragas_mode(True)
    res = evaluate_ragas_sample(
        question="Question 1",
        answer="Answer 1",
        contexts=["Context 1"],
        ground_truth="Truth 1",
    )

    assert isinstance(res, RagasMetricsResult)
    mock_openai.assert_not_called()


def test_golden_dataset_compatibility_with_ragas():
    """Verify golden_dataset.json contains necessary fields for RAGAS evaluation."""
    dataset_path = os.path.join("eval", "golden_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    samples = []
    for c in cases:
        samples.append({
            "question": c["question"],
            "answer": c.get("expected_answer", ""),
            "contexts": [f"Source: {c['source_pdf']}, Page {c['page_number']}: {c['expected_answer']}"],
            "ground_truth": c["expected_answer"],
        })

    batch_res = evaluate_ragas_batch(samples[:5], use_mock=True)
    assert isinstance(batch_res, RagasMetricsResult)
    assert batch_res.faithfulness > 0.0
    assert batch_res.answer_relevance > 0.0
    assert batch_res.context_recall > 0.0


def test_evaluate_golden_set_with_ragas_integration(monkeypatch):
    """Integration test for evaluate_golden_set producing both retrieval and RAGAS metrics."""
    from interfaces.query_interface import (
        QueryResponseInterface,
        CitationInterface,
        CitationValidationSummaryInterface,
    )

    def mock_workflow(req):
        return QueryResponseInterface(
            question=req.question,
            answer="The five Core Functions of the NIST Privacy Framework are Identify-P, Govern-P, Control-P, Communicate-P, and Protect-P.",
            citations=[
                CitationInterface(
                    document_id="NIST.CSWP.01162020.pdf",
                    page=10,
                    section="1.0",
                    chunk_id="chunk_gold_1",
                    snippet="Core Functions: Identify-P, Govern-P, Control-P, Communicate-P, Protect-P.",
                    is_valid=True,
                )
            ],
            validation_summary=CitationValidationSummaryInterface(
                total_citations=1, valid_count=1, invalid_count=0, fabricated_count=0, missing_count=0, is_fully_validated=True
            ),
            confidence_score=0.98,
            processing_time_ms=115.0,
        )

    monkeypatch.setattr("eval.evaluator.execute_query_workflow", mock_workflow)

    test_cases = [
        {
            "question": "What are the five Core Functions of the NIST Privacy Framework Version 1.0?",
            "document_ids": ["NIST.CSWP.01162020.pdf"],
            "expected_chunks": ["chunk_gold_1"],
            "expected_answer": "Identify-P, Govern-P, Control-P, Communicate-P, Protect-P.",
            "vector_chunks": ["chunk_gold_1"],
            "bm25_chunks": ["chunk_gold_1"],
            "graph_chunks": [],
            "rrf_chunks": ["chunk_gold_1"],
        }
    ]

    report = evaluate_golden_set(test_cases, include_ragas=True)
    assert isinstance(report, EvaluationBenchmarkReport)
    assert report.total_test_cases == 1
    assert report.mean_precision_at_k == 1.0
    assert report.mean_recall_at_k == 1.0
    assert report.mean_mrr == 1.0
    assert report.mean_ndcg_at_k == 1.0

    # RAGAS metrics populated
    assert report.ragas_faithfulness > 0.0
    assert report.ragas_answer_relevance > 0.0
    assert report.ragas_context_precision > 0.0
    assert report.ragas_context_recall > 0.0


def test_evaluate_golden_dataset_file_execution(monkeypatch):
    """Verify evaluate_golden_dataset_file loads from disk and benchmarks."""
    from interfaces.query_interface import (
        QueryResponseInterface,
        CitationInterface,
        CitationValidationSummaryInterface,
    )

    def mock_workflow(req):
        return QueryResponseInterface(
            question=req.question,
            answer="Mocked grounded answer for golden dataset query.",
            citations=[
                CitationInterface(
                    document_id="NIST.CSWP.01162020.pdf",
                    page=1,
                    section="Intro",
                    chunk_id="chk_01",
                    snippet="Excerpt supporting claim.",
                    is_valid=True,
                )
            ],
            validation_summary=CitationValidationSummaryInterface(
                total_citations=1, valid_count=1, invalid_count=0, fabricated_count=0, missing_count=0, is_fully_validated=True
            ),
            confidence_score=0.9,
            processing_time_ms=80.0,
        )

    monkeypatch.setattr("eval.evaluator.execute_query_workflow", mock_workflow)

    report = evaluate_golden_dataset_file(max_cases=2, include_ragas=True)
    assert isinstance(report, EvaluationBenchmarkReport)
    assert report.total_test_cases == 2
    assert report.ragas_faithfulness >= 0.0
    assert report.ragas_answer_relevance >= 0.0


def test_no_api_keys_in_code_or_ragas_leakage():
    """Verify that no OpenAI API keys are hardcoded in eval files."""
    eval_dir = "eval"
    for fname in ["ragas_evaluator.py", "evaluator.py"]:
        fpath = os.path.join(eval_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            assert "sk-" not in content, f"Possible hardcoded API key in {fname}"
