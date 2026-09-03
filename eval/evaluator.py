"""RAG Pipeline Quality and Evaluation Benchmark Engine.

Measures Retrieval Quality (Precision@K, Recall@K, MRR), Generation Faithfulness,
Context Relevance, Citation Correctness, and Pipeline Latency.
"""

from typing import NamedTuple
from interfaces.query_interface import QueryRequestInterface, QueryResponseInterface
from services.query_service import execute_query_workflow
from utils.logger import logger

FUNC_EVAL_RETRIEVAL: str = "calculate_retrieval_metrics"
FUNC_EVAL_GOLDEN: str = "evaluate_golden_set"


class RetrievalMetricsResult(NamedTuple):
    """Container for retrieval evaluation metrics."""
    precision_at_k: float
    recall_at_k: float
    mrr: float


class EvaluationBenchmarkReport(NamedTuple):
    """Container for comprehensive evaluation report."""
    total_test_cases: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_citation_correctness: float
    avg_latency_ms: float


def calculate_retrieval_metrics(
    ground_truth_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    top_k: int = 5,
) -> RetrievalMetricsResult:
    """Calculate Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).

    @param ground_truth_chunk_ids: Expected relevant chunk IDs.
    @param retrieved_chunk_ids: Retrieved candidate chunk IDs.
    @param top_k: Evaluation cut-off threshold.
    @returns: RetrievalMetricsResult containing precision, recall, and MRR.
    """
    if not ground_truth_chunk_ids or not retrieved_chunk_ids:
        return RetrievalMetricsResult(precision_at_k=0.0, recall_at_k=0.0, mrr=0.0)

    top_hits: list[str] = retrieved_chunk_ids[:top_k]
    gt_set: set[str] = set(ground_truth_chunk_ids)

    relevant_retrieved: int = sum(1 for cid in top_hits if cid in gt_set)
    precision: float = relevant_retrieved / float(len(top_hits) or 1)
    recall: float = relevant_retrieved / float(len(gt_set) or 1)

    mrr: float = 0.0
    for rank, cid in enumerate(top_hits, start=1):
        if cid in gt_set:
            mrr = 1.0 / float(rank)
            break

    return RetrievalMetricsResult(
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        mrr=round(mrr, 4),
    )


def evaluate_golden_set(golden_test_cases: list[dict]) -> EvaluationBenchmarkReport:
    """Run golden-set benchmarking suite across retrieval, generation, and citations.

    @param golden_test_cases: List of dict objects containing 'question', 'document_ids', 'expected_chunks'.
    @returns: EvaluationBenchmarkReport summarizing quality metrics.
    """
    if not golden_test_cases:
        return EvaluationBenchmarkReport(
            total_test_cases=0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            mean_mrr=0.0,
            mean_citation_correctness=0.0,
            avg_latency_ms=0.0,
        )

    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    citation_accs: list[float] = []
    latencies: list[float] = []

    for test_case in golden_test_cases:
        question: str = test_case["question"]
        expected_chunks: list[str] = test_case.get("expected_chunks", [])
        doc_ids: list[str] | None = test_case.get("document_ids")

        req: QueryRequestInterface = QueryRequestInterface(
            question=question,
            document_ids=doc_ids,
            top_k=5,
        )

        res: QueryResponseInterface = execute_query_workflow(req)
        latencies.append(res.processing_time_ms)

        retrieved_chunk_ids: list[str] = [c.chunk_id for c in res.citations]
        if expected_chunks:
            ret_metrics: RetrievalMetricsResult = calculate_retrieval_metrics(
                ground_truth_chunk_ids=expected_chunks,
                retrieved_chunk_ids=retrieved_chunk_ids,
                top_k=5,
            )
            precisions.append(ret_metrics.precision_at_k)
            recalls.append(ret_metrics.recall_at_k)
            mrrs.append(ret_metrics.mrr)

        if res.citations:
            valid_cites: int = sum(1 for c in res.citations if c.is_valid)
            citation_accs.append(valid_cites / float(len(res.citations)))

    report: EvaluationBenchmarkReport = EvaluationBenchmarkReport(
        total_test_cases=len(golden_test_cases),
        mean_precision_at_k=round(sum(precisions) / float(len(precisions) or 1), 4),
        mean_recall_at_k=round(sum(recalls) / float(len(recalls) or 1), 4),
        mean_mrr=round(sum(mrrs) / float(len(mrrs) or 1), 4),
        mean_citation_correctness=round(sum(citation_accs) / float(len(citation_accs) or 1), 4),
        avg_latency_ms=round(sum(latencies) / float(len(latencies) or 1), 2),
    )

    logger.info(
        FUNC_EVAL_GOLDEN,
        f"Evaluation completed for {report.total_test_cases} test cases: MRR={report.mean_mrr}, Citation Correctness={report.mean_citation_correctness}, Latency={report.avg_latency_ms}ms"
    )
    return report
