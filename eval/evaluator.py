"""RAG Pipeline Quality and Evaluation Benchmark Engine.

Measures Retrieval Quality (Precision@K, Recall@K, MRR, NDCG@K), Multi-Retriever Channel Breakdown (Vector, BM25, Graph, RRF),
Generation Faithfulness, Context Relevance, Citation Correctness, and Latency Percentiles (p50, p90, p99).
"""

import os
import json
import math
from typing import NamedTuple, Optional
from interfaces.query_interface import QueryRequestInterface, QueryResponseInterface
from services.query_service import execute_query_workflow
from utils.logger import logger

from eval.ragas_evaluator import evaluate_ragas_batch, RagasMetricsResult

FUNC_EVAL_RETRIEVAL: str = "calculate_retrieval_metrics"
FUNC_EVAL_GOLDEN: str = "evaluate_golden_set"


class RetrievalMetricsResult(NamedTuple):
    """Container for retrieval evaluation metrics."""
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


class MultiRetrieverChannelMetrics(NamedTuple):
    """Container for individual retrieval channel metrics."""
    vector_mrr: float
    bm25_mrr: float
    graph_mrr: float
    rrf_mrr: float


class EvaluationBenchmarkReport(NamedTuple):
    """Container for comprehensive evaluation report."""
    total_test_cases: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    mean_citation_correctness: float
    avg_latency_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    vector_mrr: float
    bm25_mrr: float
    graph_mrr: float
    rrf_mrr: float
    ragas_faithfulness: float = 0.0
    ragas_answer_relevance: float = 0.0
    ragas_context_precision: float = 0.0
    ragas_context_recall: float = 0.0


def calculate_ndcg_at_k(
    ground_truth_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    top_k: int = 5,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K).

    @param ground_truth_chunk_ids: Expected relevant chunk IDs.
    @param retrieved_chunk_ids: Retrieved candidate chunk IDs.
    @param top_k: Evaluation cut-off threshold.
    @returns: NDCG@K score rounded to 4 decimal places.
    """
    if not ground_truth_chunk_ids or not retrieved_chunk_ids:
        return 0.0

    top_hits: list[str] = retrieved_chunk_ids[:top_k]
    gt_set: set[str] = set(ground_truth_chunk_ids)

    dcg: float = 0.0
    for rank, cid in enumerate(top_hits, start=1):
        if cid in gt_set:
            dcg += 1.0 / math.log2(rank + 1)

    idcg: float = 0.0
    ideal_hits_count = min(len(gt_set), top_k)
    for rank in range(1, ideal_hits_count + 1):
        idcg += 1.0 / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0

    return round(dcg / idcg, 4)


def calculate_percentile_latencies(latencies: list[float]) -> dict[str, float]:
    """Calculate p50, p90, and p99 percentile latencies in milliseconds.

    @param latencies: List of latency values in milliseconds.
    @returns: Dictionary containing 'p50', 'p90', and 'p99'.
    """
    if not latencies:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}

    sorted_lat: list[float] = sorted(latencies)
    n: int = len(sorted_lat)

    def get_percentile(p: float) -> float:
        if n == 1:
            return float(sorted_lat[0])
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(sorted_lat[int(k)])
        d0 = sorted_lat[int(f)] * (c - k)
        d1 = sorted_lat[int(c)] * (k - f)
        return float(d0 + d1)

    return {
        "p50": round(get_percentile(50.0), 2),
        "p90": round(get_percentile(90.0), 2),
        "p99": round(get_percentile(99.0), 2),
    }


def calculate_retrieval_metrics(
    ground_truth_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    top_k: int = 5,
) -> RetrievalMetricsResult:
    """Calculate Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and NDCG@K.

    @param ground_truth_chunk_ids: Expected relevant chunk IDs.
    @param retrieved_chunk_ids: Retrieved candidate chunk IDs.
    @param top_k: Evaluation cut-off threshold.
    @returns: RetrievalMetricsResult containing precision, recall, MRR, and NDCG@K.
    """
    if not ground_truth_chunk_ids or not retrieved_chunk_ids:
        return RetrievalMetricsResult(precision_at_k=0.0, recall_at_k=0.0, mrr=0.0, ndcg_at_k=0.0)

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

    ndcg: float = calculate_ndcg_at_k(
        ground_truth_chunk_ids=ground_truth_chunk_ids,
        retrieved_chunk_ids=retrieved_chunk_ids,
        top_k=top_k,
    )

    return RetrievalMetricsResult(
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        mrr=round(mrr, 4),
        ndcg_at_k=round(ndcg, 4),
    )


def evaluate_channel_mrr(ground_truth_chunk_ids: list[str], candidate_chunk_ids: list[str]) -> float:
    """Helper to evaluate MRR for a specific retrieval channel (Vector, BM25, Graph, RRF)."""
    if not ground_truth_chunk_ids or not candidate_chunk_ids:
        return 0.0
    gt_set: set[str] = set(ground_truth_chunk_ids)
    for rank, cid in enumerate(candidate_chunk_ids, start=1):
        if cid in gt_set:
            return round(1.0 / float(rank), 4)
    return 0.0


def evaluate_golden_set(
    golden_test_cases: list[dict],
    include_ragas: bool = True,
) -> EvaluationBenchmarkReport:
    """Run golden-set benchmarking suite across retrieval, generation, citations, and RAGAS.

    @param golden_test_cases: List of dict objects containing 'question', 'document_ids', 'expected_chunks', 'expected_answer'.
    @param include_ragas: Whether to calculate RAGAS metrics (Faithfulness, Answer Relevance, Context Precision, Recall).
    @returns: EvaluationBenchmarkReport summarizing quality, RAGAS, and latency metrics.
    """
    if not golden_test_cases:
        return EvaluationBenchmarkReport(
            total_test_cases=0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            mean_mrr=0.0,
            mean_ndcg_at_k=0.0,
            mean_citation_correctness=0.0,
            avg_latency_ms=0.0,
            latency_p50_ms=0.0,
            latency_p90_ms=0.0,
            latency_p99_ms=0.0,
            vector_mrr=0.0,
            bm25_mrr=0.0,
            graph_mrr=0.0,
            rrf_mrr=0.0,
            ragas_faithfulness=0.0,
            ragas_answer_relevance=0.0,
            ragas_context_precision=0.0,
            ragas_context_recall=0.0,
        )

    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    citation_accs: list[float] = []
    latencies: list[float] = []

    vector_mrrs: list[float] = []
    bm25_mrrs: list[float] = []
    graph_mrrs: list[float] = []
    rrf_mrrs: list[float] = []

    ragas_samples: list[dict] = []

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
            ndcgs.append(ret_metrics.ndcg_at_k)

            # Evaluate channel hits if provided in test_case
            if "vector_chunks" in test_case:
                vector_mrrs.append(evaluate_channel_mrr(expected_chunks, test_case["vector_chunks"]))
            if "bm25_chunks" in test_case:
                bm25_mrrs.append(evaluate_channel_mrr(expected_chunks, test_case["bm25_chunks"]))
            if "graph_chunks" in test_case:
                graph_mrrs.append(evaluate_channel_mrr(expected_chunks, test_case["graph_chunks"]))
            if "rrf_chunks" in test_case:
                rrf_mrrs.append(evaluate_channel_mrr(expected_chunks, test_case["rrf_chunks"]))

        if res.citations:
            valid_cites: int = sum(1 for c in res.citations if c.is_valid)
            citation_accs.append(valid_cites / float(len(res.citations)))

        # Prepare contexts for RAGAS evaluation
        contexts: list[str] = [c.snippet for c in res.citations if c.snippet]
        if not contexts and res.citations:
            contexts = [f"Doc: {c.document_id}, Page: {c.page}, Section: {c.section}" for c in res.citations]

        ragas_samples.append({
            "question": question,
            "answer": res.answer,
            "contexts": contexts,
            "ground_truth": test_case.get("expected_answer", ""),
        })

    perc_latencies: dict[str, float] = calculate_percentile_latencies(latencies)

    # Evaluate RAGAS metrics across collected samples
    ragas_res: Optional[RagasMetricsResult] = None
    if include_ragas and ragas_samples:
        ragas_res = evaluate_ragas_batch(ragas_samples)

    report: EvaluationBenchmarkReport = EvaluationBenchmarkReport(
        total_test_cases=len(golden_test_cases),
        mean_precision_at_k=round(sum(precisions) / float(len(precisions) or 1), 4),
        mean_recall_at_k=round(sum(recalls) / float(len(recalls) or 1), 4),
        mean_mrr=round(sum(mrrs) / float(len(mrrs) or 1), 4),
        mean_ndcg_at_k=round(sum(ndcgs) / float(len(ndcgs) or 1), 4),
        mean_citation_correctness=round(sum(citation_accs) / float(len(citation_accs) or 1), 4),
        avg_latency_ms=round(sum(latencies) / float(len(latencies) or 1), 2),
        latency_p50_ms=perc_latencies["p50"],
        latency_p90_ms=perc_latencies["p90"],
        latency_p99_ms=perc_latencies["p99"],
        vector_mrr=round(sum(vector_mrrs) / float(len(vector_mrrs) or 1), 4),
        bm25_mrr=round(sum(bm25_mrrs) / float(len(bm25_mrrs) or 1), 4),
        graph_mrr=round(sum(graph_mrrs) / float(len(graph_mrrs) or 1), 4),
        rrf_mrr=round(sum(rrf_mrrs) / float(len(rrf_mrrs) or 1), 4),
        ragas_faithfulness=ragas_res.faithfulness if ragas_res else 0.0,
        ragas_answer_relevance=ragas_res.answer_relevance if ragas_res else 0.0,
        ragas_context_precision=ragas_res.context_precision if ragas_res else 0.0,
        ragas_context_recall=ragas_res.context_recall if ragas_res else 0.0,
    )

    logger.info(
        FUNC_EVAL_GOLDEN,
        f"Evaluation completed for {report.total_test_cases} test cases: MRR={report.mean_mrr}, NDCG@K={report.mean_ndcg_at_k}, Citation Correctness={report.mean_citation_correctness}, RAGAS Faithfulness={report.ragas_faithfulness}, Relevancy={report.ragas_answer_relevance}, Latency p50={report.latency_p50_ms}ms"
    )
    return report


def evaluate_golden_dataset_file(
    dataset_path: str = "eval/golden_dataset.json",
    max_cases: Optional[int] = None,
    include_ragas: bool = True,
) -> EvaluationBenchmarkReport:
    """Load persistent golden_dataset.json file and benchmark across all retrieval and RAGAS metrics.

    @param dataset_path: Path to golden_dataset.json file.
    @param max_cases: Optional limit on number of test cases to evaluate.
    @param include_ragas: Whether to evaluate RAGAS metrics.
    @returns: EvaluationBenchmarkReport summarizing quality and latency results.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Golden dataset not found at '{dataset_path}'")

    with open(dataset_path, "r", encoding="utf-8") as f:
        cases: list[dict] = json.load(f)

    if max_cases is not None and max_cases > 0:
        cases = cases[:max_cases]

    return evaluate_golden_set(golden_test_cases=cases, include_ragas=include_ragas)

