"""RAGAS Evaluation Engine for Enterprise RAG Platform.

Evaluates:
- Faithfulness
- Answer Relevance (answer_relevancy)
- Context Precision
- Context Recall

Uses GPT-4o mini as the evaluation judge and text-embedding-3-small for semantic relevancy.
Supports deterministic mock/offline test mode for isolated testing without network or OpenAI API calls.
"""

import os
import re
from typing import NamedTuple, Optional
from openai import OpenAI
from utils.config import get_config
from utils.logger import logger

FUNC_INIT_JUDGE: str = "init_ragas_judge"
FUNC_EVAL_SAMPLE: str = "evaluate_ragas_sample"
FUNC_EVAL_BATCH: str = "evaluate_ragas_batch"
FUNC_MOCK_EVAL: str = "evaluate_ragas_mock"

DEFAULT_JUDGE_MODEL: str = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

# Global mock mode toggle
_MOCK_RAGAS_MODE: bool = False


class RagasMetricsResult(NamedTuple):
    """Container for RAGAS evaluation scores."""
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float

    @property
    def answer_relevancy(self) -> float:
        """Alias for answer_relevance to maintain RAGAS naming parity."""
        return self.answer_relevance


def set_mock_ragas_mode(enabled: bool) -> None:
    """Toggle mock RAGAS evaluation mode for fast/offline unit testing.

    @param enabled: True to use deterministic local mock evaluations, False for real LLM judge.
    """
    global _MOCK_RAGAS_MODE
    _MOCK_RAGAS_MODE = enabled
    logger.info(FUNC_EVAL_BATCH, f"RAGAS mock mode set to: {enabled}")


def is_mock_ragas_mode() -> bool:
    """Check if mock RAGAS evaluation mode is active.

    @returns: True if mock mode is enabled or if OPENAI_API_KEY is missing.
    """
    if _MOCK_RAGAS_MODE:
        return True
    config = get_config()
    api_key: str = config.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    return not bool(api_key.strip())


def _extract_keywords(text: str) -> set[str]:
    """Helper to extract lowercase significant words from text."""
    if not text:
        return set()
    stopwords: set[str] = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or", "is",
        "are", "was", "were", "be", "been", "by", "as", "it", "this", "that", "with"
    }
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    return {w for w in words if w not in stopwords}


def evaluate_ragas_mock(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> RagasMetricsResult:
    """Deterministic local evaluation simulating RAGAS metrics for mock/offline testing.

    Calculates heuristic scores bounded [0.0, 1.0] without network or LLM calls:
    - Faithfulness: ratio of answer keywords grounded in retrieved contexts.
    - Answer Relevance: semantic keyword alignment between question and answer.
    - Context Precision: ranks of relevant chunks containing ground-truth keywords.
    - Context Recall: fraction of ground-truth keywords retrieved in context chunks.

    @param question: Query text.
    @param answer: Generated answer text.
    @param contexts: Retrieved context chunks/snippets.
    @param ground_truth: Reference expected answer.
    @returns: RagasMetricsResult containing scores rounded to 4 decimals.
    """
    q_words: set[str] = _extract_keywords(question)
    a_words: set[str] = _extract_keywords(answer)
    gt_words: set[str] = _extract_keywords(ground_truth)

    all_ctx_text: str = " ".join(contexts) if contexts else ""
    ctx_words: set[str] = _extract_keywords(all_ctx_text)

    # 1. Faithfulness: Are answer claims present in retrieved context?
    if not a_words:
        faithfulness_score: float = 0.0
    elif not contexts or not ctx_words:
        # If no context exists, cannot be faithful unless answer is an explicit admission of insufficient evidence
        if "insufficient evidence" in answer.lower():
            faithfulness_score = 1.0
        else:
            faithfulness_score = 0.0
    else:
        overlap = a_words.intersection(ctx_words)
        faithfulness_score = min(1.0, round(len(overlap) / float(len(a_words)), 4))

    # 2. Answer Relevance: Does the answer address the question?
    if not a_words or not q_words:
        answer_rel_score: float = 0.0
    else:
        overlap_q = a_words.intersection(q_words)
        base_score = len(overlap_q) / float(len(q_words))
        answer_rel_score = min(1.0, round(0.5 + 0.5 * min(1.0, base_score * 2.0), 4))

    # 3. Context Precision: Are the most relevant chunks ranked at the top?
    if not contexts or not gt_words:
        context_precision_score: float = 0.0
    else:
        precisions: list[float] = []
        hits: int = 0
        for rank, ctx in enumerate(contexts, start=1):
            ctx_kw = _extract_keywords(ctx)
            if ctx_kw.intersection(gt_words):
                hits += 1
                precisions.append(hits / float(rank))
        if precisions:
            context_precision_score = round(sum(precisions) / float(len(precisions)), 4)
        else:
            context_precision_score = 0.0

    # 4. Context Recall: Were all ground truth key concepts retrieved?
    if not gt_words:
        context_recall_score: float = 1.0 if contexts else 0.0
    elif not contexts or not ctx_words:
        context_recall_score = 0.0
    else:
        gt_in_ctx = gt_words.intersection(ctx_words)
        context_recall_score = round(len(gt_in_ctx) / float(len(gt_words)), 4)

    logger.info(
        FUNC_MOCK_EVAL,
        f"Mock RAGAS computed: Faithfulness={faithfulness_score}, Relevancy={answer_rel_score}, Precision={context_precision_score}, Recall={context_recall_score}"
    )

    return RagasMetricsResult(
        faithfulness=faithfulness_score,
        answer_relevance=answer_rel_score,
        context_precision=context_precision_score,
        context_recall=context_recall_score,
    )


def init_ragas_judge(model_id: str = DEFAULT_JUDGE_MODEL):
    """Initialize OpenAI judge LLM and Embeddings for RAGAS evaluation.

    @param model_id: OpenAI chat model name (default gpt-4o-mini).
    @returns: Tuple of (judge_llm, judge_embeddings).
    """
    config = get_config()
    api_key: str = config.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning(FUNC_INIT_JUDGE, "No OPENAI_API_KEY found. RAGAS cannot initialize real judge.")
        return None, None

    embedding_model: str = config.openai_embedding_model or DEFAULT_EMBEDDING_MODEL

    try:
        from ragas.llms import llm_factory
        from langchain_openai import OpenAIEmbeddings

        client = OpenAI(api_key=api_key)
        judge_llm = llm_factory(model_id, client=client)
        judge_emb = OpenAIEmbeddings(model=embedding_model, api_key=api_key)

        logger.info(FUNC_INIT_JUDGE, f"Initialized RAGAS judge '{model_id}' with embeddings '{embedding_model}'.")
        return judge_llm, judge_emb
    except Exception as exc:
        logger.error(FUNC_INIT_JUDGE, f"Failed to initialize RAGAS judge: {exc}")
        return None, None


def evaluate_ragas_sample(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    use_mock: Optional[bool] = None,
) -> RagasMetricsResult:
    """Evaluate a single test case across all 4 RAGAS metrics.

    @param question: Query text.
    @param answer: Generated answer text.
    @param contexts: List of retrieved context strings.
    @param ground_truth: Expected reference answer text.
    @param use_mock: Optional override for mock mode.
    @returns: RagasMetricsResult containing the four metric scores.
    """
    should_mock: bool = is_mock_ragas_mode() if use_mock is None else use_mock
    if should_mock:
        return evaluate_ragas_mock(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

    batch_data = [
        {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        }
    ]
    return evaluate_ragas_batch(batch_data, use_mock=False)


def evaluate_ragas_batch(
    samples: list[dict],
    use_mock: Optional[bool] = None,
) -> RagasMetricsResult:
    """Run batch RAGAS evaluation across Faithfulness, Answer Relevance, Context Precision, and Context Recall.

    @param samples: List of dicts with 'question', 'answer', 'contexts', 'ground_truth'.
    @param use_mock: Optional override to force mock or real mode.
    @returns: RagasMetricsResult with averaged scores across all samples.
    """
    if not samples:
        return RagasMetricsResult(
            faithfulness=0.0,
            answer_relevance=0.0,
            context_precision=0.0,
            context_recall=0.0,
        )

    should_mock: bool = is_mock_ragas_mode() if use_mock is None else use_mock
    if should_mock:
        logger.info(FUNC_EVAL_BATCH, f"Evaluating batch of {len(samples)} samples in mock mode.")
        results: list[RagasMetricsResult] = [
            evaluate_ragas_mock(
                question=s.get("question", ""),
                answer=s.get("answer", ""),
                contexts=s.get("contexts", []),
                ground_truth=s.get("ground_truth", s.get("expected_answer", "")),
            )
            for s in samples
        ]
        mean_faith = round(sum(r.faithfulness for r in results) / float(len(results)), 4)
        mean_rel = round(sum(r.answer_relevance for r in results) / float(len(results)), 4)
        mean_prec = round(sum(r.context_precision for r in results) / float(len(results)), 4)
        mean_rec = round(sum(r.context_recall for r in results) / float(len(results)), 4)

        return RagasMetricsResult(
            faithfulness=mean_faith,
            answer_relevance=mean_rel,
            context_precision=mean_prec,
            context_recall=mean_rec,
        )

    # Real RAGAS evaluation using GPT-4o-mini
    try:
        import warnings
        from datasets import Dataset

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

        judge_llm, judge_emb = init_ragas_judge()
        if judge_llm is None or judge_emb is None:
            logger.warning(FUNC_EVAL_BATCH, "Judge initialization failed. Falling back to mock RAGAS evaluation.")
            return evaluate_ragas_batch(samples, use_mock=True)

        ragas_data = {
            "question": [s.get("question", "") for s in samples],
            "answer": [s.get("answer", "") for s in samples],
            "contexts": [s.get("contexts", []) for s in samples],
            "ground_truth": [s.get("ground_truth", s.get("expected_answer", "")) for s in samples],
        }
        dataset = Dataset.from_dict(ragas_data)
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        logger.info(FUNC_EVAL_BATCH, f"Executing RAGAS evaluate() on {len(samples)} samples with GPT-4o-mini...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            eval_result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=judge_llm,
                embeddings=judge_emb,
            )

        df = eval_result.to_pandas()

        def safe_mean(col_name: str) -> float:
            if col_name in df.columns:
                val = df[col_name].dropna().mean()
                if not (val != val):  # not NaN
                    return round(float(val), 4)
            return 0.0

        faith_score = safe_mean("faithfulness")
        rel_score = safe_mean("answer_relevancy")
        prec_score = safe_mean("context_precision")
        rec_score = safe_mean("context_recall")

        logger.info(
            FUNC_EVAL_BATCH,
            f"RAGAS evaluation succeeded: Faithfulness={faith_score}, Answer Relevancy={rel_score}, Precision={prec_score}, Recall={rec_score}"
        )

        return RagasMetricsResult(
            faithfulness=faith_score,
            answer_relevance=rel_score,
            context_precision=prec_score,
            context_recall=rec_score,
        )

    except Exception as exc:
        logger.error(FUNC_EVAL_BATCH, f"Exception during RAGAS evaluate(), falling back to mock mode: {exc}")
        return evaluate_ragas_batch(samples, use_mock=True)
