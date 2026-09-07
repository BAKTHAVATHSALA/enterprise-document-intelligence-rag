"""Grounded LLM Answer Generation Helper.

Receives user query and retrieved evidence context, constructs controlled prompts,
and generates evidence-grounded answers avoiding unsupported claims.
"""

from typing import NamedTuple
from interfaces.retrieval_interface import RerankedCandidateInterface
from data_access.llm_data_access import generate_llm_completion
from utils.logger import logger
from utils.tracing import trace_step

FUNC_GENERATE_ANSWER: str = "generate_grounded_answer"

INSUFFICIENT_EVIDENCE_MSG: str = "Insufficient evidence in ingested documents to answer the question."

SYSTEM_PROMPT: str = (
    "You are an enterprise AI document assistant. "
    "Your primary duty is to answer user questions with 100% factual accuracy, "
    "grounded strictly in the retrieved evidence context provided below.\n\n"
    "STRICT GROUNDING RULES:\n"
    "1. Answer ONLY using facts directly stated in the context.\n"
    "2. Do NOT extrapolate, speculate, or draw conclusions not explicitly supported by context.\n"
    "3. Do NOT use outside pre-trained knowledge or facts from outside the context.\n"
    "4. If the retrieved evidence does not contain sufficient facts to answer the question, state: "
    f"'{INSUFFICIENT_EVIDENCE_MSG}'\n"
    "5. Include inline citations in the format [Doc: <doc_id>, Page: <page>, Section: <section>] for key facts."
)


class GenerationResult(NamedTuple):
    """Container for generated answer and confidence metric."""
    answer: str
    confidence_score: float


def _format_context(evidence_list: list[RerankedCandidateInterface]) -> str:
    """Format retrieved evidence list into prompt context block.

    @param evidence_list: Top reranked candidate chunks.
    @returns: Formatted text string.
    """
    context_blocks: list[str] = []
    for cand in evidence_list:
        meta = cand.chunk.metadata
        ref_header: str = f"[Doc: {meta.document_id}, Page: {meta.page}, Section: {meta.section}, Chunk: {meta.chunk_id}, Source: {meta.source}]"
        context_blocks.append(f"{ref_header}\n{cand.chunk.text}")
    return "\n\n".join(context_blocks)


def _extract_generation_metadata(args, kwargs, result, error):
    q = args[0] if args else kwargs.get("question", "")
    ev = args[1] if len(args) > 1 else kwargs.get("evidence_list", [])
    meta = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "evidence_count": len(ev) if isinstance(ev, list) else 0,
        "question_length": len(q) if isinstance(q, str) else 0,
    }
    if result is not None:
        meta["confidence_score"] = result.confidence_score
        meta["answer_length"] = len(result.answer) if hasattr(result, "answer") else 0
    return meta


@trace_step(name="llm_generation", run_type="llm", extract_metadata=_extract_generation_metadata)
def generate_grounded_answer(
    question: str,
    evidence_list: list[RerankedCandidateInterface],
) -> GenerationResult:
    """Generate answer grounded strictly in retrieved evidence context using OpenAI gpt-4o-mini.

    @param question: Natural language question string.
    @param evidence_list: Ordered list of top retrieved RerankedCandidateInterface hits.
    @returns: GenerationResult containing answer string and confidence score.
    """
    if not question or not question.strip():
        return GenerationResult(answer=INSUFFICIENT_EVIDENCE_MSG, confidence_score=0.0)

    if not evidence_list:
        logger.info(FUNC_GENERATE_ANSWER, "No evidence chunks provided for answer generation.")
        return GenerationResult(answer=INSUFFICIENT_EVIDENCE_MSG, confidence_score=0.0)

    context_str: str = _format_context(evidence_list)
    user_prompt: str = (
        f"Retrieved Evidence Context:\n{context_str}\n\n"
        f"User Question: {question}\n\n"
        "Answer:"
    )

    answer_text: str = generate_llm_completion(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model_id="gpt-4o-mini",
        temperature=0.0,
    )

    # Basic protection against unsupported claims / insufficient evidence
    if INSUFFICIENT_EVIDENCE_MSG.lower() in answer_text.lower() or "insufficient evidence" in answer_text.lower():
        logger.info(FUNC_GENERATE_ANSWER, "Model indicated insufficient evidence in context.")
        return GenerationResult(answer=INSUFFICIENT_EVIDENCE_MSG, confidence_score=0.0)

    confidence: float = 0.95 if evidence_list and evidence_list[0].rerank_score > 0.0 else 0.85
    logger.info(FUNC_GENERATE_ANSWER, f"Generated grounded LLM answer for query: '{question}'")
    return GenerationResult(answer=answer_text, confidence_score=confidence)
