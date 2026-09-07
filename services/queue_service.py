"""Job Queue Abstraction Layer.

Defines the pluggable JobQueueInterface and an AsyncInMemoryJobQueue implementation
for asynchronous document ingestion workflows. Architected so Redis or another
message broker can replace the in-memory queue without modifying application logic.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field

from utils.logger import logger

FUNC_ENQUEUE: str = "queue_enqueue"
FUNC_DEQUEUE: str = "queue_dequeue"
FUNC_CANCEL: str = "queue_cancel"


class IngestionJobPayload(BaseModel):
    """Payload representing a single enqueued document ingestion job."""
    job_id: str = Field(..., description="Unique job execution identifier")
    document_id: str = Field(..., description="Target document identifier")
    tenant_id: str = Field(default="default_tenant", description="Tenant identifier for isolation")
    filename: str = Field(..., description="Uploaded document filename")
    title: Optional[str] = Field(default=None, description="Document title override")
    pdf_bytes: bytes = Field(..., description="Raw PDF document byte payload")
    correlation_id: Optional[str] = Field(default=None, description="Request correlation identifier")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation ISO timestamp",
    )
    retry_count: int = Field(default=0, ge=0, description="Current retry attempt")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry limit")


class JobQueueInterface(ABC):
    """Abstract interface for ingestion job queues."""

    @abstractmethod
    async def enqueue(self, payload: IngestionJobPayload) -> str:
        """Enqueue an ingestion job payload for asynchronous processing.

        @param payload: IngestionJobPayload instance.
        @returns: job_id string.
        """
        pass

    @abstractmethod
    async def dequeue(self, timeout: float = 1.0) -> Optional[IngestionJobPayload]:
        """Dequeue the next available ingestion job payload.

        @param timeout: Maximum seconds to wait for an available job.
        @returns: IngestionJobPayload instance or None if queue is empty.
        """
        pass

    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Mark a job as cancelled.

        @param job_id: Target job identifier.
        @returns: True if found and marked for cancellation, False otherwise.
        """
        pass

    @abstractmethod
    def is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been requested for cancellation.

        @param job_id: Target job identifier.
        @returns: True if cancelled, False otherwise.
        """
        pass

    @abstractmethod
    def get_size(self) -> int:
        """Return the count of currently pending jobs in the queue."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Flush all jobs from the queue."""
        pass


from collections import deque
import time


class AsyncInMemoryJobQueue(JobQueueInterface):
    """Asynchronous in-memory job queue backed by deque and cancellation registry."""

    def __init__(self, maxsize: int = 1000) -> None:
        """Initialize in-memory queue."""
        self._maxsize: int = maxsize
        self._items: deque[IngestionJobPayload] = deque()
        self._cancelled_jobs: set[str] = set()
        self._enqueued_jobs: dict[str, IngestionJobPayload] = {}

    async def enqueue(self, payload: IngestionJobPayload) -> str:
        """Add job payload to queue."""
        self._enqueued_jobs[payload.job_id] = payload
        self._items.append(payload)
        logger.info(FUNC_ENQUEUE, f"Enqueued job {payload.job_id} for doc {payload.document_id} (queue size: {len(self._items)})")
        return payload.job_id

    async def dequeue(self, timeout: float = 1.0) -> Optional[IngestionJobPayload]:
        """Wait up to timeout seconds to dequeue next job."""
        start_t = time.time()
        while (time.time() - start_t) < timeout:
            if self._items:
                return self._items.popleft()
            await asyncio.sleep(0.05)
        if self._items:
            return self._items.popleft()
        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Register job_id in cancellation set."""
        self._cancelled_jobs.add(job_id)
        logger.info(FUNC_CANCEL, f"Registered cancellation request for job {job_id}")
        return True

    def is_cancelled(self, job_id: str) -> bool:
        """Check if job_id was registered in cancellation set."""
        return job_id in self._cancelled_jobs

    def get_size(self) -> int:
        """Return approximate count of items currently in queue."""
        return len(self._items)

    def clear(self) -> None:
        """Empty all pending items in queue."""
        self._items.clear()
        self._cancelled_jobs.clear()
        self._enqueued_jobs.clear()


# Global Singleton Queue Instance
_GLOBAL_JOB_QUEUE: Optional[JobQueueInterface] = None


def get_job_queue() -> JobQueueInterface:
    """Return the global JobQueueInterface instance (defaults to AsyncInMemoryJobQueue)."""
    global _GLOBAL_JOB_QUEUE
    if _GLOBAL_JOB_QUEUE is None:
        _GLOBAL_JOB_QUEUE = AsyncInMemoryJobQueue()
    return _GLOBAL_JOB_QUEUE


def set_job_queue(queue: Optional[JobQueueInterface]) -> None:
    """Set or reset global JobQueueInterface instance (useful for test isolation)."""
    global _GLOBAL_JOB_QUEUE
    _GLOBAL_JOB_QUEUE = queue
