"""PostgreSQL Data Access Layer.

Handles authoritative relational storage, schema initialization, document/chunk persistence,
status tracking, transactional operations, and startup hydration queries.
"""

import json
import psycopg2
from psycopg2.extras import DictCursor
from typing import Optional
from interfaces.document_interface import (
    DocumentMetadataInterface,
    DocumentStatusInterface,
    DocumentStatusEnum,
    IngestionStageEnum,
    IngestionJobInterface,
    ChunkInterface,
    ChunkMetadataInterface,
)
from utils.config import get_config
from utils.logger import logger

FUNC_GET_CONN: str = "get_db_connection"
FUNC_INIT_SCHEMA: str = "ensure_db_schema"
FUNC_SAVE_DOC: str = "save_document_to_db"
FUNC_UPDATE_STATUS: str = "update_document_status_in_db"
FUNC_SAVE_CHUNKS: str = "save_chunks_to_db"
FUNC_GET_META: str = "get_document_metadata_from_db"
FUNC_GET_STATUS: str = "get_document_status_from_db"
FUNC_LOAD_CHUNKS: str = "load_all_chunks_from_db"
FUNC_DELETE_DOC: str = "delete_document_from_db"


def get_db_connection():
    """Establish connection to PostgreSQL database using application configuration.

    @returns: Connected psycopg2 Connection object or None if unconfigured/failed.
    """
    config = get_config()
    db_url = config.database_url

    if not db_url:
        host = config.postgres_host
        user = config.postgres_user
        password = config.postgres_password
        dbname = config.postgres_db
        if not host or not user:
            logger.warning(FUNC_GET_CONN, "PostgreSQL connection details missing.")
            return None
        try:
            conn = psycopg2.connect(
                host=host,
                database=dbname,
                user=user,
                password=password,
                sslmode="require",
            )
            return conn
        except Exception as exc:
            logger.error(FUNC_GET_CONN, "Failed connecting to PostgreSQL database", exc=exc)
            return None

    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as exc:
        logger.error(FUNC_GET_CONN, "Failed connecting to PostgreSQL via DATABASE_URL", exc=exc)
        return None


def ensure_db_schema() -> bool:
    """Safely initialize PostgreSQL documents and chunks tables and indexes.

    @returns: True if schema verified/initialized successfully, False otherwise.
    """
    conn = get_db_connection()
    if conn is None:
        logger.warning(FUNC_INIT_SCHEMA, "Database connection unavailable for schema initialization.")
        return False

    ddl_statements = [
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id VARCHAR(64) PRIMARY KEY,
            tenant_id VARCHAR(64) NOT NULL DEFAULT 'default_tenant',
            title TEXT NOT NULL,
            source_filename VARCHAR(255) NOT NULL,
            total_pages INT NOT NULL DEFAULT 1,
            total_chunks INT NOT NULL DEFAULT 0,
            status VARCHAR(32) NOT NULL DEFAULT 'PROCESSING',
            error_message TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id VARCHAR(16) PRIMARY KEY,
            document_id VARCHAR(64) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            tenant_id VARCHAR(64) NOT NULL DEFAULT 'default_tenant',
            page INT NOT NULL DEFAULT 1,
            chunk_index INT NOT NULL DEFAULT 0,
            section VARCHAR(255) NOT NULL DEFAULT 'General',
            source_filename VARCHAR(255) NOT NULL,
            text TEXT NOT NULL,
            entities JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS ingestion_jobs (
            job_id VARCHAR(64) PRIMARY KEY,
            document_id VARCHAR(64) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            tenant_id VARCHAR(64) NOT NULL DEFAULT 'default_tenant',
            status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
            stage VARCHAR(32) NOT NULL DEFAULT 'QUEUED',
            progress_percent NUMERIC(5,2) NOT NULL DEFAULT 0.00,
            retry_count INT NOT NULL DEFAULT 0,
            max_retries INT NOT NULL DEFAULT 3,
            error_message TEXT,
            correlation_id VARCHAR(64),
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);",
        "CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON chunks(tenant_id);",
        "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_doc ON ingestion_jobs(document_id);",
        "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status);",
    ]

    try:
        with conn:
            with conn.cursor() as cur:
                for stmt in ddl_statements:
                    cur.execute(stmt)
        logger.info(FUNC_INIT_SCHEMA, "PostgreSQL schema verified and ready.")
        return True
    except Exception as exc:
        logger.error(FUNC_INIT_SCHEMA, "Error initializing PostgreSQL database schema", exc=exc)
        return False
    finally:
        conn.close()


def save_document_to_db(metadata: DocumentMetadataInterface) -> bool:
    """Save or update document master record in PostgreSQL documents table.

    @param metadata: DocumentMetadataInterface payload.
    @returns: True if successfully saved, False otherwise.
    """
    conn = get_db_connection()
    if conn is None:
        return False

    query = """
    INSERT INTO documents (
        document_id, tenant_id, title, source_filename, total_pages, total_chunks, status, created_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (document_id) DO UPDATE SET
        title = EXCLUDED.title,
        total_pages = EXCLUDED.total_pages,
        total_chunks = EXCLUDED.total_chunks,
        status = EXCLUDED.status;
    """

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        metadata.document_id,
                        metadata.tenant_id,
                        metadata.title,
                        metadata.source_filename,
                        metadata.total_pages,
                        metadata.total_chunks,
                        DocumentStatusEnum.PROCESSING.value,
                        metadata.created_at,
                    ),
                )
        logger.info(FUNC_SAVE_DOC, f"Saved document record to PostgreSQL: {metadata.document_id}")
        return True
    except Exception as exc:
        logger.error(FUNC_SAVE_DOC, f"Error saving document {metadata.document_id} to PostgreSQL", exc=exc)
        return False
    finally:
        conn.close()


def update_document_status_in_db(
    document_id: str,
    status: DocumentStatusEnum,
    processed_chunks: int = 0,
    error_message: Optional[str] = None,
) -> bool:
    """Update ingestion status for given document_id in PostgreSQL.

    @param document_id: Target document identifier.
    @param status: DocumentStatusEnum value.
    @param processed_chunks: Count of processed chunks.
    @param error_message: Optional failure error detail string.
    @returns: True if update succeeded, False otherwise.
    """
    conn = get_db_connection()
    if conn is None:
        return False

    query = """
    UPDATE documents
    SET status = %s,
        total_chunks = %s,
        error_message = %s
    WHERE document_id = %s;
    """

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, (status.value, processed_chunks, error_message, document_id))
        logger.info(FUNC_UPDATE_STATUS, f"Updated document status in DB for {document_id}: {status.value}")
        return True
    except Exception as exc:
        logger.error(FUNC_UPDATE_STATUS, f"Error updating status for {document_id} in PostgreSQL", exc=exc)
        return False
    finally:
        conn.close()


# Fallback in-memory jobs store for tests when database is disconnected
_LOCAL_INGESTION_JOBS_STORE: dict[str, IngestionJobInterface] = {}


def clear_local_job_store() -> None:
    """Clear local fallback job store for testing."""
    _LOCAL_INGESTION_JOBS_STORE.clear()


def save_ingestion_job_to_db(job: IngestionJobInterface) -> bool:
    """Persist ingestion job tracking record into PostgreSQL or local store.

    @param job: IngestionJobInterface instance.
    @returns: True if successful, False otherwise.
    """
    _LOCAL_INGESTION_JOBS_STORE[job.job_id] = job
    conn = get_db_connection()
    if conn is None:
        return True

    query = """
    INSERT INTO ingestion_jobs (
        job_id, document_id, tenant_id, status, stage, progress_percent,
        retry_count, max_retries, error_message, correlation_id, created_at, started_at, completed_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (job_id) DO UPDATE SET
        status = EXCLUDED.status,
        stage = EXCLUDED.stage,
        progress_percent = EXCLUDED.progress_percent,
        retry_count = EXCLUDED.retry_count,
        error_message = EXCLUDED.error_message,
        started_at = COALESCE(ingestion_jobs.started_at, EXCLUDED.started_at),
        completed_at = EXCLUDED.completed_at;
    """

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        job.job_id,
                        job.document_id,
                        job.tenant_id,
                        job.status.value,
                        job.stage.value,
                        job.progress_percent,
                        job.retry_count,
                        job.max_retries,
                        job.error_message,
                        job.correlation_id,
                        job.created_at,
                        job.started_at,
                        job.completed_at,
                    ),
                )
        return True
    except Exception as exc:
        logger.error("save_ingestion_job_to_db", f"Error saving job {job.job_id} to DB: {exc}")
        return False
    finally:
        conn.close()


def update_job_progress_in_db(
    job_id: str,
    status: DocumentStatusEnum,
    stage: IngestionStageEnum,
    progress_percent: float,
    error_message: Optional[str] = None,
    retry_count: Optional[int] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> bool:
    """Update job status, stage, and progress percentage in PostgreSQL and synchronized local store.

    @returns: True if update succeeded, False otherwise.
    """
    if job_id in _LOCAL_INGESTION_JOBS_STORE:
        j = _LOCAL_INGESTION_JOBS_STORE[job_id]
        j.status = status
        j.stage = stage
        j.progress_percent = progress_percent
        if error_message is not None:
            j.error_message = error_message
        if retry_count is not None:
            j.retry_count = retry_count
        if started_at is not None:
            j.started_at = started_at
        if completed_at is not None:
            j.completed_at = completed_at

    conn = get_db_connection()
    if conn is None:
        return True

    query = """
    UPDATE ingestion_jobs
    SET status = %s,
        stage = %s,
        progress_percent = %s,
        error_message = COALESCE(%s, error_message),
        retry_count = COALESCE(%s, retry_count),
        started_at = COALESCE(%s, started_at),
        completed_at = COALESCE(%s, completed_at)
    WHERE job_id = %s;
    """

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        status.value,
                        stage.value,
                        progress_percent,
                        error_message,
                        retry_count,
                        started_at,
                        completed_at,
                        job_id,
                    ),
                )
        return True
    except Exception as exc:
        logger.error("update_job_progress_in_db", f"Error updating job {job_id}: {exc}")
        return False
    finally:
        conn.close()


def get_ingestion_job_from_db(job_id: str) -> Optional[IngestionJobInterface]:
    """Retrieve ingestion job record by job_id from PostgreSQL or local store."""
    if not job_id:
        return None

    conn = get_db_connection()
    if conn is None:
        return _LOCAL_INGESTION_JOBS_STORE.get(job_id)

    query = """
    SELECT job_id, document_id, tenant_id, status, stage, progress_percent,
           retry_count, max_retries, error_message, correlation_id, created_at, started_at, completed_at
    FROM ingestion_jobs WHERE job_id = %s;
    """

    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (job_id,))
            row = cur.fetchone()
            if not row:
                return _LOCAL_INGESTION_JOBS_STORE.get(job_id)

            return IngestionJobInterface(
                job_id=row["job_id"],
                document_id=row["document_id"],
                tenant_id=row["tenant_id"],
                status=DocumentStatusEnum(row["status"]),
                stage=IngestionStageEnum(row["stage"]),
                progress_percent=float(row["progress_percent"]),
                retry_count=row["retry_count"],
                max_retries=row["max_retries"],
                error_message=row["error_message"],
                correlation_id=row["correlation_id"],
                created_at=str(row["created_at"]),
                started_at=str(row["started_at"]) if row["started_at"] else None,
                completed_at=str(row["completed_at"]) if row["completed_at"] else None,
            )
    except Exception as exc:
        logger.error("get_ingestion_job_from_db", f"Error reading job {job_id}", exc=exc)
        return _LOCAL_INGESTION_JOBS_STORE.get(job_id)
    finally:
        conn.close()


def get_latest_job_for_document_from_db(document_id: str) -> Optional[IngestionJobInterface]:
    """Retrieve the most recent ingestion job record for a given document_id."""
    if not document_id:
        return None

    conn = get_db_connection()
    if conn is None:
        matching = [j for j in _LOCAL_INGESTION_JOBS_STORE.values() if j.document_id == document_id]
        if matching:
            return sorted(matching, key=lambda x: x.created_at, reverse=True)[0]
        return None

    query = """
    SELECT job_id, document_id, tenant_id, status, stage, progress_percent,
           retry_count, max_retries, error_message, correlation_id, created_at, started_at, completed_at
    FROM ingestion_jobs WHERE document_id = %s ORDER BY created_at DESC LIMIT 1;
    """

    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (document_id,))
            row = cur.fetchone()
            if not row:
                matching = [j for j in _LOCAL_INGESTION_JOBS_STORE.values() if j.document_id == document_id]
                return sorted(matching, key=lambda x: x.created_at, reverse=True)[0] if matching else None

            return IngestionJobInterface(
                job_id=row["job_id"],
                document_id=row["document_id"],
                tenant_id=row["tenant_id"],
                status=DocumentStatusEnum(row["status"]),
                stage=IngestionStageEnum(row["stage"]),
                progress_percent=float(row["progress_percent"]),
                retry_count=row["retry_count"],
                max_retries=row["max_retries"],
                error_message=row["error_message"],
                correlation_id=row["correlation_id"],
                created_at=str(row["created_at"]),
                started_at=str(row["started_at"]) if row["started_at"] else None,
                completed_at=str(row["completed_at"]) if row["completed_at"] else None,
            )
    except Exception as exc:
        logger.error("get_latest_job_for_document_from_db", f"Error reading latest job for doc {document_id}", exc=exc)
        matching = [j for j in _LOCAL_INGESTION_JOBS_STORE.values() if j.document_id == document_id]
        return sorted(matching, key=lambda x: x.created_at, reverse=True)[0] if matching else None
    finally:
        conn.close()


def save_chunks_to_db(chunks: list[ChunkInterface]) -> int:
    """Persist list of document chunks into PostgreSQL chunks table within a single transaction.

    @param chunks: List of ChunkInterface objects to insert.
    @returns: Total count of successfully inserted chunks.
    """
    if not chunks:
        return 0

    conn = get_db_connection()
    if conn is None:
        return 0

    query = """
    INSERT INTO chunks (
        chunk_id, document_id, tenant_id, page, chunk_index, section, source_filename, text, entities
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (chunk_id) DO UPDATE SET
        text = EXCLUDED.text,
        section = EXCLUDED.section,
        entities = EXCLUDED.entities;
    """

    rows = []
    for idx, chunk in enumerate(chunks):
        rows.append((
            chunk.chunk_id,
            chunk.document_id,
            chunk.tenant_id,
            chunk.metadata.page,
            idx,
            chunk.metadata.section,
            chunk.metadata.source,
            chunk.text,
            json.dumps(chunk.metadata.entities),
        ))

    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(query, rows)
        logger.info(FUNC_SAVE_CHUNKS, f"Successfully persisted {len(chunks)} chunks to PostgreSQL.")
        return len(chunks)
    except Exception as exc:
        logger.error(FUNC_SAVE_CHUNKS, f"Error saving {len(chunks)} chunks to PostgreSQL", exc=exc)
        return 0
    finally:
        conn.close()


def get_document_metadata_from_db(document_id: str) -> Optional[DocumentMetadataInterface]:
    """Retrieve document metadata by document ID from PostgreSQL.

    @param document_id: Unique document identifier string.
    @returns: DocumentMetadataInterface object or None if not found.
    """
    if not document_id:
        return None

    conn = get_db_connection()
    if conn is None:
        return None

    query = """
    SELECT document_id, tenant_id, title, source_filename, total_pages, total_chunks, created_at
    FROM documents WHERE document_id = %s;
    """

    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (document_id,))
            row = cur.fetchone()
            if not row:
                return None

            return DocumentMetadataInterface(
                document_id=row["document_id"],
                tenant_id=row["tenant_id"],
                title=row["title"],
                source_filename=row["source_filename"],
                total_pages=row["total_pages"],
                total_chunks=row["total_chunks"],
                created_at=str(row["created_at"]),
            )
    except Exception as exc:
        logger.error(FUNC_GET_META, f"Error reading document metadata for {document_id}", exc=exc)
        return None
    finally:
        conn.close()


def get_document_status_from_db(document_id: str) -> Optional[DocumentStatusInterface]:
    """Retrieve document job processing status from PostgreSQL, enriched with latest job stage."""
    if not document_id:
        return None

    conn = get_db_connection()
    if conn is None:
        latest_job = get_latest_job_for_document_from_db(document_id)
        if latest_job:
            return DocumentStatusInterface(
                job_id=latest_job.job_id,
                document_id=document_id,
                tenant_id=latest_job.tenant_id,
                status=latest_job.status,
                stage=latest_job.stage,
                progress_percent=latest_job.progress_percent,
                error_message=latest_job.error_message,
                correlation_id=latest_job.correlation_id,
            )
        return None

    query = """
    SELECT d.document_id, d.tenant_id, d.status, d.error_message, d.total_chunks,
           j.job_id, j.stage, j.progress_percent, j.correlation_id
    FROM documents d
    LEFT JOIN LATERAL (
        SELECT job_id, stage, progress_percent, correlation_id
        FROM ingestion_jobs
        WHERE document_id = d.document_id
        ORDER BY created_at DESC LIMIT 1
    ) j ON true
    WHERE d.document_id = %s;
    """

    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, (document_id,))
            row = cur.fetchone()
            if not row:
                return None

            stage_val = IngestionStageEnum(row["stage"]) if row.get("stage") else None
            return DocumentStatusInterface(
                job_id=row.get("job_id"),
                document_id=row["document_id"],
                tenant_id=row["tenant_id"],
                status=DocumentStatusEnum(row["status"]),
                stage=stage_val,
                progress_percent=float(row["progress_percent"]) if row.get("progress_percent") is not None else 0.0,
                error_message=row["error_message"],
                processed_chunks=row["total_chunks"],
                correlation_id=row.get("correlation_id"),
            )
    except Exception as exc:
        logger.error(FUNC_GET_STATUS, f"Error reading document status for {document_id}", exc=exc)
        return None
    finally:
        conn.close()


def load_all_chunks_from_db(tenant_id: Optional[str] = None) -> list[ChunkInterface]:
    """Read authoritative chunk records from PostgreSQL to hydrate BM25 inverted index at startup.

    @param tenant_id: Optional tenant ID filter.
    @returns: List of reconstructed ChunkInterface objects.
    """
    conn = get_db_connection()
    if conn is None:
        logger.warning(FUNC_LOAD_CHUNKS, "Database connection unavailable for chunk hydration.")
        return []

    if tenant_id:
        query = """
        SELECT chunk_id, document_id, tenant_id, page, section, source_filename, text, entities
        FROM chunks WHERE tenant_id = %s ORDER BY created_at ASC;
        """
        params = (tenant_id,)
    else:
        query = """
        SELECT chunk_id, document_id, tenant_id, page, section, source_filename, text, entities
        FROM chunks ORDER BY created_at ASC;
        """
        params = ()

    chunks: list[ChunkInterface] = []
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            for row in rows:
                entities_raw = row["entities"]
                entities_list = json.loads(entities_raw) if isinstance(entities_raw, str) else list(entities_raw or [])

                chunk_obj = ChunkInterface(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    tenant_id=row["tenant_id"],
                    text=row["text"],
                    metadata=ChunkMetadataInterface(
                        document_id=row["document_id"],
                        chunk_id=row["chunk_id"],
                        tenant_id=row["tenant_id"],
                        page=row["page"],
                        section=row["section"],
                        source=row["source_filename"],
                        entities=entities_list,
                    ),
                )
                chunks.append(chunk_obj)

        logger.info(FUNC_LOAD_CHUNKS, f"Loaded {len(chunks)} authoritative chunks from PostgreSQL for BM25 hydration.")
        return chunks
    except Exception as exc:
        logger.error(FUNC_LOAD_CHUNKS, "Error loading chunks from PostgreSQL database", exc=exc)
        return []
    finally:
        conn.close()


def delete_document_from_db(document_id: str) -> bool:
    """Delete document master record from PostgreSQL, triggering cascading deletion of chunks.

    @param document_id: Target document identifier string.
    @returns: True if deletion succeeded, False otherwise.
    """
    if not document_id:
        return False

    conn = get_db_connection()
    if conn is None:
        return False

    query = "DELETE FROM documents WHERE document_id = %s;"

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(query, (document_id,))
        logger.info(FUNC_DELETE_DOC, f"Deleted document {document_id} from PostgreSQL database.")
        return True
    except Exception as exc:
        logger.error(FUNC_DELETE_DOC, f"Error deleting document {document_id} from PostgreSQL", exc=exc)
        return False
    finally:
        conn.close()
