"""Application environment configuration loader utility."""

import os
from dotenv import load_dotenv
from interfaces.config_interface import AppConfigInterface
from utils.logger import logger

FUNC_GET_CONFIG: str = "get_config"


def get_config() -> AppConfigInterface:
    """Load application configuration from environment or default values.

    @returns: Instantiated AppConfigInterface configuration object.
    """
    load_dotenv()
    
    config: AppConfigInterface = AppConfigInterface(
        env=os.getenv("APP_ENV", "development"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "enterprise-rag"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
    )
    
    logger.info(FUNC_GET_CONFIG, f"Configuration loaded successfully for environment: {config.env}")
    return config
