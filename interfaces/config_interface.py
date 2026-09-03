"""Configuration interfaces.

Defines schemas for system environment configuration parameters.
"""

from pydantic import BaseModel, Field


class AppConfigInterface(BaseModel):
    """System application configuration parameters."""
    env: str = Field(default="development", description="Execution environment")
    port: int = Field(default=8000, description="FastAPI server port")
    log_level: str = Field(default="INFO", description="Logging verbosity level")
    aws_region: str = Field(default="us-east-1", description="AWS Bedrock region")
    pinecone_api_key: str = Field(default="", description="Pinecone vector store API key")
    pinecone_environment: str = Field(default="gcp-starter", description="Pinecone environment")
    pinecone_index_name: str = Field(default="enterprise-rag", description="Pinecone index name")
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith tracing flag")
    langsmith_api_key: str = Field(default="", description="LangSmith API Key")
