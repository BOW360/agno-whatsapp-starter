"""
Database Session
----------------

PostgreSQL database connection for AgentOS.
"""

from agno.db.postgres import PostgresDb
from agno.knowledge import Knowledge
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.pgvector import PgVector, SearchType
from os import getenv

from db.url import db_url

DB_ID = "agentos-db"


def get_postgres_db(contents_table: str | None = None) -> PostgresDb:
    """Create a PostgresDb instance.

    Args:
        contents_table: Optional table name for storing knowledge contents.

    Returns:
        Configured PostgresDb instance.
    """
    if contents_table is not None:
        return PostgresDb(id=DB_ID, db_url=db_url, knowledge_table=contents_table)
    return PostgresDb(id=DB_ID, db_url=db_url)


def create_knowledge(name: str, table_name: str) -> Knowledge:
    """Create a Knowledge instance with PgVector hybrid search.

    Args:
        name: Display name for the knowledge base.
        table_name: PostgreSQL table name for vector storage.

    Returns:
        Configured Knowledge instance.
    """
    return Knowledge(
        name=name,
        vector_db=PgVector(
            db_url=db_url,
            table_name=table_name,
            search_type=SearchType.hybrid,
            embedder=OpenAIEmbedder(
                id="text-embedding-3-small",
                api_key=getenv("OPENAI_EMBEDDING_API_KEY") or getenv("OPENAI_API_KEY"),
                base_url=getenv("OPENAI_EMBEDDING_BASE_URL") or getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            ),
        ),
        contents_db=get_postgres_db(contents_table=f"{table_name}_contents"),
    )
