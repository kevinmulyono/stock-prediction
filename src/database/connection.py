import os
import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.database.models import Base

logger = logging.getLogger(__name__)

# Default to PostgreSQL connection string from environment, fallback to SQLite for local development
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///data/stock_prediction.db"
)

# Connect arguments
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables in the database schema."""
    logger.info(f"Initializing database schema with engine: {engine.url.drivername}...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Ensures session commits on completion, rolls back on exception,
    and is guaranteed to close cleanly in finally block.
    
    Usage:
        with get_db_session() as session:
            session.query(...)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise e
    finally:
        session.close()

