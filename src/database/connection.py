import os
import logging
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


def get_db_session() -> Session:
    """Dependency helper to get database session."""
    session = SessionLocal()
    try:
        return session
    except Exception as e:
        session.close()
        raise e
