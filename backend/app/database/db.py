from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def _engine_url(database_url: str) -> URL:
    url = make_url(database_url)
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg")
    return url


_settings = get_settings()
_database_url = make_url(_settings.database_url)
if _database_url.get_backend_name() == "sqlite" and _database_url.database not in {None, ":memory:"}:
    Path(_database_url.database).parent.mkdir(parents=True, exist_ok=True)
_connect_args = {"check_same_thread": False} if _database_url.get_backend_name() == "sqlite" else {}
engine = create_engine(
    _engine_url(_settings.database_url),
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args=_connect_args,
)
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
