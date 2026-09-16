import json
import time
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from config import Config

engine = create_engine(Config.DATABASE_URL, connect_args=(
    {"check_same_thread": False} if Config.DATABASE_URL.startswith("sqlite") else {}
))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class CacheEntry(Base):
    __tablename__ = "cache_entries"
    id = Column(Integer, primary_key=True)
    key = Column(String(255), unique=True, index=True, nullable=False)
    value = Column(Text, nullable=False)
    created_at = Column(Float, nullable=False)


class AnalysisRun(Base):
    """Stores one full pipeline run so results can be revisited/audited."""
    __tablename__ = "analysis_runs"
    id = Column(Integer, primary_key=True)
    target = Column(String(64), nullable=False)      # e.g. "NEPSE" or "NABIL" or "AAPL"
    market = Column(String(32), nullable=False)       # "NEPSE" | "GLOBAL"
    result_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


def cache_get(key: str, ttl_seconds: int = None):
    ttl_seconds = ttl_seconds or Config.CACHE_TTL_SECONDS
    session = SessionLocal()
    try:
        entry = session.query(CacheEntry).filter_by(key=key).first()
        if entry and (time.time() - entry.created_at) < ttl_seconds:
            return json.loads(entry.value)
        return None
    finally:
        session.close()


def cache_set(key: str, value):
    session = SessionLocal()
    try:
        entry = session.query(CacheEntry).filter_by(key=key).first()
        payload = json.dumps(value, default=str)
        if entry:
            entry.value = payload
            entry.created_at = time.time()
        else:
            entry = CacheEntry(key=key, value=payload, created_at=time.time())
            session.add(entry)
        session.commit()
    finally:
        session.close()


def save_run(target: str, market: str, result: dict) -> int:
    session = SessionLocal()
    try:
        run = AnalysisRun(target=target, market=market, result_json=json.dumps(result, default=str))
        session.add(run)
        session.commit()
        return run.id
    finally:
        session.close()


def get_recent_runs(limit: int = 20):
    session = SessionLocal()
    try:
        rows = session.query(AnalysisRun).order_by(AnalysisRun.id.desc()).limit(limit).all()
        return [
            {"id": r.id, "target": r.target, "market": r.market,
             "created_at": r.created_at.isoformat(), "result": json.loads(r.result_json)}
            for r in rows
        ]
    finally:
        session.close()
