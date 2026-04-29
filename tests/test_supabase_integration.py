"""
Unit tests for Supabase database integration — AIService (FastAPI/SQLAlchemy).

Tasks covered:
  7.2 — SQLAlchemy engine is created with pool_pre_ping=True
  7.3 — settings.DATABASE_URL contains sslmode=require
  7.4 — AIService .env.example contains a Supabase-format DATABASE_URL placeholder
  7.6 — AIService docker-compose.yml has no hardcoded DATABASE_URL
  7.8 — ensure_pgvector_schema() raises when the vector extension is absent
"""
import sys
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

AISERVICE_DIR = Path(__file__).resolve().parents[1]
COMPOSE_FILE = AISERVICE_DIR / "docker-compose.yml"
ENV_EXAMPLE_FILE = AISERVICE_DIR / ".env.example"

"""
Unit tests for Supabase database integration — AIService (FastAPI/SQLAlchemy).

Tasks covered:
  7.2 — SQLAlchemy engine is created with pool_pre_ping=True
  7.3 — settings.DATABASE_URL contains sslmode=require
  7.4 — AIService .env.example contains a Supabase-format DATABASE_URL placeholder
  7.6 — AIService docker-compose.yml has no hardcoded DATABASE_URL
  7.8 — ensure_pgvector_schema() raises when the vector extension is absent
"""
import sys
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

AISERVICE_DIR = Path(__file__).resolve().parents[1]
COMPOSE_FILE = AISERVICE_DIR / "docker-compose.yml"
ENV_EXAMPLE_FILE = AISERVICE_DIR / ".env.example"
