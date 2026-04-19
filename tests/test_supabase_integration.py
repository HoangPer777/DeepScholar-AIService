"""
Unit tests for Supabase database integration - AIService (FastAPI/SQLAlchemy).
Tasks covered:
  7.2 - SQLAlchemy engine is created with pool_pre_ping=True
  7.3 - settings.DATABASE_URL contains sslmode=require
  7.4 - AIService .env.example contains a Supabase-format DATABASE_URL placeholder
  7.6 - AIService docker-compose.yml has no hardcoded DATABASE_URL
  7.8 - ensure_pgvector_schema() raises when the vector extension is absent
"""
import sys
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

AISERVICE_DIR = Path(__file__).resolve().parents[1]
COMPOSE_FILE = AISERVICE_DIR / "docker-compose.yml"
ENV_EXAMPLE_FILE = AISERVICE_DIR / ".env.example"


class TestAIServicePoolPrePing(unittest.TestCase):
    """7.2 - Verify the SQLAlchemy engine is created with pool_pre_ping=True."""

    def test_engine_has_pool_pre_ping(self):
        fake_url = (
            "postgresql://user:pass@aws-1-ap-southeast-1.pooler.supabase.com"
            ":5432/postgres?sslmode=require"
        )
        for mod in list(sys.modules.keys()):
            if mod in ("app.core.database", "app.core.config"):
                del sys.modules[mod]

        mock_engine = MagicMock()
        mock_create_engine = MagicMock(return_value=mock_engine)

        with patch.dict(
            "os.environ",
            {"DATABASE_URL": fake_url, "INTERNAL_SERVICE_KEY": "test-key"},
            clear=False,
        ), patch("sqlalchemy.create_engine", mock_create_engine):
            if "app.core.database" in sys.modules:
                del sys.modules["app.core.database"]
            import app.core.database  # noqa: F401

        self.assertTrue(mock_create_engine.called, "create_engine was never called.")
        _, kwargs = mock_create_engine.call_args
        self.assertTrue(
            kwargs.get("pool_pre_ping"),
            f"Expected pool_pre_ping=True, got kwargs={kwargs!r}",
        )


class TestAIServiceSSLInURL(unittest.TestCase):
    """7.3 - Verify that Settings.DATABASE_URL contains sslmode=require."""

    def _load_settings(self, db_url):
        for mod in list(sys.modules.keys()):
            if mod in ("app.core.config",):
                del sys.modules[mod]
        with patch.dict(
            "os.environ",
            {"DATABASE_URL": db_url, "INTERNAL_SERVICE_KEY": "test-key"},
            clear=False,
        ):
            import app.core.config as cfg_module
            settings = cfg_module.Settings()
        return settings

    def test_database_url_contains_sslmode_require(self):
        url_with_ssl = (
            "postgresql://user:pass@aws-1-ap-southeast-1.pooler.supabase.com"
            ":5432/postgres?sslmode=require"
        )
        settings = self._load_settings(url_with_ssl)
        self.assertIn(
            "sslmode=require",
            settings.DATABASE_URL,
            f"Expected sslmode=require in DATABASE_URL, got: {settings.DATABASE_URL!r}",
        )


class TestAIServiceEnvExample(unittest.TestCase):
    """7.4 - Verify AIService .env.example documents DATABASE_URL in Supabase format."""

    def test_env_example_has_supabase_format(self):
        content = ENV_EXAMPLE_FILE.read_text()
        self.assertIn("DATABASE_URL=", content, ".env.example must contain DATABASE_URL.")
        self.assertIn("supabase.com", content, "DATABASE_URL must reference supabase.com.")
        self.assertIn("postgresql://", content, "DATABASE_URL must use postgresql:// scheme.")
        self.assertIn("sslmode=require", content, "DATABASE_URL must include sslmode=require.")


class TestAIServiceDockerCompose(unittest.TestCase):
    """7.6 - Verify AIService docker-compose.yml has no hardcoded DATABASE_URL."""

    def test_no_hardcoded_database_url(self):
        content = COMPOSE_FILE.read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if "DATABASE_URL=" in stripped:
                value = stripped.split("DATABASE_URL=", 1)[1]
                self.assertEqual(
                    value, "",
                    f"Found hardcoded DATABASE_URL in docker-compose.yml: {stripped!r}",
                )

    def test_env_file_supplies_database_url(self):
        content = COMPOSE_FILE.read_text()
        self.assertIn(
            "env_file",
            content,
            "AIService docker-compose.yml must use env_file to supply DATABASE_URL.",
        )


class TestPgvectorAbsentRaises(unittest.TestCase):
    """7.8 - Verify ensure_pgvector_schema() raises when vector extension is absent."""

    def test_raises_when_extension_absent(self):
        from sqlalchemy.exc import OperationalError

        mock_conn = MagicMock()
        mock_conn.execute.side_effect = OperationalError(
            "could not open extension control file",
            params=None,
            orig=Exception("extension vector not found"),
        )
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_ctx

        # Stub out heavy transitive dependencies so the module can be imported
        stub_modules = {
            "langchain_google_genai": MagicMock(),
            "langchain_openai": MagicMock(),
            "pgvector": MagicMock(),
            "pgvector.sqlalchemy": MagicMock(),
        }
        # Remove any cached copies of the modules we want to reload
        for mod in list(sys.modules.keys()):
            if mod.startswith("app.embeddings") or mod.startswith("app.core.database"):
                del sys.modules[mod]

        with patch.dict(sys.modules, stub_modules):
            with patch.dict(
                "os.environ",
                {"DATABASE_URL": "postgresql://u:p@host:5432/db", "INTERNAL_SERVICE_KEY": "k"},
                clear=False,
            ):
                import app.embeddings.vector_store as vs
                vs.engine = mock_engine
                with self.assertRaises(Exception):
                    vs.ensure_pgvector_schema()


if __name__ == "__main__":
    unittest.main()

