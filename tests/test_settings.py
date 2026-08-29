"""Tests for config.settings.Settings.database_url."""
from __future__ import annotations

from config.settings import Settings


def test_database_url_without_sslmode():
    s = Settings(db_host="myhost", db_port=5432, db_name="mydb",
                 db_user="myuser", db_password="mypass", db_sslmode=None)
    assert s.database_url == "postgresql+psycopg2://myuser:mypass@myhost:5432/mydb"
    assert "sslmode" not in s.database_url


def test_database_url_with_sslmode():
    s = Settings(db_host="myhost", db_port=5432, db_name="mydb",
                 db_user="myuser", db_password="mypass", db_sslmode="require")
    assert s.database_url == "postgresql+psycopg2://myuser:mypass@myhost:5432/mydb?sslmode=require"


def test_database_url_defaults_to_no_sslmode_when_unset():
    s = Settings(db_host="localhost", db_sslmode=None)
    assert s.db_sslmode is None
    assert "sslmode" not in s.database_url
