"""Unit tests for json_columns configuration across SQL destinations.

Tests the explicit json_columns feature and backward-compatible heuristic
behaviour for PostgreSQL, MySQL, and ClickHouse destinations.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from drt.config.models import (
    ClickHouseDestinationConfig,
    MySQLDestinationConfig,
    PostgresDestinationConfig,
    SyncOptions,
)
from drt.destinations.clickhouse import ClickHouseDestination
from drt.destinations.mysql import MySQLDestination, _serialize_value as mysql_serialize
from drt.destinations.postgres import PostgresDestination


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pg_config(**overrides: Any) -> PostgresDestinationConfig:
    defaults: dict[str, Any] = {
        "type": "postgres",
        "host": "localhost",
        "dbname": "testdb",
        "user": "testuser",
        "password": "testpass",
        "table": "public.scores",
        "upsert_key": ["id"],
    }
    defaults.update(overrides)
    return PostgresDestinationConfig(**defaults)


def _mysql_config(**overrides: Any) -> MySQLDestinationConfig:
    defaults: dict[str, Any] = {
        "type": "mysql",
        "host": "localhost",
        "dbname": "testdb",
        "user": "testuser",
        "password": "testpass",
        "table": "test_table",
        "upsert_key": ["id"],
    }
    defaults.update(overrides)
    return MySQLDestinationConfig(**defaults)


def _ch_config(**overrides: Any) -> ClickHouseDestinationConfig:
    defaults: dict[str, Any] = {
        "type": "clickhouse",
        "host": "localhost",
        "database": "default",
        "user": "default",
        "password": "",
        "table": "test_table",
    }
    defaults.update(overrides)
    return ClickHouseDestinationConfig(**defaults)


def _options(**kwargs: Any) -> SyncOptions:
    return SyncOptions(**kwargs)


# ---------------------------------------------------------------------------
# Config validation — json_columns field
# ---------------------------------------------------------------------------


class TestJsonColumnsConfig:
    """Verify json_columns field exists and accepts expected values."""

    def test_postgres_json_columns_defaults_to_none(self) -> None:
        config = _pg_config()
        assert config.json_columns is None

    def test_postgres_json_columns_accepts_list(self) -> None:
        config = _pg_config(json_columns=["metadata", "tags"])
        assert config.json_columns == ["metadata", "tags"]

    def test_postgres_json_columns_empty_list(self) -> None:
        config = _pg_config(json_columns=[])
        assert config.json_columns == []

    def test_mysql_json_columns_defaults_to_none(self) -> None:
        config = _mysql_config()
        assert config.json_columns is None

    def test_mysql_json_columns_accepts_list(self) -> None:
        config = _mysql_config(json_columns=["data"])
        assert config.json_columns == ["data"]

    def test_clickhouse_json_columns_defaults_to_none(self) -> None:
        config = _ch_config()
        assert config.json_columns is None

    def test_clickhouse_json_columns_accepts_list(self) -> None:
        config = _ch_config(json_columns=["payload"])
        assert config.json_columns == ["payload"]


# ---------------------------------------------------------------------------
# MySQL _serialize_value function (standalone)
# ---------------------------------------------------------------------------


class TestMySqlSerializeValue:
    """Test the module-level _serialize_value helper."""

    def test_explicit_column_serialized(self) -> None:
        value = {"key": "val"}
        result = mysql_serialize(value, "data", {"data"})
        assert json.loads(result) == {"key": "val"}

    def test_explicit_column_not_listed_passthrough(self) -> None:
        value = {"key": "val"}
        result = mysql_serialize(value, "other_col", {"data"})
        assert result is value  # passthrough — same object

    def test_heuristic_mode_dict_auto_serialized(self) -> None:
        value = {"key": "val"}
        result = mysql_serialize(value, "any_col", None)
        assert json.loads(result) == {"key": "val"}

    def test_heuristic_mode_list_auto_serialized(self) -> None:
        value = [1, 2, 3]
        result = mysql_serialize(value, "any_col", None)
        assert json.loads(result) == [1, 2, 3]

    def test_heuristic_mode_scalar_passthrough(self) -> None:
        result = mysql_serialize("hello", "any_col", None)
        assert result == "hello"

    def test_none_value_passthrough(self) -> None:
        result = mysql_serialize(None, "data", {"data"})
        assert result is None

    def test_explicit_column_scalar_serialized_to_json_string(self) -> None:
        """When json_columns is set, even scalar values get JSON-serialized."""
        result = mysql_serialize(42, "count", {"count"})
        assert result == "42"

    def test_nested_dict_serialized(self) -> None:
        value = {"outer": {"inner": [1, 2, 3]}}
        result = mysql_serialize(value, "data", {"data"})
        assert json.loads(result) == {"outer": {"inner": [1, 2, 3]}}

    def test_empty_dict_serialized(self) -> None:
        result = mysql_serialize({}, "data", {"data"})
        assert result == "{}"

    def test_empty_list_serialized(self) -> None:
        result = mysql_serialize([], "data", {"data"})
        assert result == "[]"


# ---------------------------------------------------------------------------
# PostgreSQL destination — _serialize_value
# ---------------------------------------------------------------------------


class TestPostgresSerializeValue:
    """Test PostgresDestination._serialize_value method."""

    def _dest(self, json_columns: list[str] | None = None) -> PostgresDestination:
        dest = PostgresDestination()
        dest._json_columns = set(json_columns) if json_columns else None
        # Inject a mock Json wrapper to avoid needing real psycopg2
        dest._json_wrapper = MagicMock(return_value="<JsonWrapped>")
        return dest

    def test_explicit_column_wrapped_with_json(self) -> None:
        dest = self._dest(json_columns=["metadata"])
        result = dest._serialize_value({"key": "val"}, "metadata")
        dest._json_wrapper.assert_called_once_with({"key": "val"})
        assert result == "<JsonWrapped>"

    def test_non_listed_column_passthrough(self) -> None:
        dest = self._dest(json_columns=["metadata"])
        result = dest._serialize_value({"key": "val"}, "other_col")
        dest._json_wrapper.assert_not_called()
        assert result == {"key": "val"}

    def test_heuristic_mode_dict_auto_wrapped(self) -> None:
        dest = self._dest(json_columns=None)
        result = dest._serialize_value({"key": "val"}, "any_col")
        dest._json_wrapper.assert_called_once_with({"key": "val"})

    def test_heuristic_mode_scalar_passthrough(self) -> None:
        dest = self._dest(json_columns=None)
        result = dest._serialize_value("hello", "any_col")
        dest._json_wrapper.assert_not_called()
        assert result == "hello"

    def test_none_value_returns_none(self) -> None:
        dest = self._dest(json_columns=["data"])
        result = dest._serialize_value(None, "data")
        assert result is None
        dest._json_wrapper.assert_not_called()

    def test_explicit_column_scalar_gets_json_wrapped(self) -> None:
        """When json_columns is set, even scalar values get wrapped."""
        dest = self._dest(json_columns=["count"])
        result = dest._serialize_value(42, "count")
        dest._json_wrapper.assert_called_once_with(42)


# ---------------------------------------------------------------------------
# PostgreSQL destination — integration (load)
# ---------------------------------------------------------------------------


class TestPostgresJsonColumnsLoad:
    """Verify json_columns flows through PostgresDestination.load()."""

    @patch("drt.destinations.postgres.PostgresDestination._connect")
    def test_json_columns_serialize_specified_columns(self, mock_connect: MagicMock) -> None:
        conn = MagicMock()
        conn.cursor.return_value = MagicMock()
        mock_connect.return_value = conn

        mock_json = MagicMock(side_effect=lambda v: f"<Json({v})>")
        records = [
            {"id": 1, "name": "test", "metadata": {"k": "v"}, "tags": ["a", "b"]},
        ]
        config = _pg_config(json_columns=["metadata", "tags"])
        dest = PostgresDestination()
        dest._json_wrapper = mock_json
        dest.load(records, config, _options())

        cur = conn.cursor()
        assert cur.execute.call_count == 1
        _sql, values = cur.execute.call_args[0]
        assert values[0] == 1
        assert values[1] == "test"
        assert mock_json.call_count == 2

    @patch("drt.destinations.postgres.PostgresDestination._connect")
    def test_no_json_columns_heuristic_mode(self, mock_connect: MagicMock) -> None:
        conn = MagicMock()
        conn.cursor.return_value = MagicMock()
        mock_connect.return_value = conn

        mock_json = MagicMock(side_effect=lambda v: f"<Json({v})>")
        records = [
            {"id": 1, "data": {"k": "v"}, "score": 0.9},
        ]
        config = _pg_config()  # json_columns is None
        dest = PostgresDestination()
        dest._json_wrapper = mock_json
        dest.load(records, config, _options())

        cur = conn.cursor()
        _sql, values = cur.execute.call_args[0]
        assert values[0] == 1
        assert values[2] == 0.9
        mock_json.assert_called_once_with({"k": "v"})

    @patch("drt.destinations.postgres.PostgresDestination._connect")
    def test_mixed_record_partial_json_columns(self, mock_connect: MagicMock) -> None:
        """Only metadata is in json_columns; data dict passes through as-is."""
        conn = MagicMock()
        conn.cursor.return_value = MagicMock()
        mock_connect.return_value = conn

        mock_json = MagicMock(side_effect=lambda v: f"<Json({v})>")
        records = [
            {"id": 1, "metadata": {"a": 1}, "raw_data": {"b": 2}, "name": "x"},
        ]
        config = _pg_config(json_columns=["metadata"])
        dest = PostgresDestination()
        dest._json_wrapper = mock_json
        dest.load(records, config, _options())

        cur = conn.cursor()
        _sql, values = cur.execute.call_args[0]
        mock_json.assert_called_once_with({"a": 1})
        assert values[2] == {"b": 2}  # raw_data passthrough


# ---------------------------------------------------------------------------
# MySQL destination — integration (load)
# ---------------------------------------------------------------------------


class TestMySQLJsonColumnsLoad:
    """Verify json_columns flows through MySQLDestination.load()."""

    @patch("drt.destinations.mysql.MySQLDestination._connect")
    def test_json_columns_serialize_specified_columns(self, mock_connect: MagicMock) -> None:
        conn = MagicMock()
        conn.cursor.return_value = MagicMock()
        mock_connect.return_value = conn

        records = [
            {"id": 1, "name": "test", "metadata": {"k": "v"}, "tags": ["a", "b"]},
        ]
        config = _mysql_config(json_columns=["metadata", "tags"])
        MySQLDestination().load(records, config, _options())

        cur = conn.cursor()
        _sql, values = cur.execute.call_args[0]
        assert values[0] == 1
        assert values[1] == "test"
        assert json.loads(values[2]) == {"k": "v"}
        assert json.loads(values[3]) == ["a", "b"]

    @patch("drt.destinations.mysql.MySQLDestination._connect")
    def test_no_json_columns_heuristic_mode(self, mock_connect: MagicMock) -> None:
        conn = MagicMock()
        conn.cursor.return_value = MagicMock()
        mock_connect.return_value = conn

        records = [
            {"id": 1, "data": {"k": "v"}, "score": 0.9},
        ]
        config = _mysql_config()  # json_columns is None
        MySQLDestination().load(records, config, _options())

        cur = conn.cursor()
        _sql, values = cur.execute.call_args[0]
        assert values[0] == 1
        assert json.loads(values[1]) == {"k": "v"}
        assert values[2] == 0.9

    @patch("drt.destinations.mysql.MySQLDestination._connect")
    def test_dict_not_in_json_columns_passthrough(self, mock_connect: MagicMock) -> None:
        """When json_columns is set, dict values on unlisted columns pass through."""
        conn = MagicMock()
        conn.cursor.return_value = MagicMock()
        mock_connect.return_value = conn

        records = [
            {"id": 1, "data": {"b": 2}, "metadata": {"a": 1}},
        ]
        config = _mysql_config(json_columns=["metadata"])
        MySQLDestination().load(records, config, _options())

        cur = conn.cursor()
        _sql, values = cur.execute.call_args[0]
        assert values[1] == {"b": 2}  # data passthrough (dict, not in json_columns)
        assert json.loads(values[2]) == {"a": 1}  # metadata serialized

    @patch("drt.destinations.mysql.MySQLDestination._connect")
    def test_nested_structures_serialized(self, mock_connect: MagicMock) -> None:
        """Deeply nested dicts and lists are serialized correctly."""
        conn = MagicMock()
        conn.cursor.return_value = MagicMock()
        mock_connect.return_value = conn

        nested = {"level1": {"level2": [1, 2, {"level3": "deep"}]}}
        records = [{"id": 1, "payload": nested}]
        config = _mysql_config(json_columns=["payload"])
        MySQLDestination().load(records, config, _options())

        cur = conn.cursor()
        _sql, values = cur.execute.call_args[0]
        assert json.loads(values[1]) == nested

    @patch("drt.destinations.mysql.MySQLDestination._connect")
    def test_empty_dict_and_list_serialized(self, mock_connect: MagicMock) -> None:
        conn = MagicMock()
        conn.cursor.return_value = MagicMock()
        mock_connect.return_value = conn

        records = [{"id": 1, "data": {}, "items": []}]
        config = _mysql_config(json_columns=["data", "items"])
        MySQLDestination().load(records, config, _options())

        cur = conn.cursor()
        _sql, values = cur.execute.call_args[0]
        assert values[1] == "{}"
        assert values[2] == "[]"

    @patch("drt.destinations.mysql.MySQLDestination._connect")
    def test_none_value_not_serialized(self, mock_connect: MagicMock) -> None:
        conn = MagicMock()
        conn.cursor.return_value = MagicMock()
        mock_connect.return_value = conn

        records = [{"id": 1, "data": None}]
        config = _mysql_config(json_columns=["data"])
        MySQLDestination().load(records, config, _options())

        cur = conn.cursor()
        _sql, values = cur.execute.call_args[0]
        assert values[1] is None


# ---------------------------------------------------------------------------
# ClickHouse destination — integration (load)
# ---------------------------------------------------------------------------


class TestClickHouseJsonColumnsLoad:
    """Verify json_columns flows through ClickHouseDestination.load()."""

    @patch("drt.destinations.clickhouse.ClickHouseDestination._connect")
    def test_json_columns_serialize_specified_columns(self, mock_connect: MagicMock) -> None:
        client = MagicMock()
        mock_connect.return_value = client

        records = [
            {"id": 1, "name": "test", "metadata": {"k": "v"}, "tags": ["a", "b"]},
        ]
        config = _ch_config(json_columns=["metadata", "tags"])
        ClickHouseDestination().load(records, config, _options())

        call_args = client.insert.call_args
        row_data = call_args[0][1]  # list[list[Any]]
        assert row_data[0][0] == 1
        assert row_data[0][1] == "test"
        assert json.loads(row_data[0][2]) == {"k": "v"}
        assert json.loads(row_data[0][3]) == ["a", "b"]

    @patch("drt.destinations.clickhouse.ClickHouseDestination._connect")
    def test_no_json_columns_heuristic_mode(self, mock_connect: MagicMock) -> None:
        client = MagicMock()
        mock_connect.return_value = client

        records = [
            {"id": 1, "data": {"k": "v"}, "score": 0.9},
        ]
        config = _ch_config()
        ClickHouseDestination().load(records, config, _options())

        call_args = client.insert.call_args
        row_data = call_args[0][1]
        assert row_data[0][0] == 1
        assert json.loads(row_data[0][1]) == {"k": "v"}
        assert row_data[0][2] == 0.9

    @patch("drt.destinations.clickhouse.ClickHouseDestination._connect")
    def test_dict_not_in_json_columns_passthrough(self, mock_connect: MagicMock) -> None:
        client = MagicMock()
        mock_connect.return_value = client

        records = [
            {"id": 1, "data": {"b": 2}, "metadata": {"a": 1}},
        ]
        config = _ch_config(json_columns=["metadata"])
        ClickHouseDestination().load(records, config, _options())

        call_args = client.insert.call_args
        row_data = call_args[0][1]
        assert row_data[0][1] == {"b": 2}  # data passthrough
        assert json.loads(row_data[0][2]) == {"a": 1}  # metadata serialized
