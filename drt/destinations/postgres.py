"""PostgreSQL destination — upsert rows into a PostgreSQL table.

Uses INSERT ... ON CONFLICT (upsert_key) DO UPDATE SET ... for idempotent writes.
Requires: pip install drt-core[postgres]

Example sync YAML:

    destination:
      type: postgres
      host_env: TARGET_PG_HOST
      dbname_env: TARGET_PG_DBNAME
      user_env: TARGET_PG_USER
      password_env: TARGET_PG_PASSWORD
      table: public.analytics_scores
      upsert_key: [id]
      json_columns: [metadata, tags]  # optional: explicit JSON serialization

JSON serialization behaviour:
  - If ``json_columns`` is set, only the listed columns have their values
    wrapped with ``psycopg2.extras.Json()``; all other columns pass through
    as-is (including dict/list values).
  - If ``json_columns`` is ``None`` (default), the legacy heuristic applies:
    dict and list values are auto-wrapped for any column.
"""

from __future__ import annotations

import json
from typing import Any

from drt.config.credentials import resolve_env
from drt.config.models import DestinationConfig, PostgresDestinationConfig, SyncOptions
from drt.destinations.base import SyncResult
from drt.destinations.row_errors import RowError


class PostgresDestination:
    """Upsert records into a PostgreSQL table."""

    def __init__(self) -> None:
        self._json_columns: set[str] | None = None
        self._json_wrapper: Any = None  # cached psycopg2.extras.Json

    def load(
        self,
        records: list[dict[str, Any]],
        config: DestinationConfig,
        sync_options: SyncOptions,
    ) -> SyncResult:
        assert isinstance(config, PostgresDestinationConfig)
        if not records:
            return SyncResult()

        # Cache json_columns set for the lifetime of this load call
        self._json_columns = set(config.json_columns) if config.json_columns else None

        conn = self._connect(config)
        result = SyncResult()

        try:
            cur = conn.cursor()
            columns = list(records[0].keys())
            update_cols = [c for c in columns if c not in config.upsert_key]

            sql = self._build_upsert_sql(config.table, columns, config.upsert_key, update_cols)

            for i, record in enumerate(records):
                try:
                    values = [
                        self._serialize_value(record.get(c), c)
                        for c in columns
                    ]
                    cur.execute(sql, values)
                    result.success += 1
                except Exception as e:
                    result.failed += 1
                    result.row_errors.append(
                        RowError(
                            batch_index=i,
                            record_preview=json.dumps(record, default=str)[:200],
                            http_status=None,
                            error_message=str(e),
                        )
                    )
                    if sync_options.on_error == "fail":
                        conn.rollback()
                        return result
                    # on_error == "skip": rollback this row, continue
                    conn.rollback()
                    # Re-open transaction for next rows
                    cur = conn.cursor()
                    continue

            conn.commit()
        finally:
            conn.close()

        return result

    def _serialize_value(self, value: Any, column: str) -> Any:
        """Prepare *value* for a parameterised INSERT.

        When ``json_columns`` is configured:
          - listed columns → wrapped with ``psycopg2.extras.Json()``
          - other columns → passed through as-is

        When ``json_columns`` is ``None`` (backward-compatible default):
          - dict/list values → auto-wrapped with ``psycopg2.extras.Json()``
          - everything else → passed through as-is
        """
        if value is None:
            return None

        # Import psycopg2.extras lazily but cache the reference on the
        # class so repeated calls within the same load() don't re-import.
        if self._json_wrapper is None:
            try:
                import psycopg2.extras as _extras  # type: ignore[import-untyped]

                self._json_wrapper = _extras.Json
            except ImportError:
                # psycopg2 not installed — fall back to passthrough
                return value

        if self._json_columns is not None:
            # Explicit mode: only serialize listed columns
            if column in self._json_columns:
                return self._json_wrapper(value)
            return value
        else:
            # Heuristic mode: auto-serialize dict/list
            if isinstance(value, (dict, list)):
                return self._json_wrapper(value)
            return value

    @staticmethod
    def _build_upsert_sql(
        table: str,
        columns: list[str],
        upsert_key: list[str],
        update_cols: list[str],
    ) -> str:
        """Build INSERT ... ON CONFLICT DO UPDATE SQL."""
        cols_str = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join(["%s"] * len(columns))
        conflict_str = ", ".join(f'"{c}"' for c in upsert_key)

        if update_cols:
            set_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
            return (
                f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders}) "
                f"ON CONFLICT ({conflict_str}) DO UPDATE SET {set_clause}"
            )
        # All columns are part of the key — just ignore duplicates
        return (
            f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_str}) DO NOTHING"
        )

    @staticmethod
    def _connect(config: PostgresDestinationConfig) -> Any:
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "PostgreSQL destination requires: pip install drt-core[postgres]"
            ) from e

        # Connection string takes precedence
        conn_str = (
            resolve_env(None, config.connection_string_env)
            if config.connection_string_env
            else None
        )
        if conn_str:
            return psycopg2.connect(conn_str)

        # Fall back to individual parameters
        host = resolve_env(config.host, config.host_env)
        dbname = resolve_env(config.dbname, config.dbname_env)
        user = resolve_env(config.user, config.user_env)
        password = resolve_env(config.password, config.password_env)

        if not host:
            raise ValueError("PostgreSQL destination: host could not be resolved.")
        if not dbname:
            raise ValueError("PostgreSQL destination: dbname could not be resolved.")

        kwargs: dict[str, Any] = {
            "host": host,
            "port": config.port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }

        if config.ssl and config.ssl.enabled:
            kwargs["sslmode"] = "require"
            ca = resolve_env(None, config.ssl.ca_env)
            if ca:
                kwargs["sslrootcert"] = ca
            cert = resolve_env(None, config.ssl.cert_env)
            if cert:
                kwargs["sslcert"] = cert
            key = resolve_env(None, config.ssl.key_env)
            if key:
                kwargs["sslkey"] = key

        return psycopg2.connect(**kwargs)
