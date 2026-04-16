"""ClickHouse destination — insert rows into a ClickHouse table.

Uses clickhouse-connect for HTTP-based inserts. Each record is inserted
individually to enable row-level error tracking (consistent with the
PostgreSQL and MySQL destination pattern).

Deduplication is handled by ClickHouse's ReplacingMergeTree engine at merge
time — the destination performs simple INSERTs.

Requires: pip install drt-core[clickhouse]

Example sync YAML:

    destination:
      type: clickhouse
      host_env: TARGET_CH_HOST
      database_env: TARGET_CH_DATABASE
      user_env: TARGET_CH_USER
      password_env: TARGET_CH_PASSWORD
      table: analytics_scores
      json_columns: [metadata, tags]  # optional: explicit JSON serialization

JSON serialization behaviour:
  - If ``json_columns`` is set, only the listed columns have their values
    serialized with ``json.dumps()``; all other columns pass through as-is.
  - If ``json_columns`` is ``None`` (default), the legacy heuristic applies:
    dict and list values are auto-serialized for any column.
"""

from __future__ import annotations

import json
from typing import Any

from drt.config.credentials import resolve_env
from drt.config.models import ClickHouseDestinationConfig, DestinationConfig, SyncOptions
from drt.destinations.base import SyncResult
from drt.destinations.row_errors import RowError


class ClickHouseDestination:
    """Insert records into a ClickHouse table."""

    def __init__(self) -> None:
        self._json_columns: set[str] | None = None

    def load(
        self,
        records: list[dict[str, Any]],
        config: DestinationConfig,
        sync_options: SyncOptions,
    ) -> SyncResult:
        assert isinstance(config, ClickHouseDestinationConfig)
        if not records:
            return SyncResult()

        # Cache json_columns set for the lifetime of this load call
        self._json_columns = set(config.json_columns) if config.json_columns else None

        client = self._connect(config)
        result = SyncResult()

        try:
            columns = list(records[0].keys())

            # TODO: batch insert with fallback to row-by-row on error
            for i, record in enumerate(records):
                try:
                    row = [
                        [self._serialize_value(record.get(c), c) for c in columns]
                    ]
                    client.insert(config.table, row, column_names=columns)
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
                        return result
                    continue
        finally:
            client.close()

        return result

    def _serialize_value(self, value: Any, column: str) -> Any:
        """Prepare *value* for insertion.

        When ``json_columns`` is configured:
          - listed columns → serialized with ``json.dumps()``
          - other columns → passed through as-is

        When ``json_columns`` is ``None`` (backward-compatible default):
          - dict/list values → auto-serialized with ``json.dumps()``
          - everything else → passed through as-is
        """
        if value is None:
            return None

        if self._json_columns is not None:
            if column in self._json_columns:
                return json.dumps(value, ensure_ascii=False)
            return value
        else:
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return value

    @staticmethod
    def _connect(config: ClickHouseDestinationConfig) -> Any:
        try:
            import clickhouse_connect  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "ClickHouse destination requires: pip install drt-core[clickhouse]"
            ) from e

        # Connection string takes precedence
        conn_str = (
            resolve_env(None, config.connection_string_env)
            if config.connection_string_env
            else None
        )
        if conn_str:
            return clickhouse_connect.get_client(dsn=conn_str)

        # Fall back to individual parameters
        host = resolve_env(config.host, config.host_env)
        database = resolve_env(config.database, config.database_env)
        user = resolve_env(config.user, config.user_env)
        password = resolve_env(config.password, config.password_env) or ""

        if not host:
            raise ValueError("ClickHouse destination: host could not be resolved.")
        if not database:
            raise ValueError("ClickHouse destination: database could not be resolved.")

        return clickhouse_connect.get_client(
            host=host,
            port=config.port,
            database=database,
            username=user or "default",
            password=password,
            secure=config.secure,
        )
