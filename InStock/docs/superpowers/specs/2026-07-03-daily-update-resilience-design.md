# Daily Update Resilience Design

## Goal

Make the desktop InStock updater safe and self-recovering when upstream data sources disconnect, return malformed rows, or database insertion fails. External sources may still be unavailable, but a failed run must never erase previously valid data or be reported as successful.

## Evidence and Root Cause

Recent update logs show one recurring failure family rather than unrelated daily bugs:

- On 2026-06-30, primary fund-flow and sector-flow requests repeatedly failed with empty JSON responses or disconnected HTTP connections.
- On 2026-07-02, duplicate individual and concept rows caused primary-key errors in `cn_stock_fund_flow` and `cn_stock_fund_flow_concept`.
- On 2026-07-03, a duplicate industry row caused a primary-key error in `cn_stock_fund_flow_industry`.

The current architecture amplifies these source problems:

1. Daily jobs delete the current date partition using one autocommit connection.
2. They insert replacement rows using a separate SQLAlchemy connection.
3. `insert_other_db_from_df` catches insertion exceptions and only logs them.
4. Job functions also catch exceptions, so the Python process usually exits with code zero.
5. The launcher detects some logged errors, but aborts on the first critical failure and has no retry or final database health check.

All relevant production tables use InnoDB, so a single-connection transaction can make partition replacement atomic.

## Chosen Architecture

Use defense in depth at three boundaries:

1. Validate incoming data before any database mutation.
2. Replace a date partition atomically inside one transaction.
3. Retry failed jobs and verify critical tables after the complete workflow.

This is preferred over continuing to add table-specific duplicate patches, which cannot address partial writes or swallowed exceptions. It is also preferred over staging-table renames, which add DDL locking and cleanup complexity unnecessary for date-partition updates.

## Components

### 1. Atomic partition replacement

Add a focused database API in `instock/lib/database.py`:

```python
replace_partition_from_df(
    data,
    table_name,
    partition_column,
    partition_value,
    key_columns,
    cols_type,
    write_index=False,
    primary_keys=None,
    indexs=None,
)
```

Before opening a transaction, the helper must reject:

- `None` or empty input;
- missing partition or key columns;
- null partition/key values;
- rows whose partition value differs from the requested partition;
- duplicate key-column combinations.

For an existing table, use one SQLAlchemy `engine.begin()` connection to:

1. Delete only the requested partition with a parameterized statement.
2. Append the DataFrame through the same connection.
3. Count the committed candidate partition inside the transaction.
4. Raise if the inserted count differs from the DataFrame row count.

Any exception rolls back both deletion and insertion. Log the table and error, then re-raise so callers and the launcher can observe failure.

For a missing table, create and populate it through the existing schema path, create its primary/index keys, verify the row count, and propagate any failure. There is no previous partition to preserve in this case.

The existing `insert_db_from_df` and `insert_other_db_from_df` APIs remain available for non-partition callers, but their caught exceptions must be re-raised after logging instead of being silently swallowed.

### 2. Daily-job migration

Replace every separate `DELETE` plus `insert_db_from_df` sequence used by the five jobs launched from the desktop shortcut:

- `basic_data_daily_job.py`
- `basic_data_other_daily_job.py`
- `selection_data_daily_job.py`
- `indicators_data_daily_job.py`
- `strategy_data_daily_job.py`

Each call supplies its actual date partition and primary-key columns. Existing source-specific normalization remains in place; the generic database validator becomes the last invariant boundary for all tables.

Job-level exception handlers must log a stable `DAILY_UPDATE_WRITE_FAILED` marker with the table or function name. They may finish local cleanup, but must not convert a failed write into a success message.

Jobs not launched by the desktop shortcut, web handlers, schemas, and analytical calculations are outside this change.

### 3. Launcher retry

Modify `update-instock-data.ps1` so `Invoke-InStockJob` attempts each Python job at most three times:

- Attempt 1 runs immediately.
- Attempt 2 waits 3 seconds.
- Attempt 3 waits 10 seconds.
- Each attempt gets separate captured stdout/stderr and an explicit log heading.
- A nonzero exit code, `ALL_SOURCES_FAILED`, `DAILY_UPDATE_WRITE_FAILED`, duplicate invariant marker, or database insertion error makes that attempt fail.
- Transient primary-source warnings remain warnings when a fallback succeeds.
- If a later attempt succeeds, the workflow continues without rerunning already completed jobs.
- If all attempts fail, the launcher stops, reports the failed job and log path, and preserves old database data through transactional rollback.

Retries operate at job granularity because each job is idempotent after atomic partition replacement.

### 4. Final health check

Add `instock/job/verify_daily_update.py` and run it after the five update jobs. It determines the expected latest trading date using the same trade-date library as the updater and validates these core tables:

- `cn_stock_spot`
- `cn_etf_spot`
- `cn_stock_fund_flow`
- `cn_stock_fund_flow_industry`
- `cn_stock_fund_flow_concept`

For each table it requires:

- at least one row for the expected date;
- no null key columns;
- total row count equal to distinct primary-key count.

The verifier prints a concise per-table summary and exits nonzero on failure. Selection, indicator, and strategy result tables are not required to be nonempty because legitimately no stocks may match on a given day; their writes are still protected by the atomic helper.

## Data Flow

The resulting workflow is:

1. Fetch primary source; use existing fallback when necessary.
2. Normalize source-specific rows.
3. Merge periods or calculate derived data.
4. Validate partition and keys generically.
5. Delete and insert within one transaction.
6. Roll back automatically on any write or count mismatch.
7. Retry only the failed job up to three attempts.
8. Run final database health checks before reporting success.

## Error Semantics

- A primary-source failure followed by successful fallback is a warning.
- Invalid incoming data is a critical failure before database mutation.
- A database error is logged once with `DAILY_UPDATE_WRITE_FAILED` and re-raised.
- A failed attempt may be retried, but it cannot damage the prior partition.
- “Done” is printed only after all jobs and the final health check succeed.
- After three failures, the user receives one actionable final error and the existing data remains usable.

## Testing

### Unit tests

Add focused tests for:

1. Empty input, missing keys, null keys, mixed partitions, and duplicate keys are rejected before database access.
2. A valid DataFrame executes delete, append, and row-count verification on the same transaction connection.
3. An insertion exception is re-raised rather than swallowed.
4. Migrated jobs call the atomic helper and no longer issue independent delete statements.
5. The health checker accepts complete unique partitions and rejects empty or duplicated partitions.

### Fault-injection integration test

Use a uniquely named scratch InnoDB table in the local MariaDB instance:

1. Insert a known old partition.
2. Force the replacement insert to fail after deletion.
3. Verify the transaction rolls back and the old rows remain.
4. Run a valid replacement and verify the new partition is complete and unique.
5. Drop the scratch table in `finally` cleanup.

### End-to-end verification

Run the desktop workflow with its real virtual environment and database settings, then confirm:

- all Python tests pass;
- fault injection proves rollback;
- the five jobs finish;
- the final health checker exits zero;
- no critical markers or duplicate-key errors appear in the new log;
- the launcher prints `Done` only after validation.

## Scope and Success Criteria

This change hardens the existing updater; it does not replace upstream providers or promise that unavailable external data can always be fetched.

Success means:

- source or database failure never destroys an existing valid date partition;
- transient failures receive bounded automatic retries;
- persistent failures are reported clearly instead of silently succeeding;
- successful runs end with verified current, nonempty, unique core data;
- no unrelated schemas, calculations, or user data are changed.
