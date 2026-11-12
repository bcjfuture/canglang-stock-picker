# Sector Fund Flow Duplicate-Key Fix Design

## Goal

Make the daily sector fund-flow update tolerate duplicate sector names returned by either data source while preserving the database invariant that each `(date, name)` pair is unique. The fix covers both industry and concept fund-flow tables.

## Observed Failure

The 2026-07-03 industry update failed while inserting `cn_stock_fund_flow_industry` because the incoming data contained duplicate sector names. The logged duplicate was `通信服务`; a later read-only sample contained two `风电设备` rows in the ten-day period instead. This variation shows that the fallback source is unstable rather than one fixed sector being malformed.

In the later sample:

- Current period: 90 rows and 90 unique names.
- Five-day period: 41 rows and 41 unique names.
- Ten-day period: 90 rows and 89 unique names; `风电设备` appeared twice with slightly different change rates.
- Merging the periods produced 91 rows but only 90 unique names.

The existing job deletes the current date partition before inserting the merged result. The insert then fails on primary key `(date, name)`, leaving the industry table empty for that date.

## Chosen Approach

Normalize sector rows at the source boundary and enforce a strict invariant immediately before any database mutation. This is preferred over silently dropping duplicates only after merging or changing the database insert to an upsert, because it prevents many-to-many merge expansion while still exposing regressions before existing data is deleted.

## Design

### Source-boundary normalization

Add a focused helper in `instock/core/stockfetch.py` for sector fund-flow results:

- Return `None` unchanged.
- Require a `name` column.
- Detect duplicate `name` values.
- Log one warning containing the sector type, period, and number of removed rows.
- Keep the last row for each name, matching the existing individual fund-flow normalization convention.
- Return a copied DataFrame with a reset index.

Apply the helper to both the primary Eastmoney result and the Tonghuashun fallback result inside `fetch_stocks_sector_fund_flow`. This guarantees that each period has at most one row per sector name before the period merge.

### Pre-write invariant

In `stock_sector_fund_flow_data`, validate the fully merged DataFrame before inserting the date column and before selecting, deleting, or writing a database table.

If duplicate names remain:

- Log a critical error containing a stable marker, the sector type, and duplicate-name count.
- Return without calling the database existence check, delete, or insert functions.
- Do not silently deduplicate at this layer; a duplicate here indicates a normalization or merge regression.

This ordering ensures invalid incoming data cannot erase a previously valid date partition.

### Database recovery

No schema change or upsert is required. After the fix passes tests, rerun the affected daily job. Its existing delete-and-replace path will recreate the 2026-07-03 industry partition from normalized data. The already complete concept partition may be refreshed safely by the same job.

## Error Handling

- Source-boundary duplicate rows are recoverable and produce a warning.
- Post-merge duplicate rows are a critical invariant violation and block all database mutation for that sector type.
- Empty-period and all-sources-failed behavior remain unchanged.
- Existing database exceptions remain visible to the desktop launcher through its critical-error scan.

## Tests

Extend `tests/test_fund_flow_fallback.py` with focused regression tests:

1. Mock a sector fallback response containing two rows with the same `name`; verify the fetch function returns one row and keeps the last value.
2. Supply period results that still produce duplicate names and verify the job guard does not call database existence checks, deletion, or insertion.
3. Keep existing individual fund-flow tests unchanged to confirm the new helper does not alter that path.

Verification proceeds from narrow to broad:

- Run the focused sector regression tests.
- Run the complete test suite.
- Run `basic_data_other_daily_job.py` with the same virtual environment and database variables as the desktop shortcut.
- Verify `cn_stock_fund_flow_industry` for 2026-07-03 has a positive row count equal to its distinct-name count.
- Verify no duplicate-key or critical sector-flow error appears in the new job log.

## Scope

Only sector fund-flow normalization, the pre-write invariant, regression tests, and recovery of the affected 2026-07-03 data are in scope. Generic database insert behavior, schemas, unrelated jobs, and source-ranking logic remain unchanged.
