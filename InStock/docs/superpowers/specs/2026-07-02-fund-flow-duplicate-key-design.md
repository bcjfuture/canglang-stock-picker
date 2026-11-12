# Fund Flow Duplicate-Key Fix Design

## Goal

Make the daily individual fund-flow update tolerate duplicate stock rows returned by either data source while preserving the table invariant that each `(date, code)` pair is unique.

## Observed Failure

The 2026-07-02 fallback responses contained duplicate stock codes before database insertion:

- Current-period data: 4,584 rows, 4,579 unique codes, five duplicated codes.
- Three-day data: 4,585 rows, 4,584 unique codes; code `600141` appeared twice.
- Five-day and ten-day data had no duplicates.
- Merging the four periods produced 4,585 rows but only 4,579 unique codes.

The bulk insert then failed on primary key `(date, code)` and left 2,733 rows for 2026-07-02 in `cn_stock_fund_flow`.

## Design

### Source-boundary normalization

Add a small helper in `instock/core/stockfetch.py` that normalizes individual fund-flow results before they leave `fetch_stocks_fund_flow`:

- Return `None` unchanged.
- Require a `code` column.
- Detect duplicate `code` values.
- Log one warning with the period and number of rows removed.
- Keep the last row for each code, matching the existing project convention used by stock-selection data.
- Return a copied/reset DataFrame so callers do not receive a duplicated index.

Apply this helper to both the primary Eastmoney result and the Tonghuashun fallback result. This prevents many-to-many expansion during the later period merge.

### Pre-write invariant

In `save_nph_stock_fund_flow_data`, validate the fully merged DataFrame before deleting current database rows. If duplicate codes remain, log a critical error and return without touching the existing date partition. Do not silently discard duplicates at this layer; duplicates here would indicate a merge or normalization regression.

This ordering protects an existing complete partition from being deleted when incoming data is invalid.

### Database recovery

No schema change or upsert is required. After the code fix passes tests, rerunning the existing daily job will delete the incomplete 2026-07-02 partition and insert the complete unique dataset through the existing path.

## Error Handling

- Duplicate rows at the source boundary are recoverable and produce a warning.
- Duplicate rows after merging are a critical invariant violation and prevent deletion/insertion.
- Existing all-sources-failed behavior remains unchanged.
- Database exceptions remain visible to the desktop launcher through its existing critical-error scan.

## Tests

Add tests to `tests/test_fund_flow_fallback.py`:

1. Mock a fallback response containing two rows for the same code and verify `fetch_stocks_fund_flow` returns one row, keeping the last value.
2. Build merged period results containing a duplicate after normalization is bypassed and verify the pre-write invariant does not call database delete or insert.

Run the focused tests first, then the complete test suite. Finally run `basic_data_other_daily_job.py` through the same environment as the desktop shortcut and verify:

- No duplicate-key error is logged.
- The job exits without critical errors.
- `cn_stock_fund_flow` has equal total and distinct-code counts for 2026-07-02.
- Later daily jobs complete through the desktop update workflow.

## Scope

Only individual fund-flow normalization, its pre-write guard, and regression tests are in scope. Generic database insert behavior, sector fund-flow processing, schemas, and unrelated update jobs remain unchanged.
