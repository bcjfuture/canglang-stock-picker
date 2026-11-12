# Sector Fund Flow Duplicate-Key Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent duplicate sector names from causing `(date, name)` primary-key failures and safely restore the 2026-07-03 industry fund-flow partition.

**Architecture:** Normalize each sector-period DataFrame at the fetch boundary by keeping the last row for each `name`. Add a strict post-merge uniqueness guard before any database call so malformed data cannot delete a valid partition.

**Tech Stack:** Python 3, pandas, unittest, unittest.mock, MariaDB, PowerShell

---

## File Map

- Modify `tests/test_fund_flow_fallback.py`: add source-boundary and pre-write regression tests.
- Modify `instock/core/stockfetch.py`: add and apply sector-row normalization.
- Modify `instock/job/basic_data_other_daily_job.py`: enforce merged-name uniqueness before database mutation.
- No schema, launcher, or generic database-layer changes.

### Task 1: Source-Boundary Sector Deduplication

**Files:**
- Modify: `tests/test_fund_flow_fallback.py`
- Modify: `instock/core/stockfetch.py:184-204`

- [ ] **Step 1: Write the failing fallback deduplication test**

Add a test that builds a sector DataFrame using the configured period columns, duplicates one `name` with different `change_rate_10` values, forces the primary source to fail, and asserts that the fallback result contains one row with the last value:

```python
def test_stockfetch_deduplicates_sector_fallback_rows_by_name(self):
    columns = list(tablestructure.CN_STOCK_SECTOR_FUND_FLOW[1][2]["columns"])
    first = {column: None for column in columns}
    first.update({"name": "风电设备", "change_rate_10": -2.29})
    second = first.copy()
    second["change_rate_10"] = -2.26
    fallback = pd.DataFrame([first, second], columns=columns)

    with patch.object(
        stockfetch.sff,
        "stock_sector_fund_flow_rank",
        side_effect=ConnectionError("primary unavailable"),
    ), patch(
        "instock.core.fund_flow_fallback.fetch_sector_rank",
        return_value=fallback,
    ):
        result = stockfetch.fetch_stocks_sector_fund_flow(0, 2)

    self.assertEqual(len(result.index), 1)
    self.assertEqual(result.iloc[0]["name"], "风电设备")
    self.assertEqual(result.iloc[0]["change_rate_10"], -2.26)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
$env:PYTHONPATH='C:\InStockLocal\InStock'
& 'C:\InStockLocal\venv\Scripts\python.exe' -m unittest tests.test_fund_flow_fallback.FundFlowFallbackTests.test_stockfetch_deduplicates_sector_fallback_rows_by_name -v
```

Expected: FAIL because the current fetch function returns both rows.

- [ ] **Step 3: Implement minimal sector normalization**

Add immediately before `fetch_stocks_sector_fund_flow`:

```python
def _deduplicate_sector_fund_flow_rows(data, index_sector, index_indicator):
    if data is None:
        return None
    if "name" not in data.columns:
        raise ValueError("sector fund-flow data is missing the name column")

    duplicate_rows = data.duplicated(subset="name", keep=False)
    if duplicate_rows.any():
        before = len(data.index)
        data = data.drop_duplicates(subset="name", keep="last")
        logging.warning(
            "sector fund-flow data contains duplicate names: sector=%s period=%s removed=%s",
            index_sector,
            index_indicator,
            before - len(data.index),
        )
    return data.reset_index(drop=True)
```

Replace both `return data` statements in `fetch_stocks_sector_fund_flow` with:

```python
return _deduplicate_sector_fund_flow_rows(data, index_sector, index_indicator)
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command again. Expected: one test passes.

### Task 2: Post-Merge Pre-Write Invariant

**Files:**
- Modify: `tests/test_fund_flow_fallback.py`
- Modify: `instock/job/basic_data_other_daily_job.py:135-168`

- [ ] **Step 1: Write the failing database-protection test**

Add a test that bypasses source normalization, returns two current-period rows with the same `name`, and asserts no database function is called:

```python
def test_sector_job_does_not_delete_or_insert_when_merged_names_are_not_unique(self):
    duplicate = pd.DataFrame([
        {"name": "通信服务", "change_rate": 1.10},
        {"name": "通信服务", "change_rate": 1.20},
    ])

    with patch.object(
        fund_flow_job,
        "run_check_stock_sector_fund_flow",
        return_value={0: duplicate},
    ), patch.object(fund_flow_job.mdb, "checkTableIsExist") as check_table, \
         patch.object(fund_flow_job.mdb, "executeSql") as execute_sql, \
         patch.object(fund_flow_job.mdb, "insert_db_from_df") as insert_rows:
        fund_flow_job.stock_sector_fund_flow_data(datetime(2026, 7, 3), 0)

    check_table.assert_not_called()
    execute_sql.assert_not_called()
    insert_rows.assert_not_called()
```

- [ ] **Step 2: Run the guard test and verify RED**

Run:

```powershell
& 'C:\InStockLocal\venv\Scripts\python.exe' -m unittest tests.test_fund_flow_fallback.FundFlowFallbackTests.test_sector_job_does_not_delete_or_insert_when_merged_names_are_not_unique -v
```

Expected: FAIL because the current job calls the database.

- [ ] **Step 3: Implement the pre-write invariant**

Insert after the empty-data check and before `data.insert(...)`:

```python
duplicate_names = data.loc[
    data.duplicated(subset="name", keep=False), "name"
].drop_duplicates().tolist()
if duplicate_names:
    logging.error(
        "DUPLICATE_SECTOR_FUND_FLOW_NAMES merged sector fund-flow data is not unique: sector=%s count=%s sample=%s",
        index_sector,
        len(duplicate_names),
        duplicate_names[:10],
    )
    return
```

- [ ] **Step 4: Run both new tests and verify GREEN**

Run both named tests. Expected: two tests pass and database mocks remain untouched in the guard test.

### Task 3: Regression and Live Recovery

**Files:**
- Verify: `tests/test_fund_flow_fallback.py`
- Verify: `instock/core/stockfetch.py`
- Verify: `instock/job/basic_data_other_daily_job.py`

- [ ] **Step 1: Run the complete test suite**

Run:

```powershell
$env:PYTHONPATH='C:\InStockLocal\InStock'
& 'C:\InStockLocal\venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Run the affected daily job in the production environment**

Set the same `db_host`, `db_user`, `db_password`, `db_database`, and `db_port` variables used by the desktop launcher, then run:

```powershell
& 'C:\InStockLocal\venv\Scripts\python.exe' 'C:\InStockLocal\InStock\instock\job\basic_data_other_daily_job.py'
```

Expected: process exits successfully with no `IntegrityError 1062` or `DUPLICATE_SECTOR_FUND_FLOW_NAMES` error.

- [ ] **Step 3: Verify the repaired database partition**

Query `cn_stock_fund_flow_industry` for `2026-07-03` and compare `COUNT(*)` with `COUNT(DISTINCT name)`.

Expected: positive count and equal values. Also verify the concept table remains positive and unique.

- [ ] **Step 4: Review the diff and preserve unrelated user changes**

Since this directory is not a Git repository, inspect only the three intended files and do not create commits. Confirm the implementation matches the approved design and no launcher, schema, or generic database files changed.
