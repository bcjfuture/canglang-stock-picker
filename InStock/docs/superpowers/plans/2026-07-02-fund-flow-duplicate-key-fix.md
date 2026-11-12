# Fund Flow Duplicate-Key Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent duplicate stock codes from breaking the daily fund-flow insert and protect the existing date partition when merged input violates uniqueness.

**Architecture:** Normalize each individual fund-flow response at the `stockfetch` source boundary so both Eastmoney and Tonghuashun produce one row per code. Add a second invariant check in the daily job before any database deletion, then rerun the live job to replace the incomplete 2026-07-02 partition.

**Tech Stack:** Python 3, pandas, unittest, unittest.mock, SQLAlchemy, MariaDB

---

## File Structure

- Modify `C:\InStockLocal\InStock\instock\core\stockfetch.py`: central source-boundary deduplication for individual fund-flow data.
- Modify `C:\InStockLocal\InStock\instock\job\basic_data_other_daily_job.py`: merged-data uniqueness guard before database mutation.
- Modify `C:\InStockLocal\InStock\tests\test_fund_flow_fallback.py`: regression tests for source deduplication and pre-write protection.

### Task 1: Deduplicate individual fund-flow results at the source boundary

**Files:**
- Modify: `C:\InStockLocal\InStock\tests\test_fund_flow_fallback.py`
- Modify: `C:\InStockLocal\InStock\instock\core\stockfetch.py`

- [ ] **Step 1: Write the failing fallback-duplicate test**

Add this import to `C:\InStockLocal\InStock\tests\test_fund_flow_fallback.py`:

```python
import instock.core.tablestructure as tablestructure
```

Add this method to `FundFlowFallbackTests`:

```python
def test_stockfetch_deduplicates_fallback_rows_by_code(self):
    columns = list(tablestructure.CN_STOCK_FUND_FLOW[0]["columns"])
    first = {column: None for column in columns}
    first.update({"code": "600141", "name": "兴发集团", "new_price": 21.10})
    second = first.copy()
    second["new_price"] = 21.25
    fallback = pd.DataFrame([first, second], columns=columns)

    with patch.object(
        stockfetch.sff,
        "stock_individual_fund_flow_rank",
        side_effect=ConnectionError("primary unavailable"),
    ), patch(
        "instock.core.fund_flow_fallback.fetch_individual_rank",
        return_value=fallback,
    ):
        result = stockfetch.fetch_stocks_fund_flow(0)

    self.assertEqual(len(result.index), 1)
    self.assertEqual(result.iloc[0]["code"], "600141")
    self.assertEqual(result.iloc[0]["new_price"], 21.25)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
$env:PYTHONPATH='C:\InStockLocal\InStock'
& 'C:\InStockLocal\venv\Scripts\python.exe' -m unittest tests.test_fund_flow_fallback.FundFlowFallbackTests.test_stockfetch_deduplicates_fallback_rows_by_code -v
```

Expected: FAIL because `fetch_stocks_fund_flow(0)` returns two rows instead of one.

- [ ] **Step 3: Implement the minimal normalization helper**

Add this helper immediately before `fetch_stocks_fund_flow` in `C:\InStockLocal\InStock\instock\core\stockfetch.py`:

```python
def _deduplicate_fund_flow_rows(data, index):
    if data is None:
        return None
    if "code" not in data.columns:
        raise ValueError("individual fund-flow data is missing the code column")

    duplicate_rows = data.duplicated(subset="code", keep=False)
    if duplicate_rows.any():
        before = len(data.index)
        data = data.drop_duplicates(subset="code", keep="last")
        logging.warning(
            "个股资金流数据包含重复代码：周期=%s，已移除=%s 条",
            index,
            before - len(data.index),
        )
    return data.reset_index(drop=True)
```

Change both successful return paths in `fetch_stocks_fund_flow` from `return data` to:

```python
return _deduplicate_fund_flow_rows(data, index)
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2 again.

Expected: PASS, one row remains, and the last `new_price` value is retained.

- [ ] **Step 5: Record the checkpoint**

This deployment directory is not a Git repository, so no commit is possible. Confirm that explicitly:

```powershell
Test-Path 'C:\InStockLocal\InStock\.git'
```

Expected: `False`. Preserve the passing test output as the checkpoint evidence.

### Task 2: Block invalid merged data before deleting the current partition

**Files:**
- Modify: `C:\InStockLocal\InStock\tests\test_fund_flow_fallback.py`
- Modify: `C:\InStockLocal\InStock\instock\job\basic_data_other_daily_job.py`

- [ ] **Step 1: Write the failing pre-write-guard test**

Add these imports to `C:\InStockLocal\InStock\tests\test_fund_flow_fallback.py`:

```python
from datetime import datetime
import instock.job.basic_data_other_daily_job as fund_flow_job
```

Add this method to `FundFlowFallbackTests`:

```python
def test_job_does_not_delete_or_insert_when_merged_codes_are_not_unique(self):
    duplicate = pd.DataFrame([
        {"code": "600141", "name": "兴发集团", "new_price": 21.10},
        {"code": "600141", "name": "兴发集团", "new_price": 21.25},
    ])

    with patch.object(
        fund_flow_job,
        "run_check_stock_fund_flow",
        return_value={0: duplicate},
    ), patch.object(
        fund_flow_job.mdb,
        "checkTableIsExist",
    ) as check_table, patch.object(
        fund_flow_job.mdb,
        "executeSql",
    ) as execute_sql, patch.object(
        fund_flow_job.mdb,
        "insert_db_from_df",
    ) as insert_rows:
        fund_flow_job.save_nph_stock_fund_flow_data(
            datetime(2026, 7, 2),
            before=False,
        )

    check_table.assert_not_called()
    execute_sql.assert_not_called()
    insert_rows.assert_not_called()
```

- [ ] **Step 2: Run the guard test and verify RED**

Run:

```powershell
$env:PYTHONPATH='C:\InStockLocal\InStock'
& 'C:\InStockLocal\venv\Scripts\python.exe' -m unittest tests.test_fund_flow_fallback.FundFlowFallbackTests.test_job_does_not_delete_or_insert_when_merged_codes_are_not_unique -v
```

Expected: FAIL because the current job reaches `checkTableIsExist` and database mutation code.

- [ ] **Step 3: Implement the pre-write invariant**

In `save_nph_stock_fund_flow_data`, immediately after the empty-data check and before `data.insert(0, 'date', ...)`, add:

```python
        duplicate_codes = data.loc[
            data.duplicated(subset="code", keep=False), "code"
        ].drop_duplicates().tolist()
        if duplicate_codes:
            logging.error(
                "DUPLICATE_FUND_FLOW_CODES merged fund-flow data is not unique: count=%s sample=%s",
                len(duplicate_codes),
                duplicate_codes[:10],
            )
            return
```

- [ ] **Step 4: Run both focused tests and verify GREEN**

Run:

```powershell
$env:PYTHONPATH='C:\InStockLocal\InStock'
& 'C:\InStockLocal\venv\Scripts\python.exe' -m unittest `
  tests.test_fund_flow_fallback.FundFlowFallbackTests.test_stockfetch_deduplicates_fallback_rows_by_code `
  tests.test_fund_flow_fallback.FundFlowFallbackTests.test_job_does_not_delete_or_insert_when_merged_codes_are_not_unique `
  -v
```

Expected: both tests PASS.

- [ ] **Step 5: Record the checkpoint**

No Git commit is available in this deployment directory. Save the two-test PASS output with the task notes.

### Task 3: Regression and live-data verification

**Files:**
- Verify: `C:\InStockLocal\InStock\tests\test_fund_flow_fallback.py`
- Verify: `C:\InStockLocal\logs\update-*.log`
- Verify: MariaDB table `instockdb.cn_stock_fund_flow`

- [ ] **Step 1: Run the complete test suite**

Run:

```powershell
$env:PYTHONPATH='C:\InStockLocal\InStock'
Set-Location 'C:\InStockLocal\InStock'
& 'C:\InStockLocal\venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Expected: all tests PASS with no traceback.

- [ ] **Step 2: Run the affected live job with the desktop environment**

Run from `C:\InStockLocal\InStock\instock\job` with the same database variables as the shortcut:

```powershell
$env:PYTHONPATH='C:\InStockLocal\InStock'
$env:PYTHONUNBUFFERED='1'
$env:db_host='localhost'
$env:db_user='root'
$env:db_password='root'
$env:db_database='instockdb'
$env:db_port='3306'
Set-Location 'C:\InStockLocal\InStock\instock\job'
& 'C:\InStockLocal\venv\Scripts\python.exe' 'basic_data_other_daily_job.py'
```

Expected: the individual fund-flow step logs duplicate-removal warnings but no `IntegrityError` or `DUPLICATE_FUND_FLOW_CODES` error.

- [ ] **Step 3: Verify the repaired date partition**

Run:

```powershell
$env:PYTHONPATH='C:\InStockLocal\InStock'
$env:db_host='localhost'
$env:db_user='root'
$env:db_password='root'
$env:db_database='instockdb'
$env:db_port='3306'
@'
from instock.lib.database import executeSqlFetch
print(executeSqlFetch(
    "SELECT COUNT(*), COUNT(DISTINCT code) "
    "FROM cn_stock_fund_flow WHERE date='2026-07-02'"
))
'@ | & 'C:\InStockLocal\venv\Scripts\python.exe' -
```

Expected: both counts are equal and greater than 4,500.

- [ ] **Step 4: Run the desktop update workflow**

Run the exact shortcut target non-interactively:

```powershell
cmd.exe /c "echo.| powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"C:\Users\包程杰\Documents\Codex\2026-06-25\wo\work\update-instock-data.ps1\""
```

Expected: all configured daily jobs complete, and the newest `C:\InStockLocal\logs\update-*.log` contains no critical database error.

- [ ] **Step 5: Verify the shortcut remains valid**

Run:

```powershell
$shortcut=(New-Object -ComObject WScript.Shell).CreateShortcut('C:\Users\包程杰\Desktop\更新 InStock 数据.lnk')
$shortcut.TargetPath
$shortcut.Arguments
```

Expected: target is PowerShell and arguments still reference `update-instock-data.ps1`.
