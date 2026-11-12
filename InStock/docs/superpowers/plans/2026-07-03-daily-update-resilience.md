# Daily Update Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every desktop-launched daily partition replacement atomic, retry transient job failures, and report success only after core database health checks pass.

**Architecture:** A generic database helper validates incoming keys and performs delete, append, and count verification on one SQLAlchemy transaction connection. The five desktop-launched jobs use this helper; the PowerShell launcher retries failed jobs up to three times and runs a final Python health checker.

**Tech Stack:** Python 3.11, pandas 2.2, SQLAlchemy 2.0, PyMySQL, MariaDB/InnoDB, unittest, PowerShell

---

## File Map

- Modify `instock/lib/database.py`: validation, atomic replacement, and exception propagation.
- Create `tests/test_database_atomic_replace.py`: unit coverage for validation, transaction connection reuse, and propagation.
- Modify five `instock/job/*daily_job.py` files: replace split delete/insert sequences.
- Create `tests/test_daily_jobs_use_atomic_replace.py`: prevent regression to split writes.
- Create `instock/job/verify_daily_update.py`: final core-table verification.
- Create `tests/test_verify_daily_update.py`: health-check behavior.
- Modify `C:\Users\包程杰\Documents\Codex\2026-06-25\wo\work\update-instock-data.ps1`: bounded retry and verifier invocation.

### Task 1: Atomic Database Primitive

**Files:**
- Create: `tests/test_database_atomic_replace.py`
- Modify: `instock/lib/database.py`

- [ ] **Step 1: Write validation and transaction tests**

Create tests that patch `database.engine`, use a mocked `engine.begin()` connection, and assert:

```python
class AtomicReplaceTests(unittest.TestCase):
    def test_rejects_duplicate_keys_before_opening_database(self):
        data = pd.DataFrame([
            {"date": "2026-07-03", "code": "000001"},
            {"date": "2026-07-03", "code": "000001"},
        ])
        with patch.object(database, "engine") as engine:
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                database.replace_partition_from_df(
                    data, "sample", "date", "2026-07-03",
                    ["date", "code"], None, primary_keys="`date`,`code`",
                )
        engine.assert_not_called()

    def test_uses_one_transaction_for_delete_append_and_count(self):
        data = pd.DataFrame([{"date": "2026-07-03", "code": "000001"}])
        db_engine = MagicMock()
        connection = db_engine.begin.return_value.__enter__.return_value
        connection.execute.return_value.scalar_one.return_value = 1
        with patch.object(database, "checkTableIsExist", return_value=True), \
             patch.object(database, "engine", return_value=db_engine), \
             patch.object(pd.DataFrame, "to_sql") as to_sql:
            count = database.replace_partition_from_df(
                data, "sample", "date", "2026-07-03",
                ["date", "code"], None, primary_keys="`date`,`code`",
            )
        self.assertEqual(count, 1)
        self.assertEqual(to_sql.call_args.kwargs["con"], connection)
        self.assertEqual(connection.execute.call_count, 2)

    def test_insert_failure_is_reraised(self):
        data = pd.DataFrame([{"date": "2026-07-03", "code": "000001"}])
        db_engine = MagicMock()
        with patch.object(database, "checkTableIsExist", return_value=True), \
             patch.object(database, "engine", return_value=db_engine), \
             patch.object(pd.DataFrame, "to_sql", side_effect=RuntimeError("insert failed")):
            with self.assertRaisesRegex(RuntimeError, "insert failed"):
                database.replace_partition_from_df(
                    data, "sample", "date", "2026-07-03",
                    ["date", "code"], None, primary_keys="`date`,`code`",
                )
```

Also cover empty data, missing columns, null keys, and mixed partition values with a parameterized `subTest` loop.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='C:\InStockLocal\InStock'
& 'C:\InStockLocal\venv\Scripts\python.exe' -m unittest tests.test_database_atomic_replace -v
```

Expected: import/attribute failures because `replace_partition_from_df` does not exist.

- [ ] **Step 3: Implement validation and atomic replacement**

In `database.py`, import `re` and `text` from SQLAlchemy. Add identifier validation and a partition token helper, then implement:

```python
def replace_partition_from_df(data, table_name, partition_column, partition_value,
                              key_columns, cols_type, write_index=False,
                              primary_keys=None, indexs=None):
    if data is None or data.empty:
        raise ValueError("partition data must not be empty")
    for identifier in (table_name, partition_column, *key_columns):
        if not re.fullmatch(r"[A-Za-z0-9_]+", identifier):
            raise ValueError(f"unsafe SQL identifier: {identifier}")
    required = [partition_column, *key_columns]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"partition data is missing columns: {missing}")
    if data[list(dict.fromkeys(required))].isnull().any().any():
        raise ValueError("partition/key columns contain null values")
    if data.duplicated(subset=key_columns, keep=False).any():
        raise ValueError(f"partition data contains duplicate key rows: {key_columns}")

    token = partition_value.strftime("%Y-%m-%d") if hasattr(partition_value, "strftime") else str(partition_value)
    actual = data[partition_column].map(
        lambda value: value.strftime("%Y-%m-%d") if hasattr(value, "strftime") else str(value)[:10]
    )
    if not actual.eq(token).all():
        raise ValueError(f"partition data contains values outside {token}")

    if not checkTableIsExist(table_name):
        inserted = insert_db_from_df(
            data, table_name, cols_type, write_index, primary_keys, indexs
        )
        if inserted != len(data.index):
            raise RuntimeError(
                f"partition row count mismatch: expected={len(data.index)} actual={inserted}"
            )
        return inserted

    db_engine = engine()
    try:
        with db_engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM `{table_name}` WHERE `{partition_column}` = :partition_value"),
                {"partition_value": token},
            )
            data.to_sql(table_name, con=conn, if_exists="append", index=write_index)
            inserted = conn.execute(
                text(f"SELECT COUNT(*) FROM `{table_name}` WHERE `{partition_column}` = :partition_value"),
                {"partition_value": token},
            ).scalar_one()
            if inserted != len(data.index):
                raise RuntimeError(
                    f"partition row count mismatch: expected={len(data.index)} actual={inserted}"
                )
        return inserted
    except Exception:
        logging.exception("DAILY_UPDATE_WRITE_FAILED table=%s", table_name)
        raise
    finally:
        db_engine.dispose()
```

Change `insert_db_from_df` to return `insert_other_db_from_df(...)`, make `insert_other_db_from_df` return the inserted count, and add `raise` after both logged exception handlers so failures propagate.

- [ ] **Step 4: Run atomic-helper tests and verify GREEN**

Run the Step 2 command. Expected: all new tests pass.

### Task 2: Migrate Desktop-Launched Daily Jobs

**Files:**
- Create: `tests/test_daily_jobs_use_atomic_replace.py`
- Modify: `instock/job/basic_data_daily_job.py`
- Modify: `instock/job/basic_data_other_daily_job.py`
- Modify: `instock/job/selection_data_daily_job.py`
- Modify: `instock/job/indicators_data_daily_job.py`
- Modify: `instock/job/strategy_data_daily_job.py`

- [ ] **Step 1: Write a failing architecture regression test**

```python
class DailyJobAtomicWriteTests(unittest.TestCase):
    JOB_FILES = (
        "basic_data_daily_job.py", "basic_data_other_daily_job.py",
        "selection_data_daily_job.py", "indicators_data_daily_job.py",
        "strategy_data_daily_job.py",
    )

    def test_desktop_jobs_do_not_split_delete_and_insert(self):
        job_dir = Path(__file__).parents[1] / "instock" / "job"
        for filename in self.JOB_FILES:
            source = (job_dir / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertNotIn("mdb.executeSql(del_sql)", source)
                self.assertNotIn("mdb.insert_db_from_df(", source)
                self.assertIn("mdb.replace_partition_from_df(", source)
```

- [ ] **Step 2: Run the architecture test and verify RED**

Run `python -m unittest tests.test_daily_jobs_use_atomic_replace -v` with the project virtual environment. Expected: all five subtests identify legacy split writes.

- [ ] **Step 3: Replace each split write with the atomic helper**

For every existing date-partition sequence, remove `checkTableIsExist`, `DELETE`, and `insert_db_from_df`. Call:

```python
mdb.replace_partition_from_df(
    data=data,
    table_name=table_name,
    partition_column="date",
    partition_value=date_value,
    key_columns=["date", "code"],
    cols_type=tbs.get_field_types(table_definition["columns"]),
    primary_keys="`date`,`code`",
)
```

Use `["date", "name"]` and primary key `` `date`,`name` `` for industry/concept sector tables. Use each DataFrame's actual date value for selection data and the function `date` for other jobs. Preserve all calculations and existing source normalization.

In each surrounding `except`, change the error prefix to `DAILY_UPDATE_WRITE_FAILED` while retaining the function/table context and exception.

- [ ] **Step 4: Run migration and existing tests**

Run:

```powershell
& 'C:\InStockLocal\venv\Scripts\python.exe' -m unittest tests.test_daily_jobs_use_atomic_replace tests.test_fund_flow_fallback tests.test_fallback_quote -v
```

Expected: architecture tests and existing functional tests pass.

### Task 3: Final Database Health Checker

**Files:**
- Create: `instock/job/verify_daily_update.py`
- Create: `tests/test_verify_daily_update.py`

- [ ] **Step 1: Write failing checker tests**

Test a `verify_table(cursor, table_name, date_value, key_column)` function with mocked cursor results:

```python
def test_verify_table_accepts_positive_unique_partition(self):
    cursor = MagicMock()
    cursor.fetchone.return_value = (90, 90, 0)
    result = verifier.verify_table(cursor, "industry", "2026-07-03", "name")
    self.assertEqual(result["rows"], 90)

def test_verify_table_rejects_empty_or_duplicate_partition(self):
    for row in ((0, 0, 0), (90, 89, 0), (90, 90, 1)):
        cursor = MagicMock()
        cursor.fetchone.return_value = row
        with self.subTest(row=row), self.assertRaises(RuntimeError):
            verifier.verify_table(cursor, "industry", "2026-07-03", "name")
```

- [ ] **Step 2: Run checker tests and verify RED**

Run `python -m unittest tests.test_verify_daily_update -v`. Expected: module import failure.

- [ ] **Step 3: Implement the checker**

Create a script with a `CORE_TABLES` tuple mapping stock/ETF/fund-flow tables to `code` and sector tables to `name`. `verify_table` executes one aggregate query returning total rows, distinct keys, and null keys; it raises unless total is positive, total equals distinct, and null count is zero. `main()` obtains the latest non-future trade date through `trd.get_trade_date_last()`, checks all tables using one DB connection, prints summaries, and returns 0. The `__main__` block catches failures, logs `DAILY_UPDATE_HEALTH_FAILED`, and exits 1.

- [ ] **Step 4: Run checker tests and verify GREEN**

Run the Step 2 command. Expected: checker tests pass.

### Task 4: Bounded Launcher Retry

**Files:**
- Modify: `C:\Users\包程杰\Documents\Codex\2026-06-25\wo\work\update-instock-data.ps1`

- [ ] **Step 1: Parse the existing script before editing**

Run PowerShell's parser and confirm zero syntax errors. Save this as the baseline command for the post-edit check.

- [ ] **Step 2: Implement three-attempt job execution**

Change the signature to:

```powershell
function Invoke-InStockJob([string]$Label, [string]$ScriptName, [int]$MaxAttempts = 3) {
```

Wrap process execution and log scanning in `for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++)`. Create fresh temporary files each attempt, add `=== ... attempt N/M ===` to the log, and treat nonzero exit, critical `ERROR:` lines, `DAILY_UPDATE_WRITE_FAILED`, `DAILY_UPDATE_HEALTH_FAILED`, and duplicate invariant markers as attempt failure. Wait 3 seconds before attempt 2 and 10 seconds before attempt 3. Return immediately after a successful attempt; after the last failed attempt, throw one error containing the job and final failure preview. Always remove attempt temporary files in `finally`.

- [ ] **Step 3: Invoke final verification once**

After the strategy job and before printing `Done`, add:

```powershell
Invoke-InStockJob "Verifying daily update..." "verify_daily_update.py" 1
```

- [ ] **Step 4: Verify PowerShell syntax**

Run:

```powershell
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    'C:\Users\包程杰\Documents\Codex\2026-06-25\wo\work\update-instock-data.ps1',
    [ref]$null,
    [ref]$errors
) | Out-Null
if ($errors.Count) { $errors | Format-List; exit 1 }
```

Expected: exit 0 and no parser errors.

### Task 5: Fault Injection and End-to-End Verification

**Files:**
- Verify all files above.

- [ ] **Step 1: Run complete Python tests and syntax compilation**

Run `python -m py_compile` on all modified/created Python files, then `python -m unittest discover -s tests -v`. Expected: zero failures.

- [ ] **Step 2: Prove rollback with a scratch InnoDB table**

Create a uniquely named scratch table, insert one old row, call `replace_partition_from_df` while patching `DataFrame.to_sql` to raise after deletion, and query from a separate connection. Expected: the old row remains. Then perform a valid replacement, verify its count and uniqueness, and drop the scratch table in `finally`.

- [ ] **Step 3: Run the final health checker against live data**

Use the same database environment variables as the desktop launcher and run `verify_daily_update.py`. Expected: all five core tables print positive unique counts and exit 0.

- [ ] **Step 4: Run the desktop updater non-interactively**

Pipe one newline to the existing PowerShell launcher so its final `Read-Host` does not block. Allow the complete workflow to finish. Expected: every job succeeds, health verification succeeds, and the launcher prints `Done` with a new log path.

- [ ] **Step 5: Inspect the new update log**

Assert the newest log contains the verifier section and contains none of: `IntegrityError`, `ALL_SOURCES_FAILED`, `DAILY_UPDATE_WRITE_FAILED`, `DAILY_UPDATE_HEALTH_FAILED`, `DUPLICATE_`, or `Update failed:`.

- [ ] **Step 6: Record non-Git completion state**

This directory is not a Git repository. Do not create commits; list the intended changed files and preserve all unrelated local files.
