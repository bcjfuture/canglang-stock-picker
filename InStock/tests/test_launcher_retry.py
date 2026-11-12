import unittest
from pathlib import Path


class LauncherRetryTests(unittest.TestCase):
    SCRIPT = Path(
        r"C:\Users\包程杰\Documents\Codex\2026-06-25\wo\work\update-instock-data.ps1"
    )

    def test_launcher_has_bounded_retry_and_final_health_check(self):
        source = self.SCRIPT.read_text(encoding="utf-8-sig")

        self.assertIn("[int]$MaxAttempts = 3", source)
        self.assertIn("$attempt -le $MaxAttempts", source)
        self.assertIn("@(0, 3, 10)", source)
        self.assertIn('"verify_daily_update.py" 1', source)
        self.assertIn("attempt $attempt/$MaxAttempts", source)
        self.assertNotIn("Tee-Object -FilePath", source)
        self.assertIn("Add-Content -LiteralPath $UpdateLog -Encoding UTF8", source)

    def test_launcher_times_out_and_kills_hung_job_tree(self):
        source = self.SCRIPT.read_text(encoding="utf-8-sig")

        self.assertIn("[int]$TimeoutSeconds = 1800", source)
        self.assertIn("$process.WaitForExit($TimeoutSeconds * 1000)", source)
        self.assertIn("$process.WaitForExit()", source)
        self.assertIn("$process.Refresh()", source)
        self.assertIn("System.Diagnostics.ProcessStartInfo", source)
        self.assertIn("ReadToEndAsync()", source)
        self.assertIn("taskkill.exe /PID $process.Id /T /F", source)
        self.assertIn("timed out after $TimeoutSeconds seconds", source)

    def test_launcher_refuses_overlapping_updates(self):
        source = self.SCRIPT.read_text(encoding="utf-8-sig")

        self.assertIn("[System.IO.FileShare]::None", source)
        self.assertIn("Another InStock data update is already running", source)
        self.assertIn("$UpdateLock.Dispose()", source)


if __name__ == "__main__":
    unittest.main()
