import csv
import hashlib
import json
import stat
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
ARCHIVE_ROOT = WORKSPACE / "d2pred_rubbish/ablation_contamination_cleanup_20261103/evidence/d2pred_final"
ARCHIVE_MANIFEST = WORKSPACE / "d2pred_rubbish/ablation_contamination_cleanup_20261103/archive_manifest.csv"


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class AblationContaminationTests(unittest.TestCase):
    def test_legacy_protocol_and_approval_are_disabled(self):
        protocol = json.loads((ROOT / "manifests/controlled_ablation_protocol_expert_exact_v3.json").read_text())
        approval = json.loads((ROOT / "manifests/controlled_ablation_execution_approval_v3.json").read_text())
        registry = json.loads((ROOT / "manifests/active_experiment_registry.json").read_text())
        self.assertEqual(protocol["status"], "INVALID_LEGACY_ABLATION")
        self.assertFalse(protocol["execution_enabled"])
        self.assertEqual(approval["status"], "revoked_legacy_invalid")
        self.assertEqual(registry["ablation"]["status"], "INVALID_LEGACY_ABLATION")
        self.assertIsNone(registry["ablation"]["candidate"])
        self.assertEqual(registry["ablation"]["completed_datasets"], [])

    def test_no_active_legacy_result_tree(self):
        self.assertFalse((ROOT / "results/controlled_ablation_expert_exact_v3").exists())
        self.assertFalse((ROOT / "figures/controlled_ablation_v1").exists())
        with (ROOT / "checksum_manifest.csv").open("r", newline="") as stream:
            active_paths = [row["path"] for row in csv.DictReader(stream)]
        self.assertFalse(any(path.startswith("results/controlled_ablation_expert_exact_v3/") for path in active_paths))
        self.assertFalse(any(path.startswith("figures/controlled_ablation_v1/") for path in active_paths))

    def test_v3_figure_generators_are_retired(self):
        for name in (
            "generate_acall_ablation_extreme_cases.py",
            "generate_acall_ablation_rul_cases.py",
        ):
            path = ROOT / "figures" / name
            result = subprocess.run(
                [sys.executable, str(path)], cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
            )
            self.assertNotEqual(result.returncode, 0, name)
            self.assertIn("Refusing to generate", result.stderr + result.stdout, name)

    def test_old_preexecution_pass_is_non_authorizing(self):
        audit = json.loads((ROOT / "manifests/model_preexecution_audit.json").read_text())
        self.assertEqual(audit["status"], "INVALID_LEGACY_ABLATION")
        self.assertEqual(audit["historical_execution_status"], "PASS")
        self.assertFalse(audit["execution_authorization"])

    def test_supplemental_figure_archive_is_hash_verified_and_read_only(self):
        manifest = WORKSPACE / "d2pred_rubbish/ablation_contamination_cleanup_20261103/supplemental_quarantine_manifest.csv"
        with manifest.open("r", newline="") as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 9)
        for row in records:
            archived = Path(row["archive_path"])
            self.assertEqual(row["status"], "ARCHIVED_READ_ONLY")
            self.assertEqual(sha256(archived), row["archive_sha256"])
            self.assertFalse(stat.S_IMODE(archived.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    def test_full_comparison_frozen_files_unchanged(self):
        expected = {
            "ACSim/test_predictions.csv": "699dc759936ca42f796e7eb9e5bfa9ffe8bde5e290eefeb44d3d3298966bb4cd",
            "ACCom/test_predictions.csv": "f5015cd52edc5c25683bf34329509c22bc8ed7290647f1965a2c825b00028c62",
            "ACAll/test_predictions.csv": "f929a6718ea8ed8b1e5c783d47ac085bdff14853b37c403f7a8116242d8255fa",
            "ACSim/summary_metrics.csv": "02dac3d161e7dc4e8f338d63e3b1c5118b8513d63560844f22bba4bd0bcbcd1d",
            "ACCom/summary_metrics.csv": "90a310becdd00f1e3185427c9f7419bbe120eaaca04e4888db635d04ebf987ae",
            "ACAll/summary_metrics.csv": "b3de2eba685a392944472feae459a68ab3f77efb4217d6642f34fa224f9b0d29",
        }
        base = ROOT / "results/frozen_protocol_final/comparison"
        for relative, value in expected.items():
            self.assertEqual(sha256(base / relative), value, relative)

    def test_v3_evidence_archive_is_hash_verified_and_read_only(self):
        with ARCHIVE_MANIFEST.open("r", newline="") as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 1518)
        for row in records:
            archived = Path(row["archive_path"])
            self.assertEqual(row["status"], "MATCH")
            self.assertTrue(archived.is_file(), str(archived))
            self.assertEqual(sha256(archived), row["archive_sha256"], str(archived))
            self.assertFalse(stat.S_IMODE(archived.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    def test_historical_prediction_recomputation_is_complete(self):
        path = ROOT / "ablation_final/outputs/historical_metrics_recomputed.csv"
        with path.open("r", newline="") as stream:
            records = list(csv.DictReader(stream))
        self.assertEqual(len(records), 27)
        self.assertTrue(all(float(row["RMSE"]) > 0 for row in records))
        self.assertEqual({row["dataset"] for row in records}, {"ACSim", "ACCom", "ACAll"})

    def test_retired_runner_fails_before_training(self):
        executable = ROOT.parent / "PythonProject2/.venv-linux/bin/python"
        result = subprocess.run(
            [str(executable), str(ROOT / "scripts/run_controlled_ablation.py")],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to execute", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
