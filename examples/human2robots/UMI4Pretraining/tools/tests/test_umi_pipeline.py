import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "umi_pipeline.py"
SPEC = importlib.util.spec_from_file_location("umi_pipeline", MODULE_PATH)
assert SPEC and SPEC.loader
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class PipelineTest(unittest.TestCase):
    def test_locked_plan_has_27_families(self):
        plan = PIPELINE.load_plan()
        families = {PIPELINE.family_name(name, item) for name, item in plan.items()}
        self.assertEqual(len(plan), 30)
        self.assertEqual(len(families), 27)
        self.assertIn("VISTA-UMI-5K", families)
        self.assertIn("MV-UMI", families)
        self.assertIn("UMI-3D", families)

    def test_family_selection_expands_multi_repo_family(self):
        selected = PIPELINE.selected_plan(PIPELINE.load_plan(), ["UMI-3D"])
        self.assertEqual(set(selected), {"UMI-3D-cup", "UMI-3D-curtain", "UMI-3D-door-cup"})

    def test_direct_verification_checks_exact_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = {
                "source_type": "direct",
                "filename": "sample.zip",
                "expected_bytes": 4,
            }
            archive_dir = root / "_archives"
            archive_dir.mkdir(parents=True)
            (archive_dir / "sample.zip").write_bytes(b"1234")
            result = PIPELINE.verify_one("sample", item, root, deep=False)
            self.assertEqual(result["status"], "ok")

    def test_hf_verification_understands_recursive_pattern(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = {
                "source_type": "huggingface",
                "repo": "owner/repo",
                "files": ["README.md", "task_001/**"],
            }
            base = PIPELINE.source_dir(root, "sample", item)
            (base / "task_001" / "data").mkdir(parents=True)
            (base / "README.md").write_text("ok")
            (base / "task_001" / "data" / "file.parquet").write_bytes(b"x")
            result = PIPELINE.verify_one("sample", item, root, deep=False)
            self.assertEqual(result["status"], "ok")

    def test_hf_cli_fallback_builds_include_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = type("Args", (), {"dry_run": False, "min_free_gib": 0, "hf_workers": 1})()
            item = {
                "source_type": "huggingface",
                "repo": "owner/repo",
                "files": ["README.md", "task/**"],
            }
            original_import = __import__

            def missing_hub(name, *values, **kwargs):
                if name == "huggingface_hub":
                    raise ImportError
                return original_import(name, *values, **kwargs)

            import builtins
            from unittest.mock import patch

            with patch.object(builtins, "__import__", side_effect=missing_hub), \
                 patch.object(PIPELINE.shutil, "which", return_value="/usr/bin/hf"), \
                 patch.object(PIPELINE.subprocess, "run") as run, \
                 patch.object(PIPELINE, "matches_exist", return_value=(2, [])):
                PIPELINE.hf_download("sample", item, Path(temporary), args)
            command = run.call_args.args[0]
            self.assertIn("--include", command)
            self.assertIn("task/**", command)
            run.assert_called_once_with(command, check=True)

    def test_subset_verify_does_not_write_global_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = {"source_type": "direct", "filename": "sample.zip", "expected_bytes": 4}
            (root / "_archives").mkdir()
            (root / "_archives" / "sample.zip").write_bytes(b"1234")
            args = type("Args", (), {"deep": False})()
            self.assertEqual(PIPELINE.command_verify({"sample": item}, root, args, full_plan=False), 0)
            self.assertFalse((root / ".all_available_400_sources_downloaded").exists())

    def test_full_verify_writes_global_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = {"source_type": "direct", "filename": "sample.zip", "expected_bytes": 4}
            (root / "_archives").mkdir()
            (root / "_archives" / "sample.zip").write_bytes(b"1234")
            args = type("Args", (), {"deep": False})()
            self.assertEqual(PIPELINE.command_verify({"sample": item}, root, args, full_plan=True), 0)
            self.assertTrue((root / ".all_available_400_sources_downloaded").exists())

    def test_completed_partial_is_recovered_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            item = {
                "source_type": "direct",
                "filename": "sample.zip",
                "expected_bytes": 4,
                "url": "https://example.invalid/sample.zip",
            }
            archive = root / "_archives"
            archive.mkdir()
            (archive / "sample.zip.part").write_bytes(b"1234")
            args = type("Args", (), {"dry_run": False, "min_free_gib": 0, "retries": 1, "timeout": 1})()
            result = PIPELINE.direct_download("sample", item, root, args)
            self.assertEqual(result["status"], "recovered")
            self.assertTrue((archive / "sample.zip").exists())


if __name__ == "__main__":
    unittest.main()
