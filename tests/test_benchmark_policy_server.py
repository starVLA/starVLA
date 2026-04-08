import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from deployment.model_server.tools.benchmark_policy_server import (
    build_comparison_summary,
    build_request_payload,
    parse_state,
    run_scenario,
    summarize_measurements,
)


class _FakeBenchmarkClient:
    def __init__(self) -> None:
        self.reset_calls = []
        self.infer_calls = []
        self._cache_primed = False

    def reset_cache(self, request_id: str) -> dict:
        self.reset_calls.append(request_id)
        self._cache_primed = False
        return {"status": "ok"}

    def infer(self, payload: dict) -> dict:
        self.infer_calls.append(payload["request_id"])
        cache_hit = self._cache_primed
        self._cache_primed = True
        return {
            "status": "ok",
            "data": {
                "cache_info": {
                    "hit": cache_hit,
                    "cache_bytes": 1024,
                    "cache_entries": 1,
                }
            },
            "metrics": {
                "predict_action_ms": 10.0 if cache_hit else 20.0,
            },
        }


class BenchmarkPolicyServerTests(unittest.TestCase):
    def test_parse_state_accepts_json_or_csv(self):
        self.assertEqual([0.0, 1.5, -2.0], parse_state("[0, 1.5, -2]"))
        self.assertEqual([0.0, 1.5, -2.0], parse_state("0, 1.5, -2"))
        self.assertIsNone(parse_state(""))

    def test_build_request_payload_loads_images_and_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "sample.png"
            image = Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8), mode="RGB")
            image.save(image_path)

            payload = build_request_payload(
                image_paths=[str(image_path)],
                instruction="stack blocks",
                state=[0.1, 0.2],
                use_vlm_cache=True,
            )

        self.assertTrue(payload["use_vlm_cache"])
        self.assertTrue(payload["return_cache_info"])
        self.assertEqual("stack blocks", payload["examples"][0]["lang"])
        self.assertEqual([0.1, 0.2], payload["examples"][0]["state"])
        self.assertEqual((4, 5, 3), payload["examples"][0]["image"][0].shape)

    def test_summarize_measurements_reports_cache_and_latency_stats(self):
        summary = summarize_measurements(
            "reuse_same_session",
            [
                {"request_latency_ms": 100.0, "predict_action_ms": 80.0, "cache_hit": False, "cache_bytes": 4096, "cache_entries": 1},
                {"request_latency_ms": 60.0, "predict_action_ms": 40.0, "cache_hit": True, "cache_bytes": 4096, "cache_entries": 1},
                {"request_latency_ms": 55.0, "predict_action_ms": 35.0, "cache_hit": True, "cache_bytes": 4096, "cache_entries": 1},
            ],
        )

        self.assertEqual("reuse_same_session", summary["scenario"])
        self.assertEqual(3, summary["requests"])
        self.assertEqual(2, summary["cache_hits"])
        self.assertEqual(1, summary["cache_misses"])
        self.assertAlmostEqual(2.0 / 3.0, summary["cache_hit_rate"])
        self.assertEqual({"last": 4096, "max": 4096}, summary["cache_bytes"])
        self.assertEqual({"last": 1, "max": 1}, summary["cache_entries"])
        self.assertAlmostEqual((100.0 + 60.0 + 55.0) / 3.0, summary["request_latency_ms"]["mean"])
        self.assertAlmostEqual((80.0 + 40.0 + 35.0) / 3.0, summary["predict_action_ms"]["mean"])

    def test_build_comparison_summary_reports_latency_and_throughput_deltas(self):
        baseline = summarize_measurements(
            "cold_reset_each_request",
            [
                {"request_latency_ms": 100.0, "predict_action_ms": 80.0, "cache_hit": False, "cache_bytes": 2048, "cache_entries": 1},
                {"request_latency_ms": 90.0, "predict_action_ms": 70.0, "cache_hit": False, "cache_bytes": 2048, "cache_entries": 1},
            ],
        )
        candidate = summarize_measurements(
            "reuse_same_session",
            [
                {"request_latency_ms": 60.0, "predict_action_ms": 45.0, "cache_hit": True, "cache_bytes": 2048, "cache_entries": 1},
                {"request_latency_ms": 50.0, "predict_action_ms": 35.0, "cache_hit": True, "cache_bytes": 2048, "cache_entries": 1},
            ],
        )

        comparison = build_comparison_summary(baseline, candidate)

        self.assertEqual("cold_reset_each_request", comparison["baseline_scenario"])
        self.assertEqual("reuse_same_session", comparison["candidate_scenario"])
        self.assertAlmostEqual(95.0, comparison["request_latency_ms"]["baseline_mean"])
        self.assertAlmostEqual(55.0, comparison["request_latency_ms"]["candidate_mean"])
        self.assertGreater(comparison["request_latency_ms"]["mean_delta_pct"], 0.0)
        self.assertGreater(comparison["throughput_rps"]["delta_pct"], 0.0)
        self.assertEqual(0.0, comparison["cache_hit_rate"]["baseline"])
        self.assertEqual(1.0, comparison["cache_hit_rate"]["candidate"])

    def test_run_scenario_resets_each_request_for_cold_mode(self):
        client = _FakeBenchmarkClient()

        result = run_scenario(
            client=client,
            scenario_name="cold_reset_each_request",
            request_payload={"examples": [{"lang": "pick", "image": []}]},
            warmup=0,
            runs=3,
            reset_between_requests=True,
            sleep_ms=0,
        )

        self.assertEqual(
            [
                "cold_reset_each_request-start",
                "cold_reset_each_request-reset-0",
                "cold_reset_each_request-reset-1",
                "cold_reset_each_request-reset-2",
            ],
            client.reset_calls,
        )
        self.assertEqual([False, False, False], [entry["cache_hit"] for entry in result["measurements"]])
        self.assertEqual(0, result["summary"]["cache_hits"])
        self.assertEqual(3, result["summary"]["cache_misses"])

    def test_run_scenario_reuses_same_session_without_intermediate_resets(self):
        client = _FakeBenchmarkClient()

        result = run_scenario(
            client=client,
            scenario_name="reuse_same_session",
            request_payload={"examples": [{"lang": "pick", "image": []}]},
            warmup=0,
            runs=3,
            reset_between_requests=False,
            sleep_ms=0,
        )

        self.assertEqual(["reuse_same_session-start"], client.reset_calls)
        self.assertEqual([False, True, True], [entry["cache_hit"] for entry in result["measurements"]])
        self.assertEqual({"last": 1024, "max": 1024}, result["summary"]["cache_bytes"])
        self.assertEqual({"last": 1, "max": 1}, result["summary"]["cache_entries"])


if __name__ == "__main__":
    unittest.main()
