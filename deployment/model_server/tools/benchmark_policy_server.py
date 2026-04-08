import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def load_images(image_paths: List[str]) -> List[np.ndarray]:
    loaded_images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        loaded_images.append(np.asarray(image, dtype=np.uint8))
    return loaded_images


def parse_state(state_text: str | None) -> List[float] | None:
    if not state_text:
        return None

    state_text = state_text.strip()
    if not state_text:
        return None

    if state_text.startswith("["):
        parsed = json.loads(state_text)
    else:
        parsed = [float(token.strip()) for token in state_text.split(",") if token.strip()]
    return [float(value) for value in parsed]


def build_request_payload(
    *,
    image_paths: List[str],
    instruction: str,
    state: List[float] | None = None,
    use_vlm_cache: bool = True,
) -> Dict[str, Any]:
    example: Dict[str, Any] = {
        "image": load_images(image_paths),
        "lang": instruction,
    }
    if state is not None:
        example["state"] = state

    return {
        "examples": [example],
        "use_vlm_cache": use_vlm_cache,
        "return_cache_info": True,
    }


def summarize_measurements(name: str, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
    request_latencies = np.asarray([entry["request_latency_ms"] for entry in measurements], dtype=np.float64)
    server_latencies = np.asarray(
        [entry["predict_action_ms"] for entry in measurements if entry["predict_action_ms"] is not None],
        dtype=np.float64,
    )
    cache_bytes = [int(entry["cache_bytes"]) for entry in measurements if entry.get("cache_bytes") is not None]
    cache_entries = [int(entry["cache_entries"]) for entry in measurements if entry.get("cache_entries") is not None]
    cache_hits = sum(1 for entry in measurements if entry.get("cache_hit") is True)
    cache_misses = sum(1 for entry in measurements if entry.get("cache_hit") is False)

    summary = {
        "scenario": name,
        "requests": len(measurements),
        "cache_hits": int(cache_hits),
        "cache_misses": int(cache_misses),
        "cache_hit_rate": float(cache_hits / len(measurements)) if measurements else None,
        "request_latency_ms": {
            "mean": float(request_latencies.mean()),
            "p50": float(np.percentile(request_latencies, 50)),
            "p95": float(np.percentile(request_latencies, 95)),
            "min": float(request_latencies.min()),
            "max": float(request_latencies.max()),
        },
        "throughput_rps": float(1000.0 / request_latencies.mean()) if request_latencies.mean() > 0 else None,
    }

    if server_latencies.size > 0:
        summary["predict_action_ms"] = {
            "mean": float(server_latencies.mean()),
            "p50": float(np.percentile(server_latencies, 50)),
            "p95": float(np.percentile(server_latencies, 95)),
            "min": float(server_latencies.min()),
            "max": float(server_latencies.max()),
        }

    if cache_bytes:
        summary["cache_bytes"] = {
            "last": int(cache_bytes[-1]),
            "max": int(max(cache_bytes)),
        }

    if cache_entries:
        summary["cache_entries"] = {
            "last": int(cache_entries[-1]),
            "max": int(max(cache_entries)),
        }

    return summary


def build_comparison_summary(
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
) -> Dict[str, Any]:
    def _compare_metric(metric_name: str) -> Dict[str, float] | None:
        baseline_metric = baseline_summary.get(metric_name)
        candidate_metric = candidate_summary.get(metric_name)
        if baseline_metric is None or candidate_metric is None:
            return None

        baseline_mean = float(baseline_metric["mean"])
        candidate_mean = float(candidate_metric["mean"])
        mean_delta = baseline_mean - candidate_mean
        mean_delta_pct = (mean_delta / baseline_mean * 100.0) if baseline_mean else None
        speedup = (baseline_mean / candidate_mean) if candidate_mean else None
        return {
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "mean_delta": mean_delta,
            "mean_delta_pct": mean_delta_pct,
            "speedup_factor": speedup,
        }

    baseline_throughput = baseline_summary.get("throughput_rps")
    candidate_throughput = candidate_summary.get("throughput_rps")
    throughput = None
    if baseline_throughput is not None and candidate_throughput is not None:
        throughput_delta = candidate_throughput - baseline_throughput
        throughput_delta_pct = (throughput_delta / baseline_throughput * 100.0) if baseline_throughput else None
        throughput = {
            "baseline": float(baseline_throughput),
            "candidate": float(candidate_throughput),
            "delta": float(throughput_delta),
            "delta_pct": float(throughput_delta_pct) if throughput_delta_pct is not None else None,
            "speedup_factor": (float(candidate_throughput / baseline_throughput) if baseline_throughput else None),
        }

    return {
        "baseline_scenario": baseline_summary["scenario"],
        "candidate_scenario": candidate_summary["scenario"],
        "request_latency_ms": _compare_metric("request_latency_ms"),
        "predict_action_ms": _compare_metric("predict_action_ms"),
        "throughput_rps": throughput,
        "cache_hit_rate": {
            "baseline": baseline_summary.get("cache_hit_rate"),
            "candidate": candidate_summary.get("cache_hit_rate"),
        },
    }


def run_scenario(
    *,
    client: WebsocketClientPolicy,
    scenario_name: str,
    request_payload: Dict[str, Any],
    warmup: int,
    runs: int,
    reset_between_requests: bool,
    sleep_ms: int,
) -> Dict[str, Any]:
    client.reset_cache(request_id=f"{scenario_name}-start")

    for warmup_idx in range(warmup):
        if reset_between_requests:
            client.reset_cache(request_id=f"{scenario_name}-warmup-reset-{warmup_idx}")
        warmup_payload = dict(request_payload)
        warmup_payload["request_id"] = f"{scenario_name}-warmup-{warmup_idx}"
        client.infer(warmup_payload)

    measurements: List[Dict[str, Any]] = []
    for run_idx in range(runs):
        if reset_between_requests:
            client.reset_cache(request_id=f"{scenario_name}-reset-{run_idx}")

        request = dict(request_payload)
        request["request_id"] = f"{scenario_name}-run-{run_idx}"
        started_at = time.perf_counter()
        response = client.infer(request)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        if response.get("status") != "ok":
            raise RuntimeError(f"Server returned error for {scenario_name}: {response}")

        cache_info = response.get("data", {}).get("cache_info", {})
        measurements.append(
            {
                "request_latency_ms": elapsed_ms,
                "predict_action_ms": response.get("metrics", {}).get("predict_action_ms"),
                "cache_hit": cache_info.get("hit"),
                "cache_bytes": cache_info.get("cache_bytes"),
                "cache_entries": cache_info.get("cache_entries"),
                "cache_info": cache_info,
            }
        )

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    return {
        "summary": summarize_measurements(scenario_name, measurements),
        "measurements": measurements,
    }


def print_summary(result: Dict[str, Any]) -> None:
    summary = result["summary"]
    request_stats = summary["request_latency_ms"]
    print(f"\n[{summary['scenario']}]")
    print(
        "requests={requests} cache_hits={cache_hits} cache_misses={cache_misses} hit_rate={hit_rate:.2%} throughput={throughput:.2f} req/s".format(
            requests=summary["requests"],
            cache_hits=summary["cache_hits"],
            cache_misses=summary["cache_misses"],
            hit_rate=summary["cache_hit_rate"] or 0.0,
            throughput=summary["throughput_rps"] or 0.0,
        )
    )
    print(
        "request_latency_ms: mean={mean:.2f} p50={p50:.2f} p95={p95:.2f} min={min:.2f} max={max:.2f}".format(
            **request_stats,
        )
    )

    predict_action_stats = summary.get("predict_action_ms")
    if predict_action_stats is not None:
        print(
            "predict_action_ms: mean={mean:.2f} p50={p50:.2f} p95={p95:.2f} min={min:.2f} max={max:.2f}".format(
                **predict_action_stats,
            )
        )

    cache_bytes = summary.get("cache_bytes")
    cache_entries = summary.get("cache_entries")
    if cache_bytes is not None or cache_entries is not None:
        print(
            "cache_footprint: entries_last={entries_last} entries_max={entries_max} bytes_last={bytes_last} bytes_max={bytes_max}".format(
                entries_last=(cache_entries or {}).get("last", 0),
                entries_max=(cache_entries or {}).get("max", 0),
                bytes_last=(cache_bytes or {}).get("last", 0),
                bytes_max=(cache_bytes or {}).get("max", 0),
            )
        )


def print_comparison(comparison: Dict[str, Any]) -> None:
    print(
        "\n[comparison] {candidate} vs {baseline}".format(
            candidate=comparison["candidate_scenario"],
            baseline=comparison["baseline_scenario"],
        )
    )

    request_latency = comparison.get("request_latency_ms")
    if request_latency is not None:
        print(
            "request_latency_ms: {baseline_mean:.2f} -> {candidate_mean:.2f} ({mean_delta_pct:.2f}% lower, {speedup_factor:.2f}x faster)".format(
                **request_latency,
            )
        )

    predict_action = comparison.get("predict_action_ms")
    if predict_action is not None:
        print(
            "predict_action_ms: {baseline_mean:.2f} -> {candidate_mean:.2f} ({mean_delta_pct:.2f}% lower, {speedup_factor:.2f}x faster)".format(
                **predict_action,
            )
        )

    throughput = comparison.get("throughput_rps")
    if throughput is not None:
        print(
            "throughput_rps: {baseline:.2f} -> {candidate:.2f} ({delta_pct:.2f}% higher, {speedup_factor:.2f}x)".format(
                **throughput,
            )
        )

    cache_hit_rate = comparison.get("cache_hit_rate")
    if cache_hit_rate is not None:
        print(
            "cache_hit_rate: baseline={baseline:.2%} candidate={candidate:.2%}".format(
                baseline=cache_hit_rate.get("baseline") or 0.0,
                candidate=cache_hit_rate.get("candidate") or 0.0,
            )
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark repeated policy-server requests with and without cache reuse.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--image", action="append", required=True, help="Path to an RGB image. Repeat for multiview input.")
    parser.add_argument("--instruction", required=True, help="Instruction text sent to the policy.")
    parser.add_argument("--state", default="", help="Optional state as JSON list or comma-separated floats.")
    parser.add_argument("--runs", type=int, default=10, help="Measured requests per scenario.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests per scenario.")
    parser.add_argument(
        "--mode",
        choices=["compare", "reuse", "cold"],
        default="compare",
        help="compare: run both repeated reuse and reset-per-request. reuse: one reset then repeated requests. cold: reset before every request.",
    )
    parser.add_argument("--sleep-ms", type=int, default=0, help="Optional pause between measured requests.")
    parser.add_argument("--disable-vlm-cache", action="store_true", help="Send use_vlm_cache=False in requests.")
    parser.add_argument("--output-json", default="", help="Optional path for a JSON report.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    for image_path in args.image:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    request_payload = build_request_payload(
        image_paths=args.image,
        instruction=args.instruction,
        state=parse_state(args.state),
        use_vlm_cache=not args.disable_vlm_cache,
    )

    client = WebsocketClientPolicy(host=args.host, port=args.port, api_key=(args.api_key or None))
    report: Dict[str, Any] = {
        "server_metadata": client.get_server_metadata(),
        "request_config": {
            "images": args.image,
            "instruction": args.instruction,
            "runs": args.runs,
            "warmup": args.warmup,
            "mode": args.mode,
            "sleep_ms": args.sleep_ms,
            "use_vlm_cache": not args.disable_vlm_cache,
        },
        "results": [],
    }

    try:
        scenarios = []
        if args.mode == "compare":
            scenarios = [
                ("cold_reset_each_request", True),
                ("reuse_same_session", False),
            ]
        elif args.mode == "cold":
            scenarios = [("cold_reset_each_request", True)]
        else:
            scenarios = [("reuse_same_session", False)]

        for scenario_name, reset_between_requests in scenarios:
            result = run_scenario(
                client=client,
                scenario_name=scenario_name,
                request_payload=request_payload,
                warmup=args.warmup,
                runs=args.runs,
                reset_between_requests=reset_between_requests,
                sleep_ms=args.sleep_ms,
            )
            print_summary(result)
            report["results"].append(result)

        if len(report["results"]) == 2:
            comparison = build_comparison_summary(
                report["results"][0]["summary"],
                report["results"][1]["summary"],
            )
            report["comparison"] = comparison
            print_comparison(comparison)
    finally:
        client.close()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote benchmark report to {output_path}")


if __name__ == "__main__":
    main()
