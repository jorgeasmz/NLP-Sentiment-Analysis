"""
Measures the served model on a curated set and reports inference latency.

The checkpoint is pre-trained, so there is nothing to tune here. What matters
for a service is where it is wrong and how fast it answers, so the evaluation
set is grouped by the linguistic phenomenon each case probes: negation, mixed
sentiment and sarcasm are where a sentence-level classifier tends to break.

Usage: python evaluate.py
"""

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from core.config import MODEL_NAME
from core.service import analyze_text

EVAL_SET = Path(__file__).resolve().parent / "data" / "eval_set.json"


def percentile(values: list, fraction: float) -> float:
    """Nearest-rank percentile; the sample is far too small for interpolation."""
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    print(f"Model: {MODEL_NAME}")
    print(f"Cases: {len(cases)}\n")

    print("Warming up...")
    analyze_text("warm up")

    latencies = []
    by_category = defaultdict(lambda: {"total": 0, "correct": 0})
    failures = []

    for case in cases:
        started = time.perf_counter()
        result = analyze_text(case["text"])
        latencies.append((time.perf_counter() - started) * 1000)

        correct = result["label"] == case["expected"]
        bucket = by_category[case["category"]]
        bucket["total"] += 1
        bucket["correct"] += int(correct)

        if not correct:
            failures.append((case, result))

    total = len(cases)
    correct = sum(b["correct"] for b in by_category.values())

    print(f"\nAccuracy: {correct}/{total} = {correct / total:.1%}\n")

    header = f"{'Category':<18}{'Correct':>9}{'Total':>7}{'Accuracy':>11}"
    print(header)
    print("-" * len(header))
    for category in sorted(by_category):
        bucket = by_category[category]
        print(
            f"{category:<18}{bucket['correct']:>9}{bucket['total']:>7}"
            f"{bucket['correct'] / bucket['total']:>10.0%}"
        )

    print(f"\nLatency over {len(latencies)} calls (ms, CPU, batch of one):")
    print(f"  mean {statistics.mean(latencies):.1f}")
    print(f"  p50  {percentile(latencies, 0.50):.1f}")
    print(f"  p95  {percentile(latencies, 0.95):.1f}")
    print(f"  max  {max(latencies):.1f}")

    if failures:
        print(f"\nMisclassified ({len(failures)}):")
        for case, result in failures:
            print(f"  [{case['category']}] expected {case['expected']}, "
                  f"got {result['label']} ({result['score']:.2f})")
            print(f"    {case['text']}")


if __name__ == "__main__":
    main()
