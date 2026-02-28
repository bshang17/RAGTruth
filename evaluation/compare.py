#!/usr/bin/env python3
"""Compare hallucination rates between original and unlearned model responses."""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-gpt", help="GPT judge results for original model")
    p.add_argument("--unlearned-gpt", help="GPT judge results for unlearned model")
    p.add_argument("--original-baseline", help="Baseline detector results for original model")
    p.add_argument("--unlearned-baseline", help="Baseline detector results for unlearned model")
    p.add_argument("--ground-truth", default=None,
                    help="Path to dataset/response.jsonl for judge validation")
    p.add_argument("--filter-model", default=None,
                    help="Filter ground truth to this model")
    p.add_argument("--output-json", default=None, help="Save results as JSON")
    return p.parse_args()


def load_judged(path):
    """Load judge output JSONL into a list of dicts."""
    if not path:
        return []
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def hallucination_rate(records):
    """Compute hallucination rate overall and per task_type."""
    if not records:
        return {}
    total = len(records)
    halu = sum(1 for r in records if r["pred_halu"])
    results = {"overall": {"rate": halu / total, "halu": halu, "total": total}}

    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_type"]].append(r)

    for task, recs in sorted(by_task.items()):
        h = sum(1 for r in recs if r["pred_halu"])
        results[task] = {"rate": h / len(recs), "halu": h, "total": len(recs)}

    return results


def mean_halu_count(records):
    """Average number of hallucination spans per response."""
    if not records:
        return 0.0
    counts = [len(r["pred"].get("hallucination list", [])) for r in records]
    return np.mean(counts)


def judge_validation(judged_records):
    """Compute precision/recall/F1 of judge vs ground truth labels.

    Only works when records have both 'is_halu' (ground truth) and 'pred_halu'."""
    records_with_gt = [r for r in judged_records if "is_halu" in r]
    if not records_with_gt:
        return None

    tp = sum(1 for r in records_with_gt if r["is_halu"] and r["pred_halu"])
    fp = sum(1 for r in records_with_gt if not r["is_halu"] and r["pred_halu"])
    fn = sum(1 for r in records_with_gt if r["is_halu"] and not r["pred_halu"])
    tn = sum(1 for r in records_with_gt if not r["is_halu"] and not r["pred_halu"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": len(records_with_gt)}


def mcnemar_test(original, unlearned):
    """McNemar's test for paired binary outcomes on matching source_ids.

    Returns (chi2, p_value) or None if insufficient data."""
    try:
        from scipy.stats import chi2 as chi2_dist
    except ImportError:
        return None

    # Match by source_id
    orig_by_sid = {r["source_id"]: r["pred_halu"] for r in original}
    unl_by_sid = {r["source_id"]: r["pred_halu"] for r in unlearned}
    common = set(orig_by_sid) & set(unl_by_sid)

    if len(common) < 10:
        return None

    # b = orig halu, unlearned not halu (improved)
    # c = orig not halu, unlearned halu (worsened)
    b = sum(1 for sid in common if orig_by_sid[sid] and not unl_by_sid[sid])
    c = sum(1 for sid in common if not orig_by_sid[sid] and unl_by_sid[sid])

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c, "n_paired": len(common)}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {"chi2": chi2, "p_value": p_value, "b_improved": b,
            "c_worsened": c, "n_paired": len(common)}


def inter_judge_agreement(records_a, records_b):
    """Compute agreement rate between two judges on the same records."""
    a_by_key = {r.get("_key", r["source_id"]): r["pred_halu"] for r in records_a}
    b_by_key = {r.get("_key", r["source_id"]): r["pred_halu"] for r in records_b}
    common = set(a_by_key) & set(b_by_key)

    if not common:
        return None
    agree = sum(1 for k in common if a_by_key[k] == b_by_key[k])
    return {"agreement": agree / len(common), "n": len(common)}


def print_comparison(label, orig_rates, unl_rates):
    """Print a comparison table for one judge."""
    if not orig_rates or not unl_rates:
        return

    print(f"\n{'=' * 60}")
    print(f"Judge: {label}")
    print(f"{'=' * 60}")
    print(f"{'':20s} {'Original':>12s} {'Unlearned':>12s} {'Delta':>12s}")
    print(f"{'-' * 60}")

    for key in ["overall", "QA", "Summary", "Data2txt"]:
        o = orig_rates.get(key, {})
        u = unl_rates.get(key, {})
        if not o and not u:
            continue
        o_rate = o.get("rate", 0)
        u_rate = u.get("rate", 0)
        delta = u_rate - o_rate
        print(f"{key:20s} {o_rate:11.1%} {u_rate:12.1%} {delta:+11.1%}")


def main():
    args = parse_args()
    all_results = {}

    # Load judge outputs
    orig_gpt = load_judged(args.original_gpt)
    unl_gpt = load_judged(args.unlearned_gpt)
    orig_base = load_judged(args.original_baseline)
    unl_base = load_judged(args.unlearned_baseline)

    # Compute hallucination rates
    orig_gpt_rates = hallucination_rate(orig_gpt)
    unl_gpt_rates = hallucination_rate(unl_gpt)
    orig_base_rates = hallucination_rate(orig_base)
    unl_base_rates = hallucination_rate(unl_base)

    # Print comparison tables
    if orig_gpt and unl_gpt:
        print_comparison("GPT", orig_gpt_rates, unl_gpt_rates)
        all_results["gpt"] = {
            "original": orig_gpt_rates, "unlearned": unl_gpt_rates,
            "mean_halu_count_orig": mean_halu_count(orig_gpt),
            "mean_halu_count_unl": mean_halu_count(unl_gpt),
        }

    if orig_base and unl_base:
        print_comparison("Baseline Detector", orig_base_rates, unl_base_rates)
        all_results["baseline"] = {
            "original": orig_base_rates, "unlearned": unl_base_rates,
            "mean_halu_count_orig": mean_halu_count(orig_base),
            "mean_halu_count_unl": mean_halu_count(unl_base),
        }

    # Judge validation (if ground truth is in the original judge outputs)
    print(f"\n{'=' * 60}")
    print("Judge Validation (vs. human annotations)")
    print(f"{'=' * 60}")
    for label, records in [("GPT", orig_gpt), ("Baseline", orig_base)]:
        val = judge_validation(records)
        if val:
            print(f"{label}: P={val['precision']:.3f} R={val['recall']:.3f} "
                  f"F1={val['f1']:.3f} (n={val['total']})")
            all_results[f"{label.lower()}_validation"] = val

    # Statistical significance
    print(f"\n{'=' * 60}")
    print("Statistical Significance (McNemar's test)")
    print(f"{'=' * 60}")
    for label, orig, unl in [("GPT", orig_gpt, unl_gpt),
                              ("Baseline", orig_base, unl_base)]:
        if orig and unl:
            result = mcnemar_test(orig, unl)
            if result:
                print(f"{label}: chi2={result['chi2']:.3f}, p={result['p_value']:.4f} "
                      f"(improved={result['b_improved']}, worsened={result['c_worsened']}, "
                      f"paired={result['n_paired']})")
                all_results[f"{label.lower()}_mcnemar"] = result
            else:
                print(f"{label}: insufficient data or scipy not installed")

    # Inter-judge agreement
    print(f"\n{'=' * 60}")
    print("Inter-Judge Agreement")
    print(f"{'=' * 60}")
    for label, a, b in [("Original responses", orig_gpt, orig_base),
                         ("Unlearned responses", unl_gpt, unl_base)]:
        if a and b:
            ag = inter_judge_agreement(a, b)
            if ag:
                print(f"{label}: {ag['agreement']:.1%} agreement (n={ag['n']})")
                all_results[f"agreement_{label.lower().replace(' ', '_')}"] = ag

    # Save JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
