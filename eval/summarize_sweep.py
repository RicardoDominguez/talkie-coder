#!/usr/bin/env python3
"""Aggregate pass@5 results across the minimum-data sweep runs.

Reads per-pass-run final reports and computes:
- pass@1 mean (avg resolved across the 5 pass-runs)
- pass@5 union (resolved by ANY of the 5 pass-runs)
- per-instance solve frequency
- repo coverage

Usage:
    python summarize_sweep.py
"""
import json
import glob
import os
from collections import Counter

SWEEP = [
    ("Run 3 (s20_e3)",  "talkie-1930-it-coder-s20-e3-pass5-union42",
     {"max_steps": 20,  "epochs": 3,  "examples": 251}),
    ("Run 4 (d251_e10)","talkie-1930-it-coder-d251-e10-pass5-union42",
     {"max_steps": 67,  "epochs": 10, "examples": 251}),
    ("Run 2 (s70_e3)",  "talkie-1930-it-coder-s70-e3-pass5-union42",
     {"max_steps": 70,  "epochs": 3,  "examples": 881}),
    ("Run 1 (s200_e3)", "talkie-1930-it-coder-s200-e3-pass5-union42",
     {"max_steps": 200, "epochs": 3,  "examples": 2518}),
    ("Run 5 (d881_e9)", "talkie-1930-it-coder-d881-e9-pass5-union42",
     {"max_steps": 216, "epochs": 9,  "examples": 881}),
]

UNION42_FILE = "/home/rolmedo/talkie/sft/swe_bench_eval_union42.json"


def repo(iid):
    return iid.split("__")[0]


def read_report(path):
    try:
        d = json.load(open(path))
        return set(d.get("resolved_ids", [])), d.get("submitted_instances", 0), d.get("completed_instances", 0)
    except Exception:
        return None, 0, 0


def main():
    expected_n = json.load(open(UNION42_FILE))["n_instances"]
    print(f"Eval pool: {expected_n} instances (union42)")
    print()
    print(f"{'Run':22s}  {'pass@1 (mean)':>14s}  {'pass@5 (union)':>14s}  {'completion':>10s}  {'cfg':>30s}")
    print("-" * 105)

    all_rows = []
    for label, tag, cfg in SWEEP:
        per_run = []
        completions = []
        for i in range(1, 6):
            base = f"/fast/rolmedo/swesmith/{tag}-run{i}"
            report_path = f"{base}/{tag}-run{i}.json"
            resolved, sub, comp = read_report(report_path)
            if resolved is None:
                continue
            per_run.append(resolved)
            completions.append(comp)
        if not per_run:
            print(f"{label:22s}  (no graded reports yet)")
            continue
        n_runs = len(per_run)
        mean_resolved = sum(len(r) for r in per_run) / n_runs
        union = set().union(*per_run)
        avg_comp = sum(completions) / n_runs
        cfg_s = f"steps={cfg['max_steps']:3d} ep={cfg['epochs']:2d} d={cfg['examples']:4d}"
        print(f"{label:22s}  {mean_resolved:6.1f} ({n_runs}/5)  {len(union):6d}/{expected_n}     {avg_comp:5.1f}/{expected_n}    {cfg_s}")
        all_rows.append((label, cfg, per_run, union))

    # Per-instance solve frequency across all runs
    print()
    print("=== Per-instance solve frequency (across all runs × all pass-runs) ===")
    total_solves = Counter()
    for _, _, per_run, _ in all_rows:
        for r in per_run:
            for iid in r:
                total_solves[iid] += 1
    print(f"{'instance_id':45s}  count")
    for iid, c in total_solves.most_common():
        print(f"{iid:45s}  {c}")

    # Repo breakdown of pass@5 union per run
    print()
    print("=== pass@5 union by repo, per run ===")
    print(f"{'Run':22s}  ", end="")
    repos = sorted({repo(iid) for _, _, _, u in all_rows for iid in u})
    for r in repos:
        print(f"{r:>14s}", end="")
    print()
    for label, _, _, union in all_rows:
        print(f"{label:22s}  ", end="")
        c = Counter(repo(iid) for iid in union)
        for r in repos:
            print(f"{c.get(r, 0):>14d}", end="")
        print()


if __name__ == "__main__":
    main()
