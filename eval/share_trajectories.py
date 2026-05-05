#!/usr/bin/env python3
"""
Bundle the 1930 + web SFT pass@N trajectories into a public HF dataset.

  python eval/share_trajectories.py --out /tmp/talkie-trajs --dry-run
  python eval/share_trajectories.py --out /tmp/talkie-trajs --repo-id <user>/<dataset>

Output layout (HF auto-split convention, single 'test' split):

  data/test-NNNNN-of-NNNNN.parquet   one shard per (model, run)
  config_redacted.json               agent/model/env config snapshot, paths scrubbed
  README.md                          dataset card with DO-NOT-TRAIN banner
  WARNING.md                         standalone warning file
"""

import argparse
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
SWESMITH = Path("/fast/rolmedo/swesmith")
GRADES = REPO_ROOT / "analysis"

# (model, run, traj_dir, grade_json)
RUNS = [
    ("1930", n,
     SWESMITH / f"talkie-1930-v2-lr2e5-ckpt2000-mini-446-pass5-run{n}",
     GRADES / f"talkie-1930-v2-lr2e5-ckpt2000-pass5-run{n}.json")
    for n in range(1, 6)
] + [
    ("web", n,
     SWESMITH / f"talkie-web-v2-lr2e5-ckpt2000-mini-446-pass3-run{n}",
     GRADES / f"talkie-web-v2-lr2e5-ckpt2000-pass3-run{n}.json")
    for n in range(1, 4)
]

WARNING = (
    "DO NOT TRAIN ON THIS, THIS IS TEST DATA. "
    "These are SWE-bench-Verified evaluation trajectories — "
    "training on them contaminates the benchmark."
)

REDACT = [
    (re.compile(r"/fast/rolmedo/[^\s\"'<>]*"), "/fast/<user>/<redacted>"),
    (re.compile(r"/lustre/home/rolmedo/[^\s\"'<>]*"), "/lustre/home/<user>/<redacted>"),
    (re.compile(r"/home/rolmedo/[^\s\"'<>]*"), "/home/<user>/<redacted>"),
    (re.compile(r"harbor\.is\.localnet"), "<internal-registry>"),
    (re.compile(r"http://[a-z0-9\-]+:\d+/v1"), "http://<vllm-host>/v1"),
]


def redact(s):
    if not isinstance(s, str):
        return s
    for pat, repl in REDACT:
        s = pat.sub(repl, s)
    return s


def collect_run(model, run, traj_dir, grade_path):
    if not traj_dir.is_dir():
        raise SystemExit(f"missing trajectory dir: {traj_dir}")
    if not grade_path.is_file():
        raise SystemExit(f"missing grade report: {grade_path}")

    grade = json.loads(grade_path.read_text())
    resolved = set(grade.get("resolved_ids", []))

    rows = []
    cfg_sample = None
    for p in sorted(traj_dir.rglob("*.traj.json")):
        traj = json.loads(p.read_text())
        info = traj.get("info", {})
        if cfg_sample is None and "config" in info:
            cfg_sample = info["config"]
        instance_id = traj.get("instance_id") or p.stem.removesuffix(".traj")
        messages = [
            {"role": m.get("role", ""), "content": redact(m.get("content", ""))}
            for m in traj.get("messages", [])
        ]
        rows.append({
            "instance_id": instance_id,
            "model": model,                           # "1930" or "web"
            "run": run,
            "exit_status": info.get("exit_status") or "",
            "resolved": instance_id in resolved,      # graded by swebench harness
            "submission": info.get("submission") or "",
            "n_turns": len(messages),
            "messages": messages,
            "warning": WARNING,
        })
    return rows, cfg_sample


def write_dataset(out_dir, runs):
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg_redacted = None
    totals = {"1930": 0, "web": 0}
    resolved_totals = {"1930": 0, "web": 0}

    runs_by_model = {}
    for r in runs:
        runs_by_model.setdefault(r[0], []).append(r)

    for model, mruns in runs_by_model.items():
        (data_dir / model).mkdir(parents=True, exist_ok=True)
        n = len(mruns)
        for i, (_, run, traj_dir, grade_path) in enumerate(mruns):
            rows, cfg = collect_run(model, run, traj_dir, grade_path)
            if cfg_redacted is None and cfg is not None:
                cfg_redacted = json.loads(redact(json.dumps(cfg)))

            table = pa.Table.from_pylist(rows)
            name = f"test-{i:05d}-of-{n:05d}.parquet"
            pq.write_table(table, data_dir / model / name, compression="zstd")

            totals[model] += len(rows)
            resolved_totals[model] += sum(r["resolved"] for r in rows)
            print(f"  {model}/{name}: run{run}  rows={len(rows)}  resolved={resolved_totals[model]}")

    (out_dir / "config_redacted.json").write_text(
        json.dumps(cfg_redacted, indent=2) + "\n"
    )
    (out_dir / "WARNING.md").write_text(
        f"# {WARNING}\n\n"
        "Every row in this dataset carries a `warning` column with the same text. "
        "If you are building an SFT dataset, exclude this corpus.\n"
    )
    (out_dir / "README.md").write_text(render_readme(totals, resolved_totals))
    return totals, resolved_totals


MODEL_INFO = {
    "1930": {
        "title": "Talkie 1930 — SFT eval trajectories",
        "model_id": "talkie-lm/talkie-1930-13b",
        "n_runs": 5,
        "pass1": "4.48% (σ=0.69 pp, 5 runs)",
        "blurb": (
            "Mini-SWE-Agent trajectories from the 2e-5 SFT run of "
            "[`talkie-lm/talkie-1930-13b`](https://huggingface.co/talkie-lm/talkie-1930-13b), "
            "the pre-1931-pretrained 13B base model."
        ),
        "sibling_dataset": "ricdomolm/eval-trajs-web-coder",
        "sibling_label": "web variant",
    },
    "web": {
        "title": "Talkie Web — SFT eval trajectories",
        "model_id": "talkie-lm/talkie-web-13b",
        "n_runs": 3,
        "pass1": "5.75% (σ=1.04 pp, 3 runs)",
        "blurb": (
            "Mini-SWE-Agent trajectories from the 2e-5 SFT run of "
            "[`talkie-lm/talkie-web-13b`](https://huggingface.co/talkie-lm/talkie-web-13b), "
            "the FineWeb-pretrained 13B base model."
        ),
        "sibling_dataset": "ricdomolm/eval-trajs-1930-coder",
        "sibling_label": "1930 variant",
    },
}


def render_readme(totals, resolved_totals):
    models = [m for m in ("1930", "web") if totals[m] > 0]
    if len(models) == 1:
        m = models[0]
        info = MODEL_INFO[m]
        body_title = info["title"]
        body_blurb = info["blurb"]
        sibling_line = (
            f"For the {info['sibling_label']}, see "
            f"[`{info['sibling_dataset']}`](https://huggingface.co/datasets/{info['sibling_dataset']}).\n"
        )
        table_rows = (
            f"| {m} | {info['n_runs']} | {totals[m]:,} | {resolved_totals[m]:,} |\n"
        )
        pass1_lines = f"- **{m}**: {info['pass1']}\n"
    else:
        body_title = "Talkie 1930 / Web — SFT eval trajectories"
        body_blurb = (
            "Mini-SWE-Agent trajectories from the 2e-5 SFT runs of "
            "[`talkie-lm/talkie-1930-13b`](https://huggingface.co/talkie-lm/talkie-1930-13b) "
            "and [`talkie-lm/talkie-web-13b`](https://huggingface.co/talkie-lm/talkie-web-13b)."
        )
        sibling_line = ""
        table_rows = "".join(
            f"| {m} | {MODEL_INFO[m]['n_runs']} | {totals[m]:,} | {resolved_totals[m]:,} |\n"
            for m in models
        )
        pass1_lines = "".join(
            f"- **{m}**: {MODEL_INFO[m]['pass1']}\n" for m in models
        )

    config_blocks = []
    for idx, m in enumerate(models):
        default_line = "  default: true\n" if idx == 0 and len(models) > 1 else ""
        config_blocks.append(
            f"- config_name: \"{m}\"\n"
            f"{default_line}"
            f"  data_files:\n"
            f"  - split: test\n"
            f"    path: data/{m}/test-*.parquet"
        )
    configs_yaml = "\n".join(config_blocks)

    return f"""---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- swe-bench
- agent-trajectories
- evaluation
size_categories:
- 1K<n<10K
configs:
{configs_yaml}
---

> # ⚠️ DO NOT TRAIN ON THIS — THIS IS TEST DATA ⚠️
>
> These trajectories are evaluation outputs on **SWE-bench-Verified**.
> Training on them (directly, or via distillation, rejection sampling,
> or any form of preference data) **contaminates the benchmark**.
> Every row carries a `warning` column repeating this notice.

# {body_title}

{body_blurb}

Graded by the [SWE-bench harness](https://github.com/SWE-bench/SWE-bench)
against the 446-instance
[`ricdomolm/SWE-bench_Verified-Working-Harbor`](https://huggingface.co/datasets/ricdomolm/SWE-bench_Verified-Working-Harbor)
subset.

| Model | Runs | Trajectories | Resolved (sum across runs) |
|-------|-----:|-------------:|---------------------------:|
{table_rows}
Pass@1 (mean across runs):
{pass1_lines}
{sibling_line}See the [training repo](https://github.com/RicardoDominguez/talkie-public)
for the SFT recipe, eval pipeline, and analysis notebook.

## Schema

One row per agent trajectory.

| Column | Type | Notes |
|---|---|---|
| `instance_id` | string | SWE-bench-Verified instance id |
| `model` | string | `"1930"` or `"web"` |
| `run` | int | Run index (1..5 for 1930, 1..3 for web) |
| `exit_status` | string | mini-swe-agent terminal state (`Submitted`, `LimitsExceeded`, …) |
| `resolved` | bool | Graded by the swebench harness against the gold tests |
| `submission` | string | Final unified diff submitted (may be empty) |
| `n_turns` | int | Number of messages in the trajectory |
| `messages` | list&lt;struct&gt; | `[{{role, content}}, ...]` — system + user + agent turns |
| `warning` | string | Constant `DO NOT TRAIN…` notice |

## Loading

Each model lives in its own subset; pick one as the second positional arg.

```python
from datasets import load_dataset

ds_1930 = load_dataset("ricdomolm/eval-trajs-1930-coder", "1930", split="test")
ds_web  = load_dataset("ricdomolm/eval-trajs-1930-coder", "web",  split="test")
print(ds_1930[0]["instance_id"], ds_1930[0]["resolved"], ds_1930[0]["n_turns"])
```

## Provenance

- Generated by mini-swe-agent v1.10.0 against the SWE-bench-Verified-Working-Harbor
  Docker images, with vLLM 0.19 serving the SFT checkpoints (bf16,
  `temperature=0.7`, `max_tokens=4096`, `max-model-len=32768`).
- The `config_redacted.json` at the repo root is a sample of `info.config`
  from one trajectory (agent prompts, model kwargs, env spec) with cluster
  paths scrubbed. The same config drove every trajectory in this dataset.

## License

Apache-2.0, matching the upstream models. Trajectory contents are model
generations grounded on public SWE-bench-Verified instances.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="staging dir for the dataset")
    ap.add_argument("--model", choices=["1930", "web", "both"], default="both",
                    help="which model's trajectories to include")
    ap.add_argument("--repo-id", help="HF dataset repo id; omit to dry-run")
    ap.add_argument("--private", action="store_true", help="create as private")
    args = ap.parse_args()

    runs = RUNS if args.model == "both" else [r for r in RUNS if r[0] == args.model]
    if not runs:
        raise SystemExit(f"no runs match --model={args.model}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    totals, resolved = write_dataset(out, runs)
    total_rows = sum(totals.values())
    total_resolved = sum(resolved.values())
    print(f"\nwrote {total_rows} rows ({total_resolved} resolved) to {out}")

    if not args.repo_id:
        print("dry-run: pass --repo-id to upload")
        return

    from huggingface_hub import HfApi, CommitOperationDelete
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)

    existing = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    stale = [f for f in existing
             if f.startswith("data/") and "/" not in f[len("data/"):]]
    if stale:
        api.create_commit(
            repo_id=args.repo_id, repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
            commit_message="Remove flat data layout",
        )
        print(f"  removed {len(stale)} stale files")

    api.upload_folder(folder_path=str(out), repo_id=args.repo_id,
                      repo_type="dataset",
                      commit_message="Upload talkie SFT eval trajectories")
    print(f"uploaded: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
