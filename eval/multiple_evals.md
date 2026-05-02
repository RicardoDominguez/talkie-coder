# Running multiple parallel SWE-bench evals (variance-reduced pass@1)

How to run K independent full-446 SWE-bench evals against the same checkpoint,
with a 2h per-sub-job wallclock cap and automatic re-submission of crashed
sub-jobs.

The point of running K samples per instance is to **reduce the variance of the
pass@1 estimate**. With temp 0.7 sampling, a single run swings several
percentage points across instances; averaging K runs gives a much tighter
estimate of the model's true resolved rate. We do *not* use this to report
pass@k metrics — pass@1 is the headline number.

Worked example below uses K=5 against
`/fast/rolmedo/swe-models/talkie-1930-v2-lr2e5-ckpt2000-vllm`.

---

## 1. Prerequisites

These already exist for the talkie-eval pipeline; verify before launching:

- The checkpoint is repackaged for vLLM under `/fast/rolmedo/swe-models/<name>-vllm/`
  (see `EVAL_HANDOVER.md` step 1).
- The same name is registered in
  `/home/rolmedo/mini-swe-agent/model_prices_and_context_window.json`
  (see `EVAL_HANDOVER.md` step 2).
- Pre-pulled SWE-bench harbor tarballs at `/fast/rolmedo/swesmith/docker_tarballs/`.

No script changes are needed — we reuse the standard
`launch_parallel_eval.sh` once per attempt, with a different output dir each
time.

---

## 2. Launch K independent runs

From `/lustre/home/rolmedo/talkie-eval/swe-evals/`:

```bash
MODEL=/fast/rolmedo/swe-models/talkie-1930-v2-lr2e5-ckpt2000-vllm
TAG=talkie-1930-v2-lr2e5-ckpt2000-mini-446-pass5

for i in 1 2 3 4 5; do
  ./launch_parallel_eval.sh \
    "$MODEL" \
    "/fast/rolmedo/swesmith/${TAG}-run${i}" \
    20 446
done
```

This submits K × 20 = 100 condor sub-jobs (20 per run × 23 instances each).
Each sub-job:

- requests 1 GPU (H100 or B200, A100s excluded), 8 CPUs, 256 GB RAM, 300 GB disk
- spins up vLLM + mini-extra at temp 0.7, max_tokens 4096
- has `periodic_remove > 7200s` — self-removes if running > 2h
  (catches "stuck in forever generation" cases)

Each invocation also writes `job_<idx>.sub` files into the run's output dir;
the watcher re-uses them for resubmits.

---

## 3. Start the crash-detect / re-submit watcher

Launch via `at(1)` so the process tree is parented to `atd → init` and
survives shell churn. `pass5_watchdog.sh` is a tiny respawn loop that
restarts the watcher if the python process ever dies (it only exits 0 once
all slices reach a terminal state):

```bash
echo /lustre/home/rolmedo/talkie-eval/swe-evals/pass5_watchdog.sh | at now
```

What the watcher does, every 20 min, for each `(run_dir, sub_file_idx)`:

- Look up the ClusterIds it has submitted (in `pass5_watcher_state.json`).
- If any of those clusters is currently RUNNING or IDLE in queue → wait.
- Otherwise the slice's most recent attempt is terminal:
  - Count `*.traj.json` in `$run_dir/job_sched#<cluster>.0#<schedd>/`.
  - **`cov >= slice size`** → mark **DONE** (and `condor_rm` the cluster if
    it was a phantom-hold).
  - **`cov  < slice size` and elapsed ≥ 1h30** → mark **ENDED_LATE**, no
    retry. This is the user-specified rule: a sub-job that ran a long time
    before dying is treated as a legitimate stuck-trajectory case.
  - **`cov  < slice size` and elapsed <  1h30** and resubmits < 3 →
    `condor_submit_bid 51 job_<idx>.sub`, append the new ClusterId to
    state, increment retry count. Held clusters get `condor_rm`'d first.
  - resubmits ≥ 3 → **GIVE_UP** (logged; no further attempts).

Elapsed time is read from the per-cluster condor user log
(`condor_logs/sub.<cluster>.log`) — first `001 Job executing` event to the
last event timestamp.

State: `/lustre/home/rolmedo/talkie-eval/swe-evals/pass5_watcher_state.json`
Log:   `/lustre/home/rolmedo/talkie-eval/swe-evals/pass5_watcher.log`
Watchdog log: `/lustre/home/rolmedo/talkie-eval/swe-evals/pass5_watchdog.log`

**Do not delete `pass5_watcher_state.json` between runs.** If it is missing,
the watcher will only learn about clusters currently in the queue at
startup; clusters that already terminated cleanly become invisible and the
watcher will spuriously resubmit their slices (the original output dir has
the trajectories under a different `job_sched#<old_cluster>.0#…` path, but
the watcher can't associate that dir with a slice without the state file).
The watchdog respawn loop reuses the same state file on each restart, so
this is only a concern if you manually clear state.

**Per-experiment, edit the constants near the top of `pass5_watcher.py`
in place.** It is a single-experiment-at-a-time tool — the `RUN_DIRS`,
`STATE_FILE`, and `LOG_FILE` are baked into the file (the file name is
historical; the same script is reused for pass@3, pass@5, etc.). Update:

- `RUN_DIRS` — list of base output dirs to watch.
- `STATE_FILE` / `LOG_FILE` — give each experiment its own pair so they
  don't collide on disk (e.g. `pass3_watcher_state.json`).
- `MAX_RESUBMITS` — per-slice retry cap (default 3).
- `CHECK_INTERVAL_SEC` — cycle period (default 20 min).
- `LATE_THRESHOLD_SEC` — "ran a long time before dying" cutoff
  (default 5400 = 1h30).
- `SCHEDD_SUFFIX` — schedd pid in the cluster→dir mapping; verify with
  `find <run_dir> -maxdepth 1 -type d -name 'job_sched*' | awk -F'#' '{print $NF}' | sort -u`
  (a single value across all runs — `7941` at time of writing).

To stop:
```bash
pkill -TERM -f pass5_watchdog.sh
pkill -TERM -f pass5_watcher.py
```

---

## 4. Monitor progress

```bash
# Queue overview
condor_q $USER -totals

# Per-run unique-instance coverage
for i in 1 2 3 4 5; do
  d=/fast/rolmedo/swesmith/talkie-1930-v2-lr2e5-ckpt2000-mini-446-pass5-run${i}
  inst=$(find "$d" -name '*.traj.json' 2>/dev/null \
         | awk -F/ '{print $(NF-1)}' | sort -u | wc -l)
  echo "run${i}: ${inst}/446"
done

# Watcher decisions
tail -f /lustre/home/rolmedo/talkie-eval/swe-evals/pass5_watcher.log
```

A run is finished when its 20 sub-jobs (and any resubmits) are out of the
queue and unique coverage reaches 446 (or stops growing).

---

## 5. Merge predictions and grade, per run

For each run dir, follow `EVAL_HANDOVER.md` steps 4-6: merge sub-job
preds.json files (`$BASE/*/preds.json`, latest-wins, drop empty patches),
submit a grading job, then read `<RUN_ID>.json`.

```bash
TAG=talkie-1930-v2-lr2e5-ckpt2000-mini-446-pass5
for i in 1 2 3 4 5; do
  BASE=/fast/rolmedo/swesmith/${TAG}-run${i}
  python3 -c "
import json, os, glob
paths = sorted(glob.glob('$BASE/*/preds.json'), key=os.path.getmtime, reverse=True)
merged = {}
for p in paths:
    try: d = json.load(open(p))
    except: continue
    for inst, rec in d.items():
        if inst in merged: continue
        if rec.get('model_patch','').startswith('diff --git'):
            merged[inst] = rec
print(f'run${i}: {len(merged)} valid predictions')
json.dump(merged, open('$BASE/preds.merged.json','w'), indent=2)
"
done
```

Then submit one grade job per run (template: `grade_ckpt400.sub`, see
`EVAL_HANDOVER.md` step 5).

---

## 6. Compute pass@1

After all K grade reports are written, compute the **Chen et al. 2021
unbiased pass@1** estimator with `n_i = K` forced for every instance —
i.e. instances whose sub-job timed out before the workers reached them
count as failures (one of K samples). With uniform `n_i`, Chen reduces to
the global mean:

```
pass@1 = (1/N_inst) · Σ_i (c_i / K) = total_resolved_across_runs / (K · N_inst)
```

```bash
RUN_TAG=talkie-1930-v2-lr2e5-ckpt2000-pass5    # used in <RUN_ID>.json filename
DIR_TAG=talkie-1930-v2-lr2e5-ckpt2000-mini-446-pass5
N=446; K=5
python3 -c "
import json
total = 0
for i in range(1, $K + 1):
    rpt = json.load(open(f'/fast/rolmedo/swesmith/$DIR_TAG-run{i}/$RUN_TAG-run{i}.json'))
    total += rpt['resolved_instances']
print(f'pass@1 = {total} / ($K * $N) = {100*total/($K*$N):.2f}%')
"
```

Treating missing samples as failures is the conservative reading and is
directly comparable to the standard single-sample 21/446-style number.
The K-fold averaging tightens the variance of the estimate; with K=5 the
run-to-run std on this benchmark was ~0.7pp.

We do not report pass@k for k > 1 from this setup — its purpose is the
better pass@1 estimate.

---

## 7. Why a watcher (and the gotchas behind it)

**JOB_ID override.** `run_mini_subjob.sh` builds its `OUTPUT_DIR` as
`$BASE/job_$JOB_ID`, where `JOB_ID` is set by the `.sub` file's
`environment` line. **HTCondor overrides that `JOB_ID` at runtime** with
its own GlobalJobId-style identifier (`sched#<cluster>.<proc>#<schedd>`),
so the actual on-disk dir is `job_sched#<cluster>.0#<schedd>/`. The
standard pipeline is fine with this because the post-run merge globs
`$BASE/*/preds.json`, but it means a re-submitted sub-job lands in a
*fresh* dir — mini-extra's per-OUTPUT_DIR idempotency does NOT help across
resubmits, and the resubmitted sub-job re-runs all 23 instances. Final
correctness still holds because the merge step deduplicates by instance
id, but compute is wasted on the re-runs. Cap `MAX_RESUBMITS` accordingly.

**Phantom holds.** Sub-jobs frequently end with `JobStatus = Held` and
`HoldReasonCode 34` "memory limit" *even when peak usage is well under the
limit* (e.g. peak 2.8 GB vs 230 GB limit). These often fire AFTER the
script's work is already complete. The watcher handles both flavors:

- "phantom hold, work was done" → coverage matches expected → marked DONE,
  `condor_rm` to free the slot.
- "real hold, vLLM never loaded" → coverage 0/expected, elapsed < 1h30 →
  retry. `condor_rm` first to free the slot.

**The 1h30 rule.** Without it, sub-jobs that ran almost the full 2h before
the `periodic_remove` fires would get retried, even though the underlying
issue is "stuck instance" rather than "crashed sub-job". The retry
generally hits the same stuck instance again and times out. The
`LATE_THRESHOLD_SEC = 5400` cutoff stops these wasteful retries while
still catching genuine early-death cases (vLLM crash on startup, slot
mismatch, NVML errors).
