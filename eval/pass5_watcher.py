#!/usr/bin/env python3
"""Pass@5 watcher: re-submits sub-jobs that crashed *early* (< 1h30 elapsed).

Re-submit rule (per user spec):
  - if a cluster ended (exit / hold / removal) and elapsed wallclock < 5400s,
    treat as a real crash and resubmit
  - if elapsed >= 5400s, treat as the 2h-stuck-generation case (or close to it)
    and DO NOT resubmit (instances are stuck for a real reason)

Coverage check uses the per-cluster output dir
  $run_dir/job_sched#<cluster>.0#<schedd>/
and a slice is considered DONE when traj.json count >= slice size.

Held jobs that we decide to retry are condor_rm'd first to free the slot.

Caps re-submits per slice at MAX_RESUBMITS to avoid infinite loops.
"""

import json
import os
import re
import glob
import time
import subprocess
from datetime import datetime
from pathlib import Path

BASE = "/fast/rolmedo/swesmith"
RUN_DIRS = [f"{BASE}/talkie-web-v2-lr2e5-ckpt2000-mini-446-pass3-run{i}" for i in range(1, 4)]
SCHEDD_SUFFIX = "7941"
MAX_RESUBMITS = 3
CHECK_INTERVAL_SEC = 20 * 60   # 20 min
LATE_THRESHOLD_SEC = 90 * 60   # 1h30 — past this, don't retry
CONDOR_LOG_DIR = "/lustre/home/rolmedo/talkie-eval/swe-evals/condor_logs"
STATE_FILE = "/lustre/home/rolmedo/talkie-eval/swe-evals/pass3_watcher_state.json"
LOG_FILE = "/lustre/home/rolmedo/talkie-eval/swe-evals/pass3_watcher.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except Exception:
            return {}
    return {}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    json.dump(state, open(tmp, "w"), indent=2)
    os.rename(tmp, STATE_FILE)


def parse_sub_file(path):
    txt = open(path).read()
    m = re.search(r"JOB_ID=(\d+) START=(\d+) END=(\d+)", txt)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def get_active_jobs():
    """Return dict: cluster_id -> (run_dir, job_id, status)
    status: 1=idle, 2=running, 5=held."""
    try:
        out = subprocess.check_output(
            ["condor_q", os.environ.get("USER", ""), "-af", "ClusterId", "JobStatus", "Environment"],
            text=True, errors="ignore", stderr=subprocess.DEVNULL, timeout=60,
        )
    except Exception:
        return {}
    active = {}
    for line in out.splitlines():
        m_cl = re.match(r"^\s*(\d+)\s+(\d+)\s+", line)
        m_run = re.search(r"BASE_OUTPUT_DIR=(\S+pass\d+-run\d+)", line)
        m_id = re.search(r"JOB_ID=(\d+)", line)
        if m_cl and m_run and m_id:
            active[int(m_cl.group(1))] = (
                m_run.group(1), int(m_id.group(1)), int(m_cl.group(2))
            )
    return active


def cluster_dir(run_dir, cluster):
    return Path(run_dir) / f"job_sched#{cluster}.0#{SCHEDD_SUFFIX}"


def coverage_for_cluster(run_dir, cluster):
    d = cluster_dir(run_dir, cluster)
    if not d.exists():
        return 0
    return len(list(d.glob("*/*.traj.json")))


def cluster_elapsed_seconds(cluster):
    """Parse condor user log to find seconds between exec-start and the last event.
    Returns None if the log is missing; 0 if the job never started executing."""
    log_path = f"{CONDOR_LOG_DIR}/sub.{cluster}.log"
    if not os.path.exists(log_path):
        return None
    try:
        text = open(log_path).read()
    except Exception:
        return None
    exec_m = re.search(
        r"^001 \(\S+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ",
        text, flags=re.MULTILINE,
    )
    if not exec_m:
        return 0
    last_ts = re.findall(
        r"^\d{3} \(\S+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ",
        text, flags=re.MULTILINE,
    )
    if not last_ts:
        return 0
    try:
        start = datetime.strptime(exec_m.group(1), "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(last_ts[-1], "%Y-%m-%d %H:%M:%S")
        return max(0, (end - start).total_seconds())
    except Exception:
        return None


def total_unique_coverage(run_dir):
    insts = set()
    for traj in Path(run_dir).glob("job_*/*/*.traj.json"):
        insts.add(traj.parent.name)
    return len(insts)


def submit(sub_file):
    out = subprocess.check_output(
        ["condor_submit_bid", "51", sub_file],
        text=True, stderr=subprocess.STDOUT, timeout=60,
    )
    m = re.search(r"submitted to cluster (\d+)", out)
    if not m:
        raise RuntimeError(f"could not parse cluster id from: {out!r}")
    return int(m.group(1))


def condor_rm(cluster):
    try:
        subprocess.run(
            ["condor_rm", str(cluster)],
            check=False, capture_output=True, text=True, timeout=30,
        )
        log(f"condor_rm cluster={cluster}")
    except Exception as e:
        log(f"condor_rm cluster={cluster} failed: {e}")


def seed_state_from_queue(state, active):
    """Register currently-queued clusters in state on first run."""
    for cluster, (run_dir, job_id, _status) in active.items():
        key = f"{run_dir}::{job_id}"
        rec = state.setdefault(key, {"clusters": [], "resubmits": 0, "status": "inflight"})
        if cluster not in rec["clusters"]:
            rec["clusters"].append(cluster)
            log(f"seed {key} cluster={cluster}")
    save_state(state)


def cycle(state):
    active = get_active_jobs()
    # (run_dir, job_id) -> (cluster, status) for clusters with status 1 or 2
    # (jobs that are still doing work; held clusters are treated as terminal)
    inflight_keys = {}
    for cluster, (rd, jid, st) in active.items():
        if st in (1, 2):
            inflight_keys[(rd, jid)] = (cluster, st)
    # Held clusters mapped per (rd, jid) so we can rm them when retrying
    held_clusters = {}
    for cluster, (rd, jid, st) in active.items():
        if st == 5:
            held_clusters.setdefault((rd, jid), []).append(cluster)

    all_done = True
    per_run = {}

    for run_dir in RUN_DIRS:
        sub_files = sorted(
            glob.glob(f"{run_dir}/job_*.sub"),
            key=lambda p: int(re.search(r"job_(\d+)", p).group(1)),
        )
        done = inflight = needs = late = gave_up = 0

        for sub_file in sub_files:
            parsed = parse_sub_file(sub_file)
            if not parsed:
                continue
            job_id, start, end = parsed
            expected = end - start
            key = f"{run_dir}::{job_id}"
            rec = state.setdefault(
                key, {"clusters": [], "resubmits": 0, "status": "pending"}
            )

            # Currently running or idle in queue -> wait.
            if (run_dir, job_id) in inflight_keys:
                rec["status"] = "inflight"
                inflight += 1
                all_done = False
                continue

            # Determine the "latest attempt" cluster — prefer held in queue if any,
            # else the highest cluster id we've recorded.
            latest_cluster = None
            if (run_dir, job_id) in held_clusters:
                latest_cluster = max(held_clusters[(run_dir, job_id)])
                cluster_state = "held"
            else:
                clusters = rec.get("clusters", [])
                latest_cluster = max(clusters) if clusters else None
                cluster_state = "ended"

            cov = (
                coverage_for_cluster(run_dir, latest_cluster)
                if latest_cluster is not None else 0
            )
            elapsed = (
                cluster_elapsed_seconds(latest_cluster)
                if latest_cluster is not None else None
            )

            # Already covered -> done. If currently held, also rm it (frees slot).
            if cov >= expected:
                if cluster_state == "held":
                    condor_rm(latest_cluster)
                if rec.get("status") != "done":
                    rec.update(
                        status="done", coverage=cov,
                        last_elapsed=elapsed,
                    )
                    save_state(state)
                    log(f"DONE {key} cov={cov}/{expected} elapsed={elapsed}")
                done += 1
                continue

            # Coverage < expected. Decide retry vs accept-as-late-failure.
            if elapsed is not None and elapsed >= LATE_THRESHOLD_SEC:
                # Ran long before dying — treat as legit timeout, no retry.
                if cluster_state == "held":
                    condor_rm(latest_cluster)   # free slot
                if rec.get("status") != "ended_late":
                    rec.update(
                        status="ended_late", coverage=cov,
                        last_elapsed=elapsed,
                    )
                    save_state(state)
                    log(f"ENDED_LATE {key} cov={cov}/{expected} "
                        f"elapsed={elapsed:.0f}s — no retry")
                late += 1
                continue

            attempts = rec.get("resubmits", 0)
            if attempts >= MAX_RESUBMITS:
                if cluster_state == "held":
                    condor_rm(latest_cluster)
                if rec.get("status") != "gave_up":
                    rec.update(
                        status="gave_up", coverage=cov,
                        last_elapsed=elapsed,
                    )
                    save_state(state)
                    log(f"GIVE_UP {key} cov={cov}/{expected} "
                        f"after {attempts} resubmits, last_elapsed={elapsed}")
                gave_up += 1
                continue

            # Retry: condor_rm a held cluster first, then resubmit.
            if cluster_state == "held":
                condor_rm(latest_cluster)
            try:
                new_cluster = submit(sub_file)
                rec["clusters"].append(new_cluster)
                rec["resubmits"] = attempts + 1
                rec["status"] = "resubmitting"
                rec["coverage"] = cov
                rec["last_elapsed"] = elapsed
                save_state(state)
                log(f"RESUBMIT {key} attempt={attempts + 1} cov={cov}/{expected} "
                    f"last_elapsed={elapsed} new_cluster={new_cluster}")
                needs += 1
                all_done = False
            except Exception as e:
                log(f"RESUBMIT_FAILED {key}: {e}")
                needs += 1
                all_done = False

        per_run[run_dir] = {
            "done": done, "inflight": inflight, "resubmitted": needs,
            "late": late, "gave_up": gave_up,
            "unique_cov": total_unique_coverage(run_dir),
        }

    for rd, s in per_run.items():
        tag = rd.rsplit("-", 1)[-1]
        log(f"summary {tag}: done={s['done']} inflight={s['inflight']} "
            f"resubmitted={s['resubmitted']} late={s['late']} "
            f"gave_up={s['gave_up']} unique_inst_cov={s['unique_cov']}/446")

    return all_done


def main():
    log(f"Watcher start. {len(RUN_DIRS)} runs, MAX_RESUBMITS={MAX_RESUBMITS}, "
        f"CHECK_INTERVAL_SEC={CHECK_INTERVAL_SEC}, "
        f"LATE_THRESHOLD_SEC={LATE_THRESHOLD_SEC}, SCHEDD={SCHEDD_SUFFIX}")
    state = load_state()
    seed_state_from_queue(state, get_active_jobs())

    while True:
        try:
            done = cycle(state)
        except Exception as e:
            log(f"cycle exception: {type(e).__name__}: {e}")
            done = False
        if done:
            log("All sub-jobs done, ended_late, or gave_up. Watcher exiting.")
            break
        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
