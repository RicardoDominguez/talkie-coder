# SWE-bench eval handover (talkie checkpoint → resolved %)

End-to-end recipe for evaluating a new talkie SFT checkpoint on SWE-Bench-Verified-Working-Harbor (446 instances, agentic eval via mini-swe-agent + rootless docker, served by vLLM with the transformers backend).

Scripts in this folder reference paths under `/lustre/home/rolmedo/` and `/fast/rolmedo/` that are the author's cluster paths — substitute your own. The vLLM-compatible refactor of `modeling_talkie.py` and `repackage_for_vllm.py` live in `../sft/`; the agent/harness wrappers and submit files live alongside this README.

---

## 0. One-time prerequisites (already done; check if still true)

| Component | Path | Notes |
|---|---|---|
| vLLM 0.19.0 venv | `/home/rolmedo/vllm019/` | requires `module load cuda/12.4` |
| sweagent venv | `/home/rolmedo/swa/` | required by `run_grade.py` (swebench harness) |
| mini-swe-agent venv | `/home/rolmedo/miniswa/` | `mini-extra swebench` lives here |
| mini-swe-agent source (editable) | `/lustre/home/rolmedo/mini-swe-agent/` | local edits live here |
| swerex source (editable) | `/lustre/home/rolmedo/swe-rex/` | unchanged |
| LiteLLM model registry | `/home/rolmedo/mini-swe-agent/model_prices_and_context_window.json` | new checkpoints must be registered here, see step 2 |
| Pre-pulled SWE-bench harbor tarballs | `/fast/rolmedo/swesmith/docker_tarballs/` | mini-swe-agent loads from here, no network pulls |
| Refactored modeling | `../sft/modeling_talkie.py` | source of truth — copy into model dir after each repackage |

### Local mods to mini-swe-agent (at `/lustre/home/rolmedo/mini-swe-agent/src/minisweagent/`)

- `agents/default.py:65` — tab/space fix to `render_template` (was a `TabError` blocker)
- `models/litellm_model.py:75-83` — cost calculation failures are downgraded to a warning + cost=0 instead of raising (since our local-served model isn't in litellm's price list)
- `environments/docker.py`:
  - `pull_timeout: 600` (was 120) — `docker load` of 5 GB tarballs from /fast under contention can exceed 120 s
  - `_pull_image()` wraps `docker load` in `flock /tmp/mswea_docker_load.lock` so concurrent loads serialize and each gets full bandwidth
  - `cleanup()` is **synchronous** (`subprocess.run` with timeout, not `Popen(...) &`) and now also runs `docker rmi -f <image>` per instance, so /tmp doesn't accumulate extracted layers across many instances
- `run/extra/swebench.py:149-183` — when an instance ends in any non-`Submitted` state (LimitsExceeded, ContextWindowExceededError, TimeoutExpired, FormatError, …) and the docker container is still alive, run one final `git add -A && git diff --cached` and use that as the patch. Salvages WIP edits that would otherwise be discarded.

If those edits are gone (someone reverted, fresh checkout), re-apply them or this whole pipeline will misbehave.

---

## 1. Repackage the checkpoint for vLLM

vLLM's transformers backend uses `AutoModel.from_config(...)` and expects `lm_head.weight` (not the bare `nn.Parameter` named `lm_head` that the talkie modeling saves). It also doesn't know about talkie's `lm_head_gain` scalar. Both need a one-time bake before serving.

```bash
SRC=/fast/rolmedo/swe-models/talkie-web-13b-base-swe-12h-v2/checkpoint-1000   # your checkpoint
DST=/fast/rolmedo/swe-models/talkie-web-12h-v2-ckpt1000-vllm                  # repackaged dir

python ../sft/repackage_for_vllm.py --src "$SRC" --dst "$DST"
```

What it does:
- loads `model.safetensors`
- multiplies `lm_head_gain.w_g` (a scalar) into the `lm_head` rows once and drops the gain tensor
- renames the result `lm_head.weight`
- stages the safetensors on local `/tmp` then `dd oflag=direct bs=64M` to `/fast` (~1 GB/s vs ~20 MB/s direct safetensors write)
- copies the rest of the dir (config.json, tokenizer, chat template, configuration_talkie.py, modeling_talkie.py)

After repackage, sync the refactored modeling and patch the configs:

```bash
# (a) Refactored modeling — vLLM needs TalkieModel as a PreTrainedModel + ALL_ATTENTION_FUNCTIONS dispatch
cp ../sft/modeling_talkie.py "$DST/modeling_talkie.py"
rm -rf /tmp/modules/transformers_modules/   # bust trust_remote_code cache

# (b) Add AutoModel to auto_map so vLLM's AutoModel.from_config finds TalkieModel
python3 -c "
import json, sys
p=sys.argv[1]
c=json.load(open(p))
c['auto_map']['AutoModel']='modeling_talkie.TalkieModel'
json.dump(c, open(p,'w'), indent=2)
" "$DST/config.json"

# (c) Set <|end|> as the EOS so vLLM stops on the trained turn delimiter (default eos_token is <|endoftext|>, which the model never emits)
python3 -c "
import json, sys
p=sys.argv[1]
c=json.load(open(p)); c['eos_token']='<|end|>'
json.dump(c, open(p,'w'), indent=2)
" "$DST/tokenizer_config.json"

cat > "$DST/generation_config.json" <<'EOF'
{
  "_from_model_config": true,
  "eos_token_id": [65536, 65535],
  "pad_token_id": 65535,
  "transformers_version": "4.57.3"
}
EOF
```

Validate by hitting it with HF generate before going to vLLM:

```bash
cd /lustre/home/rolmedo/talkie-eval && source /home/rolmedo/tflatest/bin/activate && export HF_HOME=/tmp
MODEL_DIR="$DST" python3 -c "
import os, torch
os.environ['HF_HOME']='/tmp'
from transformers import AutoTokenizer, AutoModelForCausalLM
mdir = os.environ['MODEL_DIR']
tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(mdir, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
end_id = tok.convert_tokens_to_ids('<|end|>')
sys_msg = (
    'You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.\n'
    'Your response must contain exactly ONE bash code block with ONE command.\n\nInclude a THOUGHT section.'
)
chat = tok.apply_chat_template(
    [{'role':'system','content':sys_msg},
     {'role':'user','content':'<pr_description>foo() in src/utils.py returns None on valid YAML.</pr_description><instructions>Fix the bug.</instructions>'}],
    tokenize=False, add_generation_prompt=True)
ids = tok([chat], return_tensors='pt').input_ids.cuda()
out = m.generate(input_ids=ids, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=[end_id, tok.eos_token_id])
new = out[0, ids.shape[1]:]
print('emit_<|end|>:', end_id in new.tolist(), 'len:', len(new))
print(repr(tok.decode(new, skip_special_tokens=False)))
"
```

You want: `emit_<|end|>: True`, output starts with `THOUGHT: ...` and contains a ` ```bash` block. If not, the model is undertrained — don't bother with the eval, fix training first. Refer to `../sft/CHAT_TOKEN_COLLAPSE.md` for the most common failure mode (chat-token rows decayed by weight decay → no EOS).

---

## 2. Register the checkpoint in the LiteLLM model registry

mini-swe-agent uses LiteLLM, which barfs on cost calculation if the model name isn't registered. Add an entry:

```bash
python3 -c "
import json
path = '/home/rolmedo/mini-swe-agent/model_prices_and_context_window.json'
key = 'hosted_vllm//fast/rolmedo/swe-models/talkie-web-12h-v2-ckpt1000-vllm'   # full vLLM-style path
data = json.load(open(path))
if key not in data:
    data[key] = {
        'input_cost_per_token': 0,
        'output_cost_per_token': 0,
        'max_input_tokens': 32000,
        'litellm_provider': 'together_ai',
        'supports_function_calling': True,
        'supports_parallel_function_calling': True,
        'mode': 'chat',
        'supports_tool_choice': True,
        'source': 'talkie SFT checkpoint, repackaged for vLLM',
    }
    json.dump(data, open(path,'w'), indent=4)
print('registered' if key not in data else 'already registered')
"
```

(With our `litellm_model.py` patch this is technically optional — cost calc just warns and continues. Still nice to add for cleanliness.)

---

## 3. Run the agent eval

The driver is `swe-evals/launch_parallel_eval.sh`, which writes per-sub-job condor submit files and submits them with a 1-second stagger between submissions.

```bash
cd /path/to/talkie-public/eval

# 100-instance eval, fan out to 20 condor sub-jobs of 5 instances each
./launch_parallel_eval.sh \
  /fast/rolmedo/swe-models/talkie-web-12h-v2-ckpt1000-vllm \
  /fast/rolmedo/swesmith/talkie-web-v2-ckpt1000-mini-100 \
  20 100

# Full 446-instance eval, 40 condor sub-jobs of ~12 each
./launch_parallel_eval.sh \
  /fast/rolmedo/swe-models/talkie-web-12h-v2-ckpt1000-vllm \
  /fast/rolmedo/swesmith/talkie-web-v2-ckpt1000-mini-446 \
  40 446
```

Each sub-job runs `run_mini_subjob.sh` → `run_mini_eval.sh`, which:
1. starts rootless docker (`docker_setup.sh`)
2. spins up a vLLM server with `--model-impl transformers --trust-remote-code --max-model-len 32768 --dtype bfloat16` (no fp8 — it's broken on talkie, see "Pitfalls")
3. waits for vLLM ready
4. fills in the model section of the eval config via `transfer_config.sh` (api_base + model name + sampling: `temperature=0.7`, `max_tokens=4096`)
5. runs `mini-extra swebench --slice $START:$END --workers 5 ...`
6. tears down vLLM and the diagnostic loop
7. **drains tmpfs** (rm -rf /tmp/docker /tmp/vllm_cache /tmp/triton /tmp/torchinductor /tmp/modules) before exit so condor's exit-time cgroup poll sees a clean slate

Per-job resources: 1 GPU (sm_9.0 H100 or sm_10.0 B200), 8 CPUs, 256 GB RAM, 300 GB disk, `+BypassLXCfs = true`. Submission bid is **51** (changed from 100; lower bid is enough on this cluster and reduces priority cost).

Requirements include `(TARGET.UtsnameNodename != "i206") && (TARGET.UtsnameNodename != "g105")` — both nodes were observed to advertise small dynamic-slot cgroup limits (~28 GB) that kill vLLM during model load even when `request_memory=256GB`. If new bad slots appear, add them to the exclusion list.

### Sampling configuration (changed from greedy)

`run_mini_eval.sh` exports `TEMPERATURE=0.7 MAX_TOKENS=4096` to `transfer_config.sh`, which writes them into the agent yaml's `model_kwargs`. Why these values:
- **Greedy decoding (`temperature=0`) hits catastrophic single-token loops** (e.g. `sympsympsymp...` for 60 KB) on roughly half of trajectories. The autoregressive trough locks in once the model picks a low-entropy attractor.
- **`temperature=0.7`** breaks those attractors stochastically. We tried temp 0.7 + repetition_penalty 1.1, but rep_penalty has bad side effects (penalizes whitespace/keywords/submission-marker tokens that legitimately repeat), so we removed it. Temperature alone is sufficient.
- **`max_tokens=4096`** caps blast radius on each turn. Even if a degenerate generation occurs, it can't burn the full 32 K context. Some long heredoc edits get truncated (~3% of empty submits) — bumping to 8192 would recover them at modest cost.
- For the `transfer_config.sh` extension that supports both env vars, see the script comments.

### Notes
- `NUM_WORKERS=5` per sub-job. With temp 0.7 + max_tokens 4096, KV-cache pressure is lower per turn than greedy long-tail generation, so 5 fits without significant preempt thrash. Going to 16 is still bad (oversubscribes the engine). The handover originally said 3 workers; 5 was empirically validated and gives ~1.5× throughput.
- If a sub-job is held with HoldReasonCode 34 ("memory limit") **and the .out shows `[run_mini_eval] mini-extra swebench exit=0`**, the work is already done — this is a "Held = work-done" phantom caused by condor's exit-time cgroup poll catching residual tmpfs before our drain finishes. Trajectory files are on disk; `condor_rm` the held jobs at end-of-eval. The peak-memory poll happens during the run, not after, so even our drain doesn't always satisfy it.
- If a sub-job is held with HoldReasonCode 34 and **no `exit=0` in .out**, that's a real OOM — usually slot-mismatch (g105 / i206 type). Add the offending node to the exclusion list and resubmit.
- mini-extra is **idempotent within an output dir**. Killing and re-launching with the same OUTPUT_DIR will skip already-resolved instances. Use this freely to recover from holds.
- Some sub-jobs get stuck on **long step-limit-bound trajectories** (each instance running 5–20 min instead of <1 min). The diagnostic loop keeps logging, but throughput tanks. After ~75% coverage, surgical-`condor_rm` of stuck R sub-jobs is reasonable; partial-grade what you have.

While running, monitor:

```bash
condor_q $USER
ls /fast/rolmedo/swesmith/talkie-web-v2-ckpt1000-mini-100/job_*/  # per-sub-job output dirs
find /fast/rolmedo/swesmith/talkie-web-v2-ckpt1000-mini-100 -name '*.traj.json' | wc -l
```

Per-instance trajectory at `<OUTPUT>/job_N/<instance_id>/<instance_id>.traj.json`. The `info.exit_status` is `Submitted` / `LimitsExceeded` / `ContextWindowExceededError` / `TimeoutExpired` / `Submitted+SalvagedWIP`. The `info.submission` field holds the patch (or salvaged WIP).

---

## 4. Build the merged preds.json for grading

mini-extra writes one preds.json per sub-job. Merge them, keeping only entries with a real `diff --git` patch:

```bash
BASE=/fast/rolmedo/swesmith/talkie-web-v2-ckpt1000-mini-100
python3 -c "
import json, os, glob
paths = sorted(glob.glob('$BASE/*/preds.json'), key=os.path.getmtime, reverse=True)
merged = {}
for p in paths:
    try: d = json.load(open(p))
    except: continue
    for inst, rec in d.items():
        if inst in merged: continue          # latest sub-job wins
        if rec.get('model_patch','').startswith('diff --git'):
            merged[inst] = rec
print(f'{len(merged)} valid predictions')
json.dump(merged, open('$BASE/preds.merged.json','w'), indent=2)
"
```

---

## 5. Run the swebench grading harness

```bash
condor_submit_bid 51 - <<EOF
executable = /lustre/home/rolmedo/talkie-eval/swe-evals/run_grade_wrapper.sh
environment = "PREDS_FILE=$BASE/preds.merged.json OUTPUT_DIR=$BASE RUN_ID=talkie-v2-ckpt1000 MAX_WORKERS=12"
request_cpus   = 16
request_gpus   = 0
request_memory = 200GB
request_disk   = 300GB
requirements = (TARGET.UtsnameNodename != "i206") && (TARGET.UtsnameNodename != "g105")
+BypassLXCfs = true
output = /lustre/home/rolmedo/talkie-eval/swe-evals/condor_logs/grade.\$(Cluster).\$(Process).out
error  = /lustre/home/rolmedo/talkie-eval/swe-evals/condor_logs/grade.\$(Cluster).\$(Process).err
log    = /lustre/home/rolmedo/talkie-eval/swe-evals/condor_logs/grade.\$(Cluster).log
queue
EOF
```

Or use the existing static submit file `grade_ckpt400.sub` as a template.

`run_grade_wrapper.sh` calls `swe-evals/run_grade.py` (vendored from `/home/rolmedo/swe-evals/run_grade.py`), which:
- spins up rootless docker
- runs `python -m swebench.harness.run_evaluation` against `ricdomolm/SWE-bench_Verified-Working-Harbor`
- writes `<OUTPUT_DIR>/<RUN_ID>.json` with the per-instance pass/fail breakdown

The condor job often ends in status 5 (Held) right *after* the harness writes the report — that's a memory-cgroup hold from the long-running grading process, not a failure. Look for `[grade] done; report at ...` in the .out file before assuming failure.

---

## 6. Read the result

```bash
python3 -c "
import json
r = json.load(open('$BASE/talkie-v2-ckpt1000.json'))
for k in ['total_instances','submitted_instances','completed_instances',
          'resolved_instances','unresolved_instances',
          'empty_patch_instances','error_instances']:
    if k in r: print(f'  {k}: {r[k]}')
print('  resolved_ids:', r.get('resolved_ids', []))
"
```

The headline is `resolved_instances / total_instances` (446 for the full run, 100 if you sliced). For comparison:
- ckpt-200 (10% trained): 0 / 32 valid patches resolved
- ckpt-400 (20% trained): 1 / 49 valid patches resolved (django__django-11163, model_to_dict None-vs-empty-list bug — legitimate one-line fix)

---

## 7. Pitfalls and known issues

- **fp8 weights blow up output**: `vllm serve --quantization fp8` produces token salad on talkie. Don't use it. The careful per-channel custom fp8 quant works but isn't wired into vLLM serving (see `sft/test_fp8_careful_inflight.py` for the offline test).
- **fp8 KV cache hurts SWE-bench unpredictably**: `--kv-cache-dtype fp8` gave a 16% → 12% hit on GSM8K but caused divergent agent trajectories on SWE-bench. Stick with bf16 KV.
- **`--max-model-len 65536` ran us OOM**: KV cache allocation at engine-init for 64 K context exceeded H100's 80 GB minus the 26 GB bf16 model. 76 / 76 sub-jobs OOM'd in the first attempt. Stay at 32768.
- **Concurrency / KV cache thrash**: at `--max-model-len 32768` greedy decoding, KV cache fits ~2-3 sequences per H100; preempt thrash above ~3 workers. With `temperature=0.7` + `max_tokens=4096`, per-turn generation is shorter, so 5 workers fits without thrash. 16 still bad.
- **Greedy decoding is broken on talkie SFT**: `temperature=0` triggers catastrophic single-token-pattern repetition (e.g. `sympsymp...` for 60 KB) on ~50% of trajectories, leading to format errors and empty submissions. Always run with `temperature ≥ 0.5`. We use 0.7 + `max_tokens=4096` as a per-turn cap.
- **`repetition_penalty` is risky on long agentic dialogues**: 1.1 measurably degrades the submission marker (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` token logits get penalized after first emission), code-syntax (whitespace, `def`, `return`), and path-references. Avoid. Temperature alone breaks the autoregressive trough.
- **`top_p` / `top_k` don't help with repetition loops**: those are concentration mechanisms. Loops occur when the next-token distribution is already concentrated on one token; the alternatives are below threshold. Only entropy (temperature) helps.
- **Rootless docker /tmp overflow**: each SWE-bench harbor image extracts to ~5 GB in `/tmp/docker/...`. With many instances per sub-job that adds up fast. The synchronous `docker rmi` after each instance is what keeps this bounded.
- **`docker load` timeout**: 5 GB tarball loads under contention can exceed 120 s. The flock + 600 s timeout in `environments/docker.py:_pull_image` solve this.
- **Empty-Submitted from `ContextWindowExceededError`**: Big shell observations (`find /`, verbose tracebacks) inflate context past 32K after just a few turns. mini-extra's `Qwen3Model.query` catches the error and returns `OUT_OF_CONTEXT_RESPONSE` ("I've run out of time…"), which submits an empty diff. The salvage hook in `run/extra/swebench.py` recovers WIP edits in this case; otherwise consider lowering `max_observation_length` in the agent config.
- **Most "Submitted with diff" patches violate the rules**: the model creates `test_*.py` reproduction files and edits `pyproject.toml` very often. The instance template says "DO NOT MODIFY" tests/configs. Currently ~60% of training-data assistant edits are test-file creations rather than source edits, so the model imitates that. A future improvement is filtering the SFT data to resolved trajectories only.
- **Empty-Submitted with no apparent error**: at `temperature=0.7`, the dominant remaining failure is **hallucinated completion** — the model writes a final THOUGHT claiming "I have successfully addressed the issue" without ever issuing an edit command, then submits with the marker. `git diff --cached` returns empty. ~15% of trajectories hit this. Distinct from greedy repetition; comes from training-data trajectories that end on a verification/syntax-check rather than the edit itself.
- **Action-level repetition (sera-lite specific)**: the sera-lite training variant produces a different failure mode at ckpt-600: model emits well-formed bash commands but **fails to update its world model from observations**. E.g., gets back a clean `ls -la /testbed` directory listing showing `django/`, claims "files are not in the current working directory", and reissues the same `ls`. Loops until context fills. Different from greedy single-token loops; this is reasoning-level. Probably a training-data issue — the variant didn't give the model practice at parsing shell observations and updating beliefs.
- **Train/test prompt parity is fine** — system + instance + observation templates byte-match between training data and `localconfig_qwen3_train_aligned.yaml`. (The training data has `MODIFY: ... in ` with empty `{{working_dir}}`; eval has `... in /testbed`. Cosmetic.)
- **Klear-trained models (`runC`) need a separate eval pipeline**. They were trained on the `mini-swe-agent-plus` harness (different submission marker `MINI_SWE_AGENT_FINAL_OUTPUT`, different `<format_example>` tag, expects an `edit_via_str_replace` helper). Scoring them through this pipeline would just measure the salvage hook. See talkie issue #1.

---

## 8. File index

```
talkie-public/
├── sft/
│   ├── modeling_talkie.py                      refactored — source of truth for vLLM/HF
│   ├── repackage_for_vllm.py                   bake lm_head_gain + rename to lm_head.weight
│   ├── CHAT_TOKEN_COLLAPSE.md                  diagnostic for the chat-token weight-decay bug
│   └── CHAT_TOKEN_COLLAPSE_FIX.md              the wd-skip + reinit fix
└── eval/
    ├── README.md                               this file
    ├── localconfig_qwen3_train_aligned.yaml    mini-swe-agent eval config (matches training prompts)
    ├── transfer_config.sh                      injects model section into eval config (reads TEMPERATURE, MAX_TOKENS, REPETITION_PENALTY env vars)
    ├── docker_setup.sh                         rootless docker startup (HOME/USER set explicitly)
    ├── launch_parallel_eval.sh                 fan-out submit script (n_jobs × instances), bid 51, excludes i206 + g105
    ├── run_mini_eval.sh                        single-job wrapper: vLLM + mini-extra; sets temp 0.7 + max_tokens 4096; drains tmpfs at end
    ├── run_mini_subjob.sh                      sub-job entry point (sets OUTPUT_DIR + SLICE)
    ├── run_grade_wrapper.sh                    grade-job wrapper around run_grade.py
    ├── run_grade.py                            calls swebench.harness.run_evaluation against the harbor dataset
    └── grade_ckpt400.sub                       reference grade submit file

/fast/rolmedo/swe-models/<checkpoint>-vllm/    one repackaged dir per checkpoint to evaluate
/fast/rolmedo/swesmith/<output_dir>/job_N/     per-sub-job mini-extra output
/fast/rolmedo/swesmith/<output_dir>/preds.merged.json   merged predictions for grading
/fast/rolmedo/swesmith/<output_dir>/<run_id>.json       final harness report
```
