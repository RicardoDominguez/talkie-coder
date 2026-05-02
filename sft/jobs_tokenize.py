#!/usr/bin/env python3
"""HTCondor launcher for tokenizing the talkie SWE dataset.

Submits one HTCondor job per shard (n_jobs=250). Each job runs
`python tokenize_messages.py ...` with its own --job_id; tokenize_messages.py
shuffles the source under seed=0 then takes its 1/n_jobs slice, so the union
of all shards covers the whole input. Run from this directory:

    cd /home/rolmedo/talkie/sft && python jobs_tokenize.py

`executable: latest_trl.sh` is resolved relative to the submit directory; that
script activates the env and exec's `python tokenize_messages.py ...`. Both
must live alongside this file.
"""
from pathlib import Path
import htcondor

JOB_BID = 101


def launch_experiment_job(
        input_dataset,
        output_dataset,
        tokenizer_path,
        custom_chat_template=False,
        pack=False,
        pack_length=64000,
        drop_too_long=False,
        n_jobs=1,
        job_id=0,
        JOB_MEMORY=200,
        JOB_CPUS=8,
    ):
    CLUSTER_LOGS_SAVE_DIR = Path('/fast/rolmedo/logs/')
    cluster_job_log_name = str(CLUSTER_LOGS_SAVE_DIR / f"$(Cluster).$(Process)")

    executable = 'latest_trl.sh'

    arguments = [
        "python tokenize_messages.py",
        f"--input_dataset {input_dataset}",
        f"--output_dataset {output_dataset}",
        f"--tokenizer_path {tokenizer_path}",
        f"--n_jobs {n_jobs}",
        f"--job_id {job_id}",
        f"--pack_length {pack_length}",
    ]
    if custom_chat_template:
        arguments.append("--custom_chat_template")
    if pack:
        arguments.append("--pack")
    if drop_too_long:
        arguments.append("--drop_too_long")
    arguments = " ".join(arguments)

    job_settings = {
        "executable": f"{executable}",
        "arguments": f"{arguments}",
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "request_cpus": f"{JOB_CPUS}",
        "request_memory": f"{JOB_MEMORY}GB",
        "request_disk": f"{JOB_MEMORY}GB",
        "jobprio": f"{JOB_BID - 1000}",
    }

    black_nodes = ['g134', 'g125', 'g123', 'g164', 'i101']
    req = " && ".join(f"(TARGET.UtsnameNodename != \"{node}\")" for node in black_nodes)
    if req:
        job_settings["requirements"] = req

    submit = htcondor.Submit(str(htcondor.Submit(job_settings)))
    sched = htcondor.Schedd()
    with sched.transaction() as txn:
        cluster_id = submit.queue(txn)
        print(f"Launched job_id={job_id} cluster-ID={cluster_id}")


if __name__ == '__main__':
    import os

    compute_args = {
        'JOB_CPUS': 4,
        'JOB_MEMORY': 128,
    }

    # mini-coder-trajs MIX: verified (max 3/instance) + unverified rows from
    # non-covered instance_ids, length-targeted at +40% raw bytes so that
    # 2016 steps @ 524288 tok/step ~ 1 epoch. 68,206 rows / 4.0 GB JSONL.
    # Tokenized for both base & web tokenizers @ 64K max-length;
    # NOT pre-packed (trainer BFD-packs).
    input_dataset = '/fast/rolmedo/swesmith/datasets/mini-coder-trajs-mix-max3.jsonl'
    custom_chat_template = False  # use talkie's chat_template.jinja
    pack = False
    drop_too_long = False
    pack_length = 65536

    tokenizer_runs = [
        ('/fast/rolmedo/models/talkie-1930-13b-base/',
         '/fast/rolmedo/swesmith/datasets/talkie-1930-mini-coder-mix-max3-64k/'),
        ('/fast/rolmedo/models/talkie-web-13b-base/',
         '/fast/rolmedo/swesmith/datasets/talkie-web-mini-coder-mix-max3-64k/'),
    ]

    # Prior runs (commented for reference):
    # input_dataset = '/fast/rolmedo/swesmith/datasets/klear-mini-swe-agent-plus-66k.jsonl'
    # tokenizer_path = '/fast/rolmedo/models/talkie-1930-13b-base/'
    # output_dir = '/fast/rolmedo/swesmith/datasets/talkie-1930-klear-66k-64k/'
    # input_dataset = '/fast/rolmedo/swesmith/datasets/talkie-swe-100k.jsonl'
    # tokenizer_path = '/fast/rolmedo/models/talkie-web-13b-base/'
    # output_dir = '/fast/rolmedo/swesmith/datasets/talkie-web-swe-100k-64k/'

    n_jobs = 250
    for tokenizer_path, output_dir in tokenizer_runs:
        for job_id in range(n_jobs):
            output_dataset = f"{output_dir}/job_{job_id}/"
            if os.path.exists(output_dataset):
                continue
            launch_experiment_job(
                input_dataset,
                output_dataset,
                tokenizer_path,
                custom_chat_template=custom_chat_template,
                pack=pack,
                pack_length=pack_length,
                drop_too_long=drop_too_long,
                n_jobs=n_jobs,
                job_id=job_id,
                **compute_args,
            )
