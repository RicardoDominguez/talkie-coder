import subprocess
import signal
import sys
import psutil
import shutil

def cleanup_process(process):
    """Clean up subprocess on exit."""
    if process and process.poll() is None:
        print("Terminating subprocess...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing subprocess...")
            process.kill()


def setup_docker_rootless():
    # Start docker rootless
    print("Starting Docker rootless...")
    docker_process = subprocess.Popen(
        ['./docker_setup.sh'], 
        # stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    # Update signal handler to clean up both processes
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, cleaning up...")
        cleanup_docker_process(docker_process)
        sys.exit(0)
    
    # Register updated signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for docker to be ready (the script will exit when ready or timeout)
    try:
        docker_process.wait()
        if docker_process.returncode != 0:
            raise RuntimeError(f"Docker setup failed with code {docker_process.returncode}")
    except subprocess.TimeoutExpired:
        cleanup_docker_process(docker_process)
        raise RuntimeError("Docker setup timed out")

def cleanup_docker_process(docker_process):
    """Clean up docker rootless process on exit."""
    if docker_process and docker_process.poll() is None:
        print("Terminating docker rootless process...")
        # Kill the dockerd-rootless.sh process and its children
        try:
            # Get the process group to kill all related processes
            parent = psutil.Process(docker_process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            
            # Wait for graceful shutdown
            try:
                docker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing docker processes...")
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--preds_file', required=True)
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--max_workers', required=True)
    parser.add_argument('--dataset', default='ricdomolm/SWE-bench_Verified-Working-Harbor')
    parser.add_argument('--split', default='test')
    parser.add_argument('--timeout', default=None)
    parser.add_argument('--keep_logs', action='store_true')  # otherwise, we log in /tmp such that it does not persist after the job is done
    args = parser.parse_args()

    preds_file = args.preds_file
    eval_max_workers = args.max_workers

    setup_docker_rootless()
    cmd = (
        f"DOCKER_HOST=unix:///tmp/docker.sock python -m swebench.harness.run_evaluation "
        f"--dataset_name {args.dataset} "
        f"--predictions_path {preds_file} "
        f"--max_workers {eval_max_workers} "
        f"--run_id {args.run_id} "
        f"--namespace harbor.is.localnet/swebench "
        f"--split {args.split} "
        # f"--instance_ids astropy__astropy-13236 "  # for testing, normally commented out
    )
    if args.timeout is not None:
        cmd += f" --timeout {args.timeout} "
    
    log_dir = args.output_dir if args.keep_logs else '/tmp/'
    subprocess.run(cmd, shell=True, check=True, cwd=log_dir)

    if log_dir != args.output_dir:
        # copy the final report
        shutil.copy(f"{log_dir}/{args.run_id}.json", f"{args.output_dir}/{args.run_id}.json")

    print('Done, all went fine')
