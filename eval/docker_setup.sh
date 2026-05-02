#!/bin/bash
# dockerd-rootless.sh requires HOME and USER to be set; under HTCondor
# the env may strip them, so set them explicitly.
export HOME="${HOME:-/lustre/home/$USER}"
export USER="${USER:-$(id -un)}"
export XDG_RUNTIME_DIR=/tmp
export PATH="$HOME/bin:/sbin:/usr/sbin:$PATH"
dockerd-rootless.sh &

# Wait for Docker daemon to be ready
echo "Waiting for Docker daemon to be ready..."
timeout=120
elapsed=0

while [ $elapsed -lt $timeout ]; do
    if docker info >/dev/null 2>&1; then
        echo "Docker daemon is ready!"
        exit 0
    fi
    echo "Docker not ready yet, waiting..."
    sleep 2
    elapsed=$((elapsed + 2))
done

echo "ERROR: Docker daemon did not become ready within $timeout seconds"
exit 1
