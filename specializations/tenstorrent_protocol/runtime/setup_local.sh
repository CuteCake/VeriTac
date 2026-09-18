#!/usr/bin/env bash
# Reproduce the local Mac Studio Linux/ARM64 simulator install.
# Docker Desktop must already be running. No host packages, privileged mode,
# device passthrough, remote execution, or existing-container restart.
set -euo pipefail
name="${1:-veritac-tt-sim}"
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid container name' >&2; exit 2; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
if docker container inspect "$name" >/dev/null 2>&1; then
  echo "Container $name already exists; preserving it. Use a new name to reproduce installation." >&2
  exit 2
fi
image='ubuntu@sha256:b3cc40b72b93588182b5410f723c7aaf142363311c2aa993d8a453ddcbb3ae15'
volume="${name}-data"
docker volume create "$volume" >/dev/null
docker run -d --name "$name" --platform linux/arm64 --cpus 4 --memory 3g \
  --label org.veritac.component=tenstorrent-simulator \
  --mount "type=volume,src=$volume,dst=/opt/tt" \
  --mount "type=bind,src=$repo_root,dst=/workspace,readonly" \
  "$image" sleep infinity
# Exact apt package versions are recorded below; package mirrors themselves
# are not frozen. Upstream source/submodule revisions and patches are pinned.
docker exec "$name" bash -lc '
set -euo pipefail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
 ca-certificates git build-essential g++-14 clang-18 lld-18 cmake ninja-build \
 pkg-config libssl-dev python3-dev python3-pip python3-venv libhwloc-dev \
 libnuma-dev libtbb-dev libcapstone-dev libatomic1 curl wget xz-utils xxd \
 openmpi-bin libopenmpi-dev nlohmann-json3-dev
mkdir -p /opt/tt/tt-metal /opt/tt/sim
cd /opt/tt/tt-metal
git init
git remote add origin https://github.com/tenstorrent/tt-metal.git
git fetch --depth 1 origin ad232e1cd799adf53841774805bc73fc5605b634
git checkout --detach FETCH_HEAD
git submodule update --init --recursive --depth 1
git apply /workspace/specializations/tenstorrent_protocol/runtime/patches/metal-arm64-profiler.patch
git apply /workspace/specializations/tenstorrent_protocol/runtime/patches/metal-mpi-c-bindings.patch
curl -fL https://github.com/tenstorrent/ttsim/releases/download/v1.10.8/libttsim_bh_aarch64.so -o /opt/tt/sim/libttsim_bh.so
echo "0c3a5af653e7feec018f6d1ce3bba116967a951c150fcaa03ad6521e8ec88065  /opt/tt/sim/libttsim_bh.so" | sha256sum -c -
cp tt_metal/soc_descriptors/blackhole_140_arch.yaml /opt/tt/sim/soc_descriptor.yaml
cmake -S . -B /opt/tt/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
 -DWITH_PYTHON_BINDINGS=OFF -DENABLE_DISTRIBUTED=ON -DENABLE_TRACY=OFF \
 -DTT_UNITY_BUILDS=OFF -DBUILD_PROGRAMMING_EXAMPLES=ON \
 -DTT_METAL_BUILD_TESTS=OFF -DTTNN_BUILD_TESTS=OFF
cmake --build /opt/tt/build --target metal_example_add_2_integers_in_riscv generate_rank_bindings fmt --parallel 1
# An empty optional cache is intentional: firmware is compiled by the tested JIT.
mkdir -p tt_metal/pre-compiled
for component in metalium-runtime metalium-dev umd-runtime umd-dev fmt-core enchantum spdlog-dev tt-logger-dev tracy; do
 cmake --install /opt/tt/build --prefix /opt/tt/install --component "$component"
done
export TT_METAL_HOME=/opt/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=/opt/tt/tt-metal
export TT_METAL_SIMULATOR=/opt/tt/sim/libttsim_bh.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
timeout 240 /opt/tt/build/programming_examples/metal_example_add_2_integers_in_riscv
dpkg-query -W > /opt/tt/installed-packages.txt
git diff > /opt/tt/metal-local.patch
git submodule status --recursive > /opt/tt/submodules.txt
sha256sum /opt/tt/sim/libttsim_bh.so /opt/tt/sim/soc_descriptor.yaml /opt/tt/build/tt_metal/libtt_metal.so > /opt/tt/installed-artifacts.sha256
'
printf 'Local simulator ready in container %s; persistent volume %s.\n' "$name" "$volume"
