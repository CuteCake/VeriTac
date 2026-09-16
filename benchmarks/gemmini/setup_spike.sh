#!/usr/bin/env bash
# Task-local setup tested on Ubuntu 24.04 / aarch64. No sudo/system installs.
set -euo pipefail
root="${1:?usage: setup_spike.sh ABSOLUTE_BUILD_DIRECTORY}"
[[ "$root" = /* ]] || { echo 'Build directory must be absolute' >&2; exit 2; }
mkdir -p "$root"/{deps,prefix}
root="$(cd "$root" && pwd)"
prefix="$root/prefix"
arch="$(dpkg-architecture -qDEB_HOST_MULTIARCH)"
export PATH="$prefix/usr/bin:$prefix/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/usr/lib/$arch${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$root/deps"
apt-get download device-tree-compiler libfdt1 libfdt-dev \
  libboost1.83-dev libboost-regex1.83-dev libboost-regex1.83.0 \
  libboost-system1.83-dev libboost-system1.83.0 \
  gcc-13-riscv64-linux-gnu cpp-13-riscv64-linux-gnu \
  gcc-13-riscv64-linux-gnu-base binutils-riscv64-linux-gnu \
  libgcc-13-dev-riscv64-cross libgcc-s1-riscv64-cross \
  libc6-dev-riscv64-cross libc6-riscv64-cross linux-libc-dev-riscv64-cross
for package in ./*.deb; do dpkg-deb -x "$package" "$prefix"; done
sha256sum ./*.deb > packages.sha256

checkout() {
  local name="$1" url="$2" revision="$3"
  if [[ ! -d "$root/$name" ]]; then
    git init "$root/$name"
    git -C "$root/$name" remote add origin "$url"
    git -C "$root/$name" fetch --depth 1 origin "$revision"
    git -C "$root/$name" checkout --detach FETCH_HEAD
  fi
  [[ "$(git -C "$root/$name" rev-parse HEAD)" = "$revision" ]] || {
    echo "Unexpected revision in $root/$name; use a fresh directory" >&2; exit 1;
  }
}
checkout spike-src https://github.com/riscv-software-src/riscv-isa-sim.git 1e05ddac3a6c351bfc0aeed0cf3a68940e7200ab
checkout libgemmini https://github.com/ucb-bar/libgemmini.git 5b1254a7dc72f757442280e7f28ef30542c78533
checkout pk-src https://github.com/riscv-software-src/riscv-pk.git 9c61d29846d8521d9487a57739330f9682d5b542
checkout gemmini-rocc-tests https://github.com/ucb-bar/gemmini-rocc-tests.git 7c540b3adf1b86ad93d07f893abe3a73489b568e
git -C "$root/gemmini-rocc-tests" submodule update --init rocc-software

mkdir -p "$root/spike-build" "$root/pk-build"
cd "$root/spike-build"
../spike-src/configure --prefix="$prefix" --with-boost="$prefix/usr" \
  --with-boost-libdir="$prefix/usr/lib/$arch" > configure.log 2>&1
make -j"${GEMMINI_BUILD_JOBS:-4}" > build.log 2>&1
make install > install.log 2>&1
cd "$root/libgemmini"
make RISCV="$prefix" > build.log 2>&1
make install RISCV="$prefix" >> build.log 2>&1
cd "$root/pk-build"
../pk-src/configure --prefix="$prefix" --host=riscv64-linux-gnu \
  CC="riscv64-linux-gnu-gcc-13 --sysroot=$prefix" \
  --with-arch=rv64gc --with-abi=lp64d > configure.log 2>&1
make -j"${GEMMINI_BUILD_JOBS:-4}" > build.log 2>&1
make install > install.log 2>&1

cd "$root"
riscv64-linux-gnu-gcc-13 --sysroot="$prefix" -static -O2 -march=rv64gc -mabi=lp64d \
  -DBAREMETAL -I gemmini-rocc-tests gemmini-rocc-tests/bareMetalC/mvin_mvout.c \
  -lm -o mvin_mvout
timeout 60 spike --extension=gemmini pk-build/pk mvin_mvout
echo "Gemmini functional simulator ready in $root (not a timing model)."
