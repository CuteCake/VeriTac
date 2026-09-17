"""Derive an isolated capacity-limited Gemmini functional model.

This changes the public simulator's parameter header, not physical hardware.
The compiler headers are changed identically; run_spike.py checks their hashes.
The toolchain and proxy kernel are shared read-only with the original build.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scratchpad-rows", type=int, default=32)
    p.add_argument("--accumulator-rows", type=int, default=16)
    args = p.parse_args()
    if args.scratchpad_rows < 32 or args.accumulator_rows < 16:
        p.error("this subset needs at least 32 scratchpad and 16 accumulator rows")
    src, dst = args.source_root.resolve(), args.output.resolve()
    dst.mkdir(parents=True, exist_ok=False)
    prefix = dst / "prefix"
    prefix.mkdir()
    for name in ("bin", "usr", "include"):
        (prefix / name).symlink_to(src / "prefix" / name, target_is_directory=True)
    (prefix / "lib").mkdir()
    for artifact in (src / "prefix/lib").iterdir():
        if artifact.name != "libgemmini.so":
            (prefix / "lib" / artifact.name).symlink_to(artifact, target_is_directory=artifact.is_dir())
    (dst / "pk-build").symlink_to(src / "pk-build", target_is_directory=True)
    upstream = dst / "gemmini-rocc-tests"
    upstream.mkdir()
    shutil.copytree(src / "gemmini-rocc-tests/include", upstream / "include")
    (upstream / "rocc-software").symlink_to(src / "gemmini-rocc-tests/rocc-software", target_is_directory=True)
    model = dst / "libgemmini"
    model.mkdir()
    for artifact in (src / "libgemmini").iterdir():
        if artifact.suffix in (".cc", ".h") or artifact.name == "Makefile":
            shutil.copy2(artifact, model / artifact.name)
    for header in (model / "gemmini_params.h", upstream / "include/gemmini_params.h"):
        content = header.read_text()
        for key, value in (("BANK_NUM", 1), ("BANK_ROWS", args.scratchpad_rows),
                           ("ACC_ROWS", args.accumulator_rows)):
            content, count = re.subn(r"^#define " + key + r" \d+$",
                                    f"#define {key} {value}", content, flags=re.MULTILINE)
            if count != 1:
                raise RuntimeError("unexpected parameter header: " + key)
        header.write_text(content)
    with (dst / "build.log").open("w") as log:
        subprocess.run(["make", "-B", "RISCV=" + str(src / "prefix")], cwd=model,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(model / "libgemmini.so", prefix / "lib/libgemmini.so")
    print(dst)


if __name__ == "__main__":
    main()
