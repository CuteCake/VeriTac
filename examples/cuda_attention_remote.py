#!/usr/bin/env python3
"""Remote helper for the VeriTac CUDA attention demo.

Runs on the CUDA host (the controller connects over SSH and pushes this file plus
a copy of Hardware/profile.py into a unique controller_runs/<UUID> directory).
It sets the build/run environment, builds the candidate runtime, and exposes
three subcommands, each printing a single JSON object to stdout (diagnostics go
to stderr, and per-run subprocess logs are captured separately):

  metadata  - normalized profile + full compiled config_info catalogue +
              canonical catalogue hash + binding hashes of EVERY loaded module
              .so, every candidate_root/*.cu, local vendor/*.h, the installed
              kernel header (derived from torch.__file__), run_candidate.py,
              and the CUTLASS git HEAD + dirty-diff.  This is an OBSERVED TRUST
              BOUNDARY, not a Lean proof.
  benchmark - invoke run_candidate.py --mode sweep for ONE config/seq/dim and
              return the single matching case.
  survey    - invoke cuda_survey.py forced auto,efficient,cudnn,flash,math
              float32 at the same B1H8 seed/shape/warmup/samples/inner and
              return the single matching case.

Environment: PATH is prefixed with the current venv bin and /usr/local/cuda/bin,
and CUDA_HOME / MAX_JOBS=2 / TORCH_CUDA_ARCH_LIST (from the actual torch.cuda
capability) are set before cpp_extension builds through run_candidate.  This
script never deletes files and never issues kill commands.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE_ROOT = "/home/cake/.cache/veritac-attention-run"
DEFAULT_CANDIDATE_ROOT = os.path.join(REMOTE_ROOT, "cuda_candidate")
DEFAULT_SURVEY = os.path.join(REMOTE_ROOT, "cuda_survey.py")

_RUNNER = None
_MODULES = {}
_TORCH = None
_PROFILE = None
_CAP = None


# ---------------------------------------------------------------------------
# Environment + profile loading
# ---------------------------------------------------------------------------

def setup_env(candidate_root):
    """Export build/run environment BEFORE any cpp_extension build.  Uses the
    current interpreter's venv (sys.executable), never a hardcoded Python."""
    venv_bin = os.path.dirname(sys.executable)
    cuda_bin = "/usr/local/cuda/bin"
    os.environ["PATH"] = (venv_bin + os.pathsep + cuda_bin +
                          os.pathsep + os.environ.get("PATH", ""))
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["MAX_JOBS"] = "2"
    os.environ.setdefault("VERITAC_CUTLASS_INCLUDE",
                          os.path.join(candidate_root, "cutlass", "include"))
    import torch
    cap = torch.cuda.get_device_capability(0)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "%d.%d" % cap
    return cap


def load_profile():
    """Load the copied profile.py under a unique module name to avoid colliding
    with the stdlib `profile` module."""
    path = os.path.join(HERE, "profile.py")
    spec = importlib.util.spec_from_file_location("veritac_remote_profile", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path):
    """Hash a REQUIRED file; raise (no None placeholders) if it is missing."""
    with open(path, "rb") as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
        return h.hexdigest()


def git_head(root):
    try:
        p = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        return (p.stdout or "").strip() if p.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def git_dirty_hash(root):
    try:
        p = subprocess.run(["git", "-C", root, "diff"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            return None
        return hashlib.sha256((p.stdout or "").encode()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def installed_kernel_header():
    """Locate the ACTUAL installed kernel_forward.h relative to torch.__file__.
    torch.__file__ lives IN the torch/ directory, so the header is under
    torch_dir/include/ATen/... (no `..` hop).  No existence fallback."""
    torch_dir = os.path.dirname(os.path.abspath(_TORCH.__file__))
    return os.path.realpath(os.path.join(
        torch_dir, "include", "ATen", "native", "transformers",
        "cuda", "mem_eff_attention", "kernel_forward.h"))


# ---------------------------------------------------------------------------
# Runtime + catalogue (uses backend's public build_runtime() API)
# ---------------------------------------------------------------------------

def ensure_built(candidate_root):
    global _RUNNER, _MODULES, _TORCH, _PROFILE, _CAP
    if _RUNNER is not None:
        return
    _CAP = setup_env(candidate_root)
    sys.path.insert(0, candidate_root)
    import torch
    _TORCH = torch
    _PROFILE = load_profile()
    import run_candidate
    if not hasattr(run_candidate, "build_runtime"):
        sys.exit("ERROR: run_candidate.build_runtime() is not available "
                 "(backend-worker public API).")
    _RUNNER, _MODULES = run_candidate.build_runtime()


def config_catalogue(candidate_root):
    """Full catalogue from Runner.infos.  Registry keys are the effective IDs
    (normal/reverse/alias variants); a config whose own id disagrees with its
    registry key is a hard failure (never silently renamed).  Unsupported
    entries still carry complete geometry metadata so the Nat fields parse, and
    Lean rejects them via supported=false."""
    ensure_built(candidate_root)
    out = {}
    for key, info in _RUNNER.infos.items():
        if not isinstance(info, dict):
            info = {"supported": False, "error": str(info)}
        info_id = info.get("id")
        if info_id is not None and info_id != key:
            sys.exit(f"ERROR: config info id {info_id} disagrees with registry "
                     f"key {key} (fail mismatch, do not rename)")
        entry = dict(info)
        entry["id"] = key
        out[key] = entry
    return out


def binding(candidate_root):
    """Hashes of every loaded module .so, all candidate_root/*.cu, vendor/*.h,
    the installed kernel header, run_candidate.py, and CUTLASS git state.  Lists
    the actual module paths.  Required files raise if missing."""
    ensure_built(candidate_root)
    # EVERY loaded module must expose its real .so __file__; fail rather than
    # silently dropping a module from the binding.
    for _name, m in _MODULES.items():
        if not hasattr(m, "__file__"):
            sys.exit(f"ERROR: loaded module {_name} has no __file__; cannot bind")
    module_paths = sorted({os.path.abspath(m.__file__) for m in _MODULES.values()})
    module_hashes = {os.path.basename(p): sha256_file(p) for p in module_paths}
    cu_paths = sorted(glob.glob(os.path.join(candidate_root, "*.cu")))
    cu_hashes = {os.path.basename(p): sha256_file(p) for p in cu_paths}
    vendor_paths = sorted(glob.glob(os.path.join(candidate_root, "vendor", "*.h")))
    vendor_hashes = {os.path.relpath(p, candidate_root): sha256_file(p)
                     for p in vendor_paths}
    run_py = os.path.join(candidate_root, "run_candidate.py")
    kernel_h = installed_kernel_header()
    cutlass_root = os.path.join(candidate_root, "cutlass")
    return {
        "module_paths": module_paths,
        "module_hashes": module_hashes,
        "cu_hashes": cu_hashes,
        "vendor_hashes": vendor_hashes,
        "run_candidate_py": sha256_file(run_py),
        "installed_kernel_forward_h": sha256_file(kernel_h),
        "installed_kernel_forward_h_path": kernel_h,
        "cutlass_git_head": git_head(cutlass_root),
        "cutlass_dirty_diff_hash": git_dirty_hash(cutlass_root),
        "torch": _TORCH.__version__,
        "torch_cuda": _TORCH.version.cuda,
        "compute_capability": "%d.%d" % _CAP,
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_metadata(args):
    infos = config_catalogue(args.candidate_root)
    prof = _PROFILE.probe(backend="cuda")
    if prof is None:
        print(json.dumps({"error": "could not probe CUDA profile"}))
        return
    _PROFILE.validate_schema(prof)
    catalogue = [infos[k] for k in infos]
    catalogue_hash = hashlib.sha256(
        json.dumps(catalogue, sort_keys=True, default=str).encode()).hexdigest()
    bind = binding(args.candidate_root)
    print(json.dumps({
        "profile": prof,
        "catalogue": catalogue,
        "config_ids": [k for k in infos],
        "catalogue_hash": catalogue_hash,
        "binding": bind,
        "module_paths": bind["module_paths"],
        "provenance": "observed trust boundary, not a Lean proof",
    }, default=str))


def cmd_benchmark(args):
    setup_env(args.candidate_root)
    run_py = os.path.join(args.candidate_root, "run_candidate.py")
    cmd = [sys.executable, run_py, "--mode", "sweep",
           "--configs", str(args.configs), "--seqs", str(args.seqs),
           "--dims", str(args.dims), "--seed", str(args.seed),
           "--warmup", str(args.warmup), "--samples", str(args.samples),
           "--inner", str(args.inner), "--out", args.out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(json.dumps({"error": "run_candidate failed",
                          "stdout": proc.stdout[-2000:],
                          "stderr": proc.stderr[-2000:]}))
        return
    with open(args.out) as f:
        doc = json.load(f)
    cases = [c for c in doc.get("cases", [])
             if c.get("config") == args.configs
             and c.get("seq") == args.seqs and c.get("dim") == args.dims]
    if len(cases) != 1:
        print(json.dumps({"error": "expected exactly one matching case, got %d"
                                     % len(cases)}))
        return
    case = dict(cases[0])
    case["config_info"] = doc.get("configs", {}).get(str(args.configs))
    case["run_metadata"] = doc.get("meta", {})
    case["runner_sha256"] = sha256_file(run_py)
    print(json.dumps(case, default=str))


def cmd_survey(args):
    setup_env(args.candidate_root)
    cmd = [sys.executable, args.survey, "--output", args.out,
           "--seqs", str(args.seqs), "--dims", str(args.dims),
           "--dtypes", "float32", "--backends", "auto,efficient,cudnn,flash,math",
           "--heads", "8", "--batch", "1", "--seed", str(args.seed),
           "--warmup", str(args.warmup), "--samples", str(args.samples),
           "--inner", str(args.inner)]
    if args.mem_cap_bytes is not None:
        cmd += ["--mem-cap-bytes", str(args.mem_cap_bytes)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(json.dumps({"error": "cuda_survey failed",
                          "stdout": proc.stdout[-2000:],
                          "stderr": proc.stderr[-2000:]}))
        return
    with open(args.out) as f:
        doc = json.load(f)
    cases = [c for c in doc.get("cases", [])
             if c.get("seq") == args.seqs and c.get("dim") == args.dims
             and c.get("dtype") == "float32"]
    if len(cases) != 1:
        print(json.dumps({"error": "expected exactly one survey case, got %d"
                                     % len(cases)}))
        return
    case = dict(cases[0])
    case["run_metadata"] = doc.get("meta", {})
    case["runner_sha256"] = sha256_file(args.survey)
    print(json.dumps(case, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Global options MUST precede the subcommand (client passes them first).
    ap.add_argument("--candidate-root", default=DEFAULT_CANDIDATE_ROOT)
    ap.add_argument("--survey", default=DEFAULT_SURVEY)
    ap.add_argument("--mem-cap-bytes", type=int, default=None,
                    help="forward to cuda_survey.py as --mem-cap-bytes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("metadata")
    pm.set_defaults(fn=cmd_metadata)

    pb = sub.add_parser("benchmark")
    for name in ("configs", "seqs", "dims"):
        pb.add_argument("--" + name, type=int, required=True)
    for name in ("seed", "warmup", "samples", "inner"):
        pb.add_argument("--" + name, type=int, default=0)
    pb.add_argument("--out", required=True)
    pb.set_defaults(fn=cmd_benchmark)

    ps = sub.add_parser("survey")
    for name in ("seqs", "dims"):
        ps.add_argument("--" + name, type=int, required=True)
    for name in ("seed", "warmup", "samples", "inner"):
        ps.add_argument("--" + name, type=int, default=0)
    ps.add_argument("--out", required=True)
    ps.set_defaults(fn=cmd_survey)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
