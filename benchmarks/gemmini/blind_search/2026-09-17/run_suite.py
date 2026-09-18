"""Execute the preregistered suite; preserve every outcome, including failures.

Run from the repository root. A suite never overwrites existing runs. Model
proposals use the separate tool-denied controller; this driver sees controls
only after all proposal rounds for a task finish.
"""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from specializations.gemmini_gemm import blind_search as blind
from specializations.gemmini_gemm.evaluation import compare_controls


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((HERE / "preregistration.json").read_text())
    fingerprint = json.loads((HERE / "acceptance_fingerprint.json").read_text())
    for name, digest in fingerprint["sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("formal acceptance fingerprint changed: " + name)
    tasks = []
    for name, digest in manifest["tasks_sha256"].items():
        task = HERE / "tasks" / name
        if sha(task) != digest:
            raise RuntimeError("preregistered task changed: " + name)
        tasks.append(task)
    runs = HERE / "runs"
    runs.mkdir(exist_ok=False)

    def run(task):
        task_id = blind.load_task(task)["id"]
        output = runs / task_id
        command = [sys.executable, str(ROOT / "examples/gemmini_blind_search.py"), "run",
                   "--task", str(task), "--out", str(output), "--rounds", "4",
                   "--model", manifest["model"], "--model-timeout", "900",
                   "--proof-timeout", "600", "--max-commands", "4096"]
        print("START", task_id, datetime.now(timezone.utc).isoformat(), flush=True)
        with (runs / (task_id + ".log")).open("w") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        outcome = {"task_id": task_id, "exit_code": process.returncode,
                   "output": str(output), "command": command}
        report_path = output / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            outcome["status"] = report["status"]
            outcome["rounds"] = [r["status"] for r in report["rounds"]]
            winner = report.get("winner")
            if winner and winner.get("certified"):
                commands = blind._reload_commands(Path(winner["commands_path"]))
                comparison = compare_controls(report["plan"], commands)
                (output / "controls.json").write_text(json.dumps(comparison, indent=2) + "\n")
                outcome["input_reduction_vs_baseline"] = comparison["input_reduction_vs_baseline"]
                outcome["input_reduction_vs_best_existing"] = comparison["input_reduction_vs_best_existing"]
        else:
            outcome["status"] = "controller_failure"
        print("DONE", json.dumps(outcome), flush=True)
        return outcome

    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                outcomes.append(future.result())
            except Exception as error:
                outcomes.append({"task_id": futures[future].stem, "status": "driver_failure", "error": repr(error)})
            (HERE / "suite_outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    return 0 if all(o.get("status") == "certified" for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
