"""Summarize all registered tasks, retaining failures and unknown token usage."""
import argparse
from collections import Counter
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summarize(runs):
    cases = []
    for task_path in sorted((HERE / "tasks").glob("*.json")):
        task = json.loads(task_path.read_text())
        directory = runs / task["id"]
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        receipts = [json.loads(path.read_text()) for path in sorted((directory / "receipts").glob("*.json"))]
        abort_path = directory / "abort.json"
        status = report["status"] if report else ("controller_failure" if abort_path.exists() else "pending")
        item = {"task_id": task["id"], "plan": task["plan"], "status": status,
                "completed_rounds": len(receipts),
                "round_status_counts": dict(Counter(r["status"] for r in receipts)),
                "model_call_wall_seconds_sum": round(sum(r["model"]["duration_seconds"] for r in receipts), 3),
                "rounds_with_unknown_usage": [r["round"] for r in receipts if r.get("usage") is None],
                "reported_tokens": {},
                "first_native_accepted_round": next((r["round"] for r in receipts if r["status"] == "native_accepted"), None),
                "byte_certified": False, "input_bytes": None, "old_best_input_bytes": None,
                "input_reduction_vs_old_best": None, "proof_check_seconds": None,
                "local_execution": None, "spike_execution": None}
        for key in ("input", "output", "reasoning", "total"):
            values = [r["usage"][key] for r in receipts if isinstance(r.get("usage"), dict) and key in r["usage"]]
            item["reported_tokens"][key] = sum(values) if values else None
        if report and report["status"] == "certified":
            winner = report["winner"]
            item.update(byte_certified=bool(winner["certified"]), input_bytes=winner["metrics"]["dma_input_bytes"],
                        proof_check_seconds=winner["certification"]["kernel_certificate"]["kernel_check_seconds"],
                        winner_round=winner["round"], command_count=winner["metrics"]["command_count"],
                        counts=winner["metrics"]["counts"])
            controls_path = directory / "controls.json"
            if controls_path.exists():
                controls = json.loads(controls_path.read_text())
                best = next(c for c in controls["controls"] if c["name"] == controls["best_existing_control"])
                item.update(old_best_input_bytes=best["counts"]["dma_in_elems_int8"],
                            input_reduction_vs_old_best=controls["input_reduction_vs_best_existing"],
                            input_reduction_vs_baseline=controls["input_reduction_vs_baseline"])
            local = directory / "local_execution.json"
            if local.exists():
                item["local_execution"] = json.loads(local.read_text())["success"]
            spike = directory / "spike/transport.json"
            if spike.exists():
                item["spike_execution"] = json.loads(spike.read_text()).get("success", False)
        cases.append(item)
    totals = {"registered_tasks": len(cases), "certified_tasks": sum(c["byte_certified"] for c in cases),
              "pending_tasks": sum(c["status"] == "pending" for c in cases),
              "completed_rounds": sum(c["completed_rounds"] for c in cases),
              "rounds_without_usage": sum(len(c["rounds_with_unknown_usage"]) for c in cases),
              "model_call_wall_seconds_sum": round(sum(c["model_call_wall_seconds_sum"] for c in cases), 3)}
    totals["round_status_counts"] = dict(sum((Counter(c["round_status_counts"]) for c in cases), Counter()))
    totals["reported_tokens"] = {}
    for key in ("input", "output", "reasoning", "total"):
        values = [c["reported_tokens"][key] for c in cases if c["reported_tokens"][key] is not None]
        totals["reported_tokens"][key] = sum(values) if values else None
    return {"runs": str(runs), "totals": totals, "cases": cases,
            "measurement_notes": [
                "Token sums cover only reported usage; timed-out responses may have consumed unreported tokens.",
                "Summed model-call wall time is not elapsed experiment time or GPU execution time; tasks run concurrently.",
                "Input bytes are modeled full-tile DMA traffic, not measured DRAM transactions or hardware latency.",
                "Old-best controls are the baseline and feasible batched-B generators fixed at registration.",
                "Provider cost zero is not evidence of zero compute cost."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.runs), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
