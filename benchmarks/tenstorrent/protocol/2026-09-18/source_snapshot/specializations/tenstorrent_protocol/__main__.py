"""CLI for restricted Tenstorrent protocol experiments."""
import argparse
import json
from pathlib import Path
from .search import strict_json_loads, extract_proposal


def read_json(path):
    p = Path(path)
    if p.stat().st_size > 100_000:
        raise ValueError('JSON input exceeds 100000 bytes')
    return strict_json_loads(p.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'certify', 'baseline', 'search'])
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--proposal', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--contract', type=Path, help='Frozen model-visible operation contract for search')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--model', default='dgxspark-glm/glm-5.3-flash')
    parser.add_argument('--model-timeout', type=float, default=600)
    parser.add_argument('--proof-timeout', type=float, default=600)
    args = parser.parse_args()
    from .adapter import validate_task, evaluate, certify, baseline
    try:
        task = validate_task(read_json(args.task))
        if args.action == 'baseline':
            result = {'program': baseline(task), 'rationale': 'Serial per-output control; not a model proposal'}
            if args.out:
                with args.out.open('x') as f:
                    f.write(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result, indent=2))
            return 0
        if args.action == 'search':
            if not args.out or not args.contract:
                parser.error('search requires --out and --contract')
            from .search import run
            result = run(task, args.out, args.contract.read_text(), rounds=args.rounds,
                         model=args.model, model_timeout=args.model_timeout,
                         proof_timeout=args.proof_timeout)
            print(json.dumps(result, indent=2))
            return 0 if result['status'] == 'certified' else 3
        if not args.proposal:
            parser.error('check/certify requires --proposal')
        if args.proposal.stat().st_size > 100_000:
            raise ValueError('proposal too large')
        proposal = extract_proposal(args.proposal.read_text())
        verdict = evaluate(task, proposal['program'])
        if args.action == 'check' or not verdict['accepted']:
            print(json.dumps(verdict, indent=2))
            return 0 if verdict['accepted'] else 3
        if not args.out:
            parser.error('certify requires --out')
        result = certify(task, proposal['program'], args.out, timeout=args.proof_timeout)
        print(json.dumps(result, indent=2))
        return 0 if result['accepted'] else 3
    except (ValueError, TypeError, OSError, RuntimeError, KeyError) as error:
        print(json.dumps({'status': 'error', 'reason': str(error)}, indent=2))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
