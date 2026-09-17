"""Discover specialization descriptions without importing target runtimes.

Descriptions locate implementations and state proof scope. They are not proof
certificates, hardware profiles, or a universal backend execution interface.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = Path(__file__).resolve().parent


def describe(name):
    if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name):
        raise ValueError('invalid specialization identifier')
    path = PACKAGES / name / 'specialization.json'
    if not path.is_file():
        raise ValueError(f'unknown specialization: {name}')
    result = json.loads(path.read_text())
    if result.get('id') != name or result.get('schema_version') != 1:
        raise ValueError(f'invalid specialization description: {name}')
    return result


def list_specializations():
    return [describe(p.parent.name) for p in sorted(PACKAGES.glob('*/specialization.json'))]
