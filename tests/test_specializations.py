"""Integration checks for package discovery, compatibility, and source relocation."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from specializations.catalog import ROOT, describe, list_specializations


class SpecializationTests(unittest.TestCase):
    def test_discovery_does_not_load_hardware_dependencies(self):
        code = "from specializations.catalog import list_specializations; import sys; assert len(list_specializations()) == 5; assert not any(n in sys.modules for n in ('torch', 'numpy', 'mlx'))"
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_catalog_paths_resolve_and_scopes_are_explicit(self):
        for spec in list_specializations():
            self.assertTrue(spec['proof_scope'])
            self.assertTrue(spec['boundaries'])
            for path in spec['implementation'] + spec['formal_sources'] + spec['evidence'] + list(spec['entrypoints'].values()):
                self.assertTrue((ROOT / path).exists(), path)

    def test_unknown_or_path_identifiers_rejected(self):
        for name in ('../CodeGen', '/tmp', 'missing', ''):
            with self.assertRaises(ValueError):
                describe(name)

    def test_legacy_imports_share_module_and_patch_state(self):
        pairs = {'lower': 'cpu_gemm.lower', 'emit_c': 'cpu_gemm.emit_c',
                 'runner': 'cpu_gemm.runner', 'gemmini': 'gemmini_gemm.backend',
                 'gemmini_encoding': 'gemmini_gemm.encoding',
                 'gemmini_certificate': 'gemmini_gemm.certificate',
                 'gemmini_lowering': 'gemmini_gemm.lowering'}
        for old, new in pairs.items():
            self.assertIs(importlib.import_module('CodeGen.' + old),
                          importlib.import_module('specializations.' + new))
        from specializations.gemmini_gemm import backend
        with patch('CodeGen.gemmini_encoding.emit_encoding', side_effect=RuntimeError('probe')):
            encoding, reason = backend.emit_encoding({}, [])
            self.assertIsNone(encoding)
            self.assertIn('probe', reason)

    def test_checker_paths_do_not_depend_on_working_directory(self):
        from specializations.gemmini_gemm import backend, certificate
        self.assertEqual(certificate.ROOT, ROOT)
        code = "from specializations.gemmini_gemm import backend as g; p=g.make_plan(16,16,16,'baseline'); assert g.check_program(p,g.gen_commands(p))[0]"
        import os
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([sys.executable, '-c', code], cwd=folder, env=env, check=True)

    def test_raw_runner_cli_matches_compatibility_path(self):
        canonical = ROOT / 'specializations/gemmini_gemm/runtime/run_raw_spike.py'
        legacy = ROOT / 'benchmarks/gemmini/run_raw_spike.py'
        self.assertEqual(canonical, legacy.resolve())
        for path in (canonical, legacy):
            result = subprocess.run([sys.executable, str(path), '--help'], cwd='/tmp', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--binary', result.stdout)

    def test_kernel_assets_remain_adjacent_to_drivers(self):
        for old, new in [('cuda_candidate', 'cuda_attention/kernels'),
                         ('partitioned', 'metal_attention/partitioned')]:
            self.assertEqual((ROOT / 'benchmarks/attention' / old).resolve(), ROOT / 'specializations' / new)
        for name in ('metal_candidate', 'partitioned/steel_attention', 'partitioned/matrix_attention', 'partitioned/partitioned_attention'):
            p = ROOT / 'specializations/metal_attention' / name
            self.assertTrue(p.with_suffix('.py').exists())
            self.assertTrue(p.with_suffix('.swift').exists())
        cuda = ROOT / 'specializations/cuda_attention/kernels'
        self.assertTrue((cuda / 'vendor/LICENSE-CUTLASS').exists())
        self.assertEqual(len(list(cuda.glob('*.cu'))), 4)


if __name__ == '__main__':
    unittest.main()
