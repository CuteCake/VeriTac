"""Compatibility entry point for specializations.cpu_gemm.runner."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("specializations.cpu_gemm.runner", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("specializations.cpu_gemm.runner")
