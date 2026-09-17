"""Compatibility entry point for specializations.gemmini_gemm.encoding."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("specializations.gemmini_gemm.encoding", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("specializations.gemmini_gemm.encoding")
