# MIT License — ANEForge Reference Material

The files under `reference/` (`e5rt_api.h`, `ane_e5rt_dispatch.mm`) are copied
unmodified from **sbryngelson/ANEForge**, which is MIT licensed. The pinned
upstream commit is:

    sbryngelson/ANEForge  67c1d4861c562b0168d60c7a5677840e2cad44ea  (2026-09-08)

The `e5rt_api.h` recovered C-API signatures are the authoritative ABI source
used by this probe (`ane_e5_probe.py`). Each signature is authoritative
individually; the probe never calls `e5rt_error_code_get_string` (C++ string
ABI).

## MIT License

Copyright (c) 2026 sbryngelson (ANEForge reference code)
Copyright (c) 2026 VeriTac contributors (probe, tests, README)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
