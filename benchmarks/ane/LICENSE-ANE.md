# ANE Probe — Adapted-Source Attribution & License

This benchmark adapts Objective-C code from **maderix/ANE**, which is MIT
licensed. The pinned upstream commit is:

    maderix/ANE  d91c9845c0784dec7753048954fc6d0e8411fe29  (2026-03-10)
      adapted files:
        bridge/ane_bridge.h   ->  src/ane_bridge.h
        bridge/ane_bridge.m   ->  src/ane_bridge.m
        inmem_basic.m         ->  IOSurface sizing / temp-dir / call conventions
        training/ane_mil_gen.h->  weight-blob byte layout (BLOBFILE offset 64)

Secondary reference (fallback, not adapted into this codebase):

    sbryngelson/ANEForge  67c1d4861c562b0168d60c7a5677840e2cad44ea  (2026-09-08)

The MIL `relu` op syntax was cross-checked against ANEForge's MIL emission.

The following upstream files were consulted and their pinned content is
preserved under `/tmp/ane-probe-work/ane/` for provenance:
`LICENSE`, `bridge/ane_bridge.m`, `training/ane_mil_gen.h`.

## MIT License

Copyright (c) 2026 maderix (original ANE code)
Copyright (c) 2026 VeriTac contributors (adaptations)

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
