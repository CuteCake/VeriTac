# ANE Artifact Inspector (read-only)

Independent, read-only inspector for **compiled ANE hardware containers** in
the feasibility experiment.  It reports what a real compiled artifact exposes
at the *container* level without pretending to be a complete instruction
decoder.

It inspects **only explicit files / run-directories**; it never probes the
GPU/ANE, never executes untrusted files, and never globs user/system caches.

## What it is

A stdlib-only Python CLI that, for each explicitly named file, reports:

- file length and SHA-256 (streamed; safe for multi-GB weight banks)
- magic detection (Mach-O, HWX, fat, big-endian variants)
- bounded generic container shell: header, load-command census, and (where a
  command is self-consistent with the public Mach-O ABI) segment / section
  metadata and a symbol string table
- bounded string extraction for compiler / target identification *hints*
- JSON / plist metadata when the file is such a document

All region reads are bounded by explicit caps; the whole file is never loaded
into memory.

## Semantic scope (important)

For any HWX / Mach-O-shaped container the report declares:

- `semantic_scope: "container_only"`
- `instruction_semantics: "instruction_semantics_unmodeled"`

The inspector **does not** decode register-write / task-descriptor opcodes or
their semantics.  M1 `v10` / `H13` register offsets, DMA apertures, and opcode
maps are **not** assumed valid on this M3 Ultra target.  A field is only
interpreted when it has a documented, generation-independent justification
(the Mach-O-shaped shell itself); everything else is preserved as raw
type / offset / size / hash.

Because the HWX command ABI is not guaranteed by type number and size alone,
generic Mach-O-shaped interpretations applied to HWX commands are labelled
`interpretation: "provisional"` and carry **no semantic claim**.

## Exactly what is decoded vs. unknown

**Decoded (format-justified, generic):**
- Magic word and its known name (`MH_MAGIC[_64]`, `MH_CIGAM[_64]`,
  `FAT_MAGIC[_64]`, `HWX_ENGINE_EXEC`)
- Header fields common to the Mach-O-shaped shell: cputype, cpusubtype,
  filetype, ncmds, sizeofcmds, flags (HWX shell header is 32 bytes)
- Load-command boundaries (`cmd`, `cmdsize`) and, only when a command number
  maps to a public Mach-O ABI *and* meets that command's own minimum
  cmdsize: `LC_SEGMENT`/`LC_SEGMENT_64` (segment + section metadata),
  `LC_SYMTAB` (string table), `LC_BUILD_VERSION`
- Bounded printable-ASCII string extraction and compiler/target *hints*
  (heuristic; no vendor-independence claim)
- JSON / plist metadata (bounded, normalized to JSON-safe values)

**Preserved raw, not interpreted:**
- Any load command that is not a confidently-known Mach-O command is reported
  as `UNKNOWN` with its raw type, offset, size, and SHA-256 — no guessed
  opcode or register semantics
- For HWX, even known-numbered commands are marked provisional
- Task-descriptor / register-write streams: never decoded

## Bounds and validation

- header must fit the file; `ncmds*8 <= sizeofcmds`; each command `>= 8`
  bytes and within both `sizeofcmds` and the file
- each known command enforces its own minimum cmdsize so it cannot borrow
  bytes from its successor
- segment/section counts must fit inside their command
- file-backed segments/sections must lie within the file; **virtual
  zero-fill regions are exempt** from the file-range check:
  - a segment with `filesize == 0` has no file bytes; HWX sections in such
    windows are provisionally treated as virtual tensor declarations
  - a section whose type (low 8 bits of flags) is a zero-fill type
    (`S_ZEROFILL 0x1`, `S_GB_ZEROFILL 0xc`, `S_THREAD_LOCAL_ZEROFILL 0x12`)
    carries a *virtual* size, not file bytes — it may be nonzero without
    occupying file space (e.g. a 1 MiB `__bss`)
  - virtual sections are reported with `"virtual": true` and their
    `section_type`; regular file-backed sections whose range exceeds EOF are
    still rejected as `out_of_bounds`
- read caps: load-command region 64 MiB, per-region string scan 8 MiB,
  aggregate string scan 32 MiB, symtab 8 MiB, metadata 16 MiB
- symlink / reference traversal is confined to the explicit input root;
  escaping symlinks are rejected and reported (not followed)

## CLI

```
PYTHONPATH=. python3 inspector.py inspect <file-or-dir> [<file-or-dir> ...]
```

Output is a JSON document on stdout:

- `inspector.{name,version}`
- `input_hash` — fingerprint over the inspected artifacts
- `artifacts[]` — per input, each artifact with path / file_length / sha256 /
  magic / container / strings / coverage, plus `status`
- `coverage` — container_only + instruction_semantics_unmodeled
- `security_notes[]` — rejected escaping symlinks, etc.
- `status` / `errors[]`

Exit status:

- `0` — inspection completed (unknown formats are reported, not errors; an
  empty directory yields `artifact_count 0` with status `ok`)
- `2` — a requested input was malformed / truncated / unreadable
- `1` — usage error

## Tests

```
PYTHONPATH=. python3 -m unittest tests/test_inspector.py -v
```

Covers: valid synthetic HWX (segment/section/symbol/unknown-command
preservation, provisional labelling, compiler hints), plain 64-bit Mach-O,
fat and big-endian Mach-O magic handling, truncated / out-of-bounds /
count-overflow / short-known-command malformed fixtures, symlink-escape
rejection, empty-directory negative case, bounded-read truncation flag,
zero-fill virtual sections (nonzero virtual size exempt from file-range
checks) vs. malformed regular sections (rejected), and JSON/plist metadata
normalization.

## Provenance

Format facts are drawn from the primary research referenced by the task:

- https://github.com/sbryngelson/ane-guide/blob/main/part-7-toolchain/23-program-format.md
- https://github.com/sbryngelson/ane-guide/blob/main/part-7-toolchain/22-compiler.md

Those describe the M1 `v10`/`H13` generation.  This inspector treats those
values as context, **not** as valid offsets for the M3 Ultra; the decode is
limited to the generation-independent container shell.
