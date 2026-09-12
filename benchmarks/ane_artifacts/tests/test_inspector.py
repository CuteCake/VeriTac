"""Tests for the read-only ANE artifact inspector."""

import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aneinspector import reader, walk
from aneinspector.inspect import inspect_file
from aneinspector.macho import ParseError
from fixtures import (build_bad_regular_section, build_big_endian_macho32,
                      build_count_overflow, build_fat, build_macho64,
                      build_out_of_bounds, build_short_known_command,
                      build_truncated, build_valid_hwx, build_zerofill_macho64)

import inspector


class ReadBoundedTest(unittest.TestCase):
    def test_truncation_flag_preserved(self):
        data = b"A" * 100
        fh = io.BytesIO(data)
        # requested length 200 exceeds cap 64 -> must report truncated True
        got, truncated = reader.read_bounded(fh, 0, 200, 64)
        self.assertEqual(len(got), 64)
        self.assertTrue(truncated)
        # requested length within cap -> truncated False
        got2, truncated2 = reader.read_bounded(fh, 0, 40, 64)
        self.assertEqual(len(got2), 40)
        self.assertFalse(truncated2)

    def test_short_file_returns_none(self):
        fh = io.BytesIO(b"AAAA")
        got, truncated = reader.read_bounded(fh, 0, 100, 64)
        self.assertIsNone(got)


class InspectFileTest(unittest.TestCase):
    def _write(self, data):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    def test_valid_hwx(self):
        path = self._write(build_valid_hwx())
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["magic"]["name"], "HWX_ENGINE_EXEC")
            self.assertEqual(res["magic"]["variant"], "hwx")
            cont = res["container"]
            self.assertEqual(cont["header"]["ncmds"], 5)
            self.assertEqual(len(cont["segments"]), 3)
            names = {s["segname"] for s in cont["segments"]}
            self.assertEqual(names, {"__TEXT", "__KERN_0", "__FVMLIB"})
            # virtual window segment must be reported, not treated as file-backed
            win = [s for s in cont["segments"] if s["segname"] == "__FVMLIB"][0]
            self.assertEqual(win["filesize"], 0)
            self.assertEqual(win["fileoff"], 0)
            cov = res["coverage"]
            self.assertEqual(cov["semantic_scope"], "container_only")
            self.assertEqual(cov["instruction_semantics"],
                             "instruction_semantics_unmodeled")
            self.assertTrue(any("zin_ane_compiler" in s
                                for s in res.get("compiler_hints", [])))
        finally:
            os.unlink(path)

    def test_unknown_command_preserved(self):
        path = self._write(build_valid_hwx())
        try:
            res = inspect_file(path)
            cmds = res["container"]["load_commands"]
            unknown = [c for c in cmds if c["type"] == 0x77777777]
            self.assertEqual(len(unknown), 1)
            u = unknown[0]
            self.assertEqual(u["name"], "UNKNOWN")
            self.assertIn("offset", u)
            self.assertIn("size", u)
            self.assertTrue(u["sha256"])
            # unknown command must NOT be interpreted as a segment
            self.assertNotIn("segment", u)
        finally:
            os.unlink(path)

    def test_symbol_names(self):
        path = self._write(build_valid_hwx())
        try:
            res = inspect_file(path)
            self.assertIn("symbol_names", res)
            names = res["symbol_names"]["sample"]
            self.assertIn("main_ane", names)
        finally:
            os.unlink(path)

    def test_macho64(self):
        path = self._write(build_macho64())
        try:
            res = inspect_file(path)
            self.assertEqual(res["magic"]["name"], "MH_MAGIC_64")
            self.assertEqual(res["container"]["header"]["header_width"], 64)
            self.assertEqual(len(res["container"]["segments"]), 1)
        finally:
            os.unlink(path)

    def test_truncated(self):
        path = self._write(build_truncated())
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "malformed")
            self.assertEqual(res["error"]["reason"], "truncated")
        finally:
            os.unlink(path)

    def test_out_of_bounds(self):
        path = self._write(build_out_of_bounds())
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "malformed")
            self.assertEqual(res["error"]["reason"], "count_overflow")
        finally:
            os.unlink(path)

    def test_count_overflow(self):
        path = self._write(build_count_overflow())
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "malformed")
            self.assertEqual(res["error"]["reason"], "count_overflow")
        finally:
            os.unlink(path)

    def test_unknown_format_file(self):
        path = self._write(b"not a container at all")
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "unknown_format")
        finally:
            os.unlink(path)

    def test_json_metadata_file(self):
        path = self._write(b'{"schema": "1.0.10", "Networks": ["n"]}')
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "metadata")
            self.assertEqual(res["metadata"]["kind"], "json")
        finally:
            os.unlink(path)

    def test_fat_magic_not_macho(self):
        path = self._write(build_fat())
        try:
            res = inspect_file(path)
            self.assertEqual(res["magic"]["variant"], "fat")
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["container"]["nfat_arch"], 2)
        finally:
            os.unlink(path)

    def test_big_endian_macho32_not_hwx(self):
        path = self._write(build_big_endian_macho32())
        try:
            res = inspect_file(path)
            # 0xCEFAEDFE is MH_CIGAM (big-endian Mach-O), NOT HWX
            self.assertEqual(res["magic"]["name"], "MH_CIGAM")
            self.assertEqual(res["magic"]["variant"], "macho_be32")
            self.assertEqual(res["magic"]["endian"], "be")
            self.assertEqual(res["status"], "ok")
        finally:
            os.unlink(path)

    def test_hwx_header_is_32_bytes(self):
        path = self._write(build_valid_hwx())
        try:
            res = inspect_file(path)
            self.assertEqual(res["magic"]["variant"], "hwx")
            self.assertEqual(res["magic"]["header_length"], 32)
            self.assertEqual(res["container"]["header"]["header_length"], 32)
        finally:
            os.unlink(path)

    def test_short_file_no_magic_no_crash(self):
        path = self._write(b"ab")
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "unknown_format")
            self.assertNotIn("magic", res)
        finally:
            os.unlink(path)

    def test_truncated_read_header(self):
        # HWX magic but file far shorter than the 32-byte header
        path = self._write(b"\xce\xfa\xef\xbe" + b"\x00" * 4)
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "malformed")
            self.assertEqual(res["error"]["reason"], "truncated")
        finally:
            os.unlink(path)

    def test_short_known_command(self):
        path = self._write(build_short_known_command())
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "malformed")
            self.assertEqual(res["error"]["reason"], "malformed")
        finally:
            os.unlink(path)

    def test_hwx_generic_interpretations_provisional(self):
        path = self._write(build_valid_hwx())
        try:
            res = inspect_file(path)
            interp = res["container"]["interpretation"]
            self.assertEqual(interp["level"], "provisional")
            segs = [c for c in res["container"]["load_commands"]
                    if "segment" in c]
            self.assertTrue(all(c.get("interpretation") == "provisional"
                                for c in segs))
        finally:
            os.unlink(path)

    def test_strings_nonempty_and_compiler_hints(self):
        path = self._write(build_valid_hwx())
        try:
            res = inspect_file(path)
            self.assertGreater(res["strings"]["count"], 0)
            self.assertTrue(any("zin_ane_compiler" in s
                                for s in res["compiler_hints"]))
            self.assertTrue(any("com.apple.ANECompilerFramework" in s
                                for s in res["compiler_hints"]))
        finally:
            os.unlink(path)

    def test_regular_section_in_empty_macho_segment_rejected(self):
        import struct
        blob = bytearray(build_zerofill_macho64())
        struct.pack_into("<I", blob, 32 + 72 + 64, 0)
        path = self._write(blob)
        try:
            self.assertEqual(inspect_file(path)["status"], "malformed")
        finally:
            os.unlink(path)

    def test_zerofill_section_not_file_backed(self):
        path = self._write(build_zerofill_macho64())
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "ok")
            bss = [s for s in res["container"]["sections"]
                   if s["sectname"] == "__bss"][0]
            self.assertTrue(bss["virtual"])
            self.assertEqual(bss["section_type"], 0x1)
            self.assertEqual(bss["size"], 0x100000)  # nonzero virtual size
        finally:
            os.unlink(path)

    def test_bad_regular_section_rejected(self):
        path = self._write(build_bad_regular_section())
        try:
            res = inspect_file(path)
            self.assertEqual(res["status"], "malformed")
            self.assertEqual(res["error"]["reason"], "out_of_bounds")
        finally:
            os.unlink(path)


class WalkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_directory(self):
        files, notes, _ = walk.collect(self.tmp)
        self.assertEqual(files, [])
        self.assertEqual(notes, [])

    def test_symlink_escape_rejected(self):
        outside_dir = tempfile.mkdtemp()
        try:
            outside_file = os.path.join(outside_dir, "secret.bin")
            with open(outside_file, "wb") as f:
                f.write(b"SECRET")
            os.symlink(outside_file, os.path.join(self.tmp, "leak.bin"))
            files, notes, _ = walk.collect(self.tmp)
            self.assertEqual(files, [])
            self.assertTrue(any("escaping" in n for n in notes))
        finally:
            import shutil
            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_directory_symlink_escape_rejected(self):
        outside_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(outside_dir, "x.bin"), "wb") as f:
                f.write(b"X")
            os.symlink(outside_dir, os.path.join(self.tmp, "dirleak"))
            files, notes, _ = walk.collect(self.tmp)
            self.assertEqual(files, [])
            self.assertTrue(any("escaping" in n for n in notes))
        finally:
            import shutil
            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_symlink_within_root_followed(self):
        inside = os.path.join(self.tmp, "inside.bin")
        with open(inside, "wb") as f:
            f.write(b"data")
        os.symlink(inside, os.path.join(self.tmp, "link.bin"))
        files, notes, _ = walk.collect(self.tmp)
        self.assertIn(inside, files)
        self.assertEqual(notes, [])


class CliTest(unittest.TestCase):
    def _write(self, data):
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    def _capture(self, argv):
        old = sys.stdout
        buf = io.StringIO()
        sys.stdout = buf
        try:
            code = inspector.main(argv)
        finally:
            sys.stdout = old
        return code, json.loads(buf.getvalue())

    def test_valid_exit_zero(self):
        path = self._write(build_valid_hwx())
        try:
            code, report = self._capture(["inspect", path])
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "ok")
            self.assertIn("inspector", report)
            self.assertIn("input_hash", report)
        finally:
            os.unlink(path)

    def test_malformed_exit_two(self):
        path = self._write(build_truncated())
        try:
            code, report = self._capture(["inspect", path])
            self.assertEqual(code, 2)
            self.assertEqual(report["status"], "error")
        finally:
            os.unlink(path)

    def test_empty_dir_exit_zero(self):
        d = tempfile.mkdtemp()
        try:
            code, report = self._capture(["inspect", d])
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "ok")
            art = report["artifacts"][0]
            self.assertEqual(art["artifact_count"], 0)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
