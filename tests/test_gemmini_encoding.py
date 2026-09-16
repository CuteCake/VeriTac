"""Worker2 audit tests for VeriTac/Gemmini/Encoding.lean (controller-owned core).

Every meaningful case is executed by the REAL Lean decoder: the test writes a
Lean script that calls VeriTac.Gemmini.Encoding.decode on concrete bytes and
runs it with `lake env lean`, then asserts on the decoder's actual output.
Python (CodeGen.gemmini_encoding) supplies the untrusted bytes; Lean must
accept the valid streams and reject every mutation listed below.

Run from the project root:  PYTHONPATH=. python3 -m unittest tests.test_gemmini_encoding
"""
import json
import os
import re
import subprocess
import tempfile
import unittest

from CodeGen import gemmini as g
from CodeGen import gemmini_encoding as ge

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODING_MODULE = "VeriTac.Gemmini.Encoding"

BASES = ge.DEFAULT_BASES          # A 0x10000000, B 0x20000000, C 0x30000000
FENCE_WORD = 0x0FF0000F
RET_WORD = 0x00008067


def lean_available():
    """The audited module must at least build before these tests can run."""
    try:
        subprocess.run(["lake", "build", ENCODING_MODULE], cwd=REPO,
                       check=True, capture_output=True, timeout=1200)
        return True
    except Exception:
        return False


def words_to_hex(words):
    return b"".join(w.to_bytes(4, "little") for w in words).hex()


def expected_packets(commands, bases=BASES):
    """Python-side expectation: one OpPacket per non-fence command."""
    out = []
    for c in commands:
        pkt = ge.command_packet(c, bases)
        if pkt is not None:
            out.append(pkt)
    return out


def lean_decode_batch(cases):
    """Run several decode cases in ONE Lean process.

    cases: list of dicts with keys name, hex, and optional bases/sizes
    (bases=(a,b,c), sizes=(sa,sb,sc)).  Returns {name: "OK packets" | "ERR msg"}.
    """
    lines = [
        "import VeriTac.Gemmini.Encoding",
        "open VeriTac.Gemmini.Encoding",
        "def dec (a b c sa sb sc : Nat) (bs : List UInt8) : String :=",
        "  match decode ⟨a, b, c⟩ (sa, sb, sc) bs with",
        "  | .ok ps => \"OK \" ++ String.intercalate \",\" "
        "(ps.map (fun p => \"(\" ++ toString p.funct ++ \",\" ++ toString p.rs1"
        " ++ \",\" ++ toString p.rs2 ++ \")\"))",
        "  | .error e => \"ERR \" ++ e",
    ]
    names = []
    for i, case in enumerate(cases):
        name = case["name"]
        names.append(name)
        a, b, c = case.get("bases", (BASES["A"], BASES["B"], BASES["C"]))
        sa, sb, sc = case.get("sizes", (16, 16, 16))
        byts = ", ".join("0x%02x" % x for x in bytes.fromhex(case["hex"]))
        lines.append("def bytes%d : List UInt8 := [%s]" % (i, byts))
        lines.append('#eval IO.println ("%s|" ++ dec %d %d %d %d %d %d bytes%d)'
                     % (name, a, b, c, sa, sb, sc, i))
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    try:
        proc = subprocess.run(["lake", "env", "lean", path], cwd=REPO,
                              capture_output=True, text=True, timeout=1200)
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        raise AssertionError("Lean script failed:\n%s\n%s"
                             % (proc.stdout[-3000:], proc.stderr[-3000:]))
    results = {}
    for line in proc.stdout.splitlines():
        m = re.match(r'^([A-Za-z0-9_]+)\|((?:OK|ERR).*)$', line.strip())
        if m:
            results[m.group(1)] = m.group(2)
    missing = [n for n in names if n not in results]
    if missing:
        raise AssertionError("missing Lean results for %s; stdout:\n%s"
                             % (missing, proc.stdout[-3000:]))
    return results


def parse_ok(result):
    """'OK (2,16777216,..),...' -> [(2,16777216,..), ...]"""
    assert result.startswith("OK "), result
    return [tuple(int(x) for x in tup.split(","))
            for tup in re.findall(r"\((\d+,\d+,\d+)\)", result)]


@unittest.skipUnless(lean_available(), "could not build " + ENCODING_MODULE)
class EncodingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan = g.make_plan(16, 16, 16, g.SCHEDULE_BASELINE)
        cls.plan16 = plan
        cls.cmds16 = g.gen_commands(plan)
        cls.enc16 = ge.emit_encoding(plan, cls.cmds16)
        cls.exp16 = expected_packets(cls.cmds16)
        plan_b = g.make_plan(32, 32, 32, g.SCHEDULE_REUSE_B)
        cls.encb = ge.emit_encoding(plan_b, g.gen_commands(plan_b))
        cls.expb = expected_packets(g.gen_commands(plan_b))

    # -- valid streams ----------------------------------------------------
    def test_valid_baseline_16(self):
        res = lean_decode_batch([{
            "name": "valid16", "hex": self.enc16["bytes_hex"],
            "sizes": (self.enc16["sizes"]["A"], self.enc16["sizes"]["B"],
                      self.enc16["sizes"]["C"])}])
        got = parse_ok(res["valid16"])
        self.assertEqual(got, [tuple(p) for p in self.exp16])
        self.assertEqual(len(got), len(self.cmds16) - 1)  # fence excluded

    def test_valid_reuse_b_32(self):
        e = self.encb
        res = lean_decode_batch([{
            "name": "validb", "hex": e["bytes_hex"],
            "sizes": (e["sizes"]["A"], e["sizes"]["B"], e["sizes"]["C"])}])
        self.assertEqual(parse_ok(res["validb"]), [tuple(p) for p in self.expb])

    # -- signed-extension edges -------------------------------------------
    def test_sign_extension_edge_values(self):
        """li64 must survive two's-complement edge constants through the real
        Lean LUI/ADDI semantics (hi12/lo12 >= 0x800 carry cases)."""
        edges = [0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF00000000, 0x8000000000000000,
                 0xABCDEF123456789F, 0x0000000000000FFF, 0x7FFFFFFFFFFFFFFF]
        words = []
        for v in edges:
            words += ge.packet_words((0, v, v))
        words += [FENCE_WORD, RET_WORD]
        res = lean_decode_batch([{"name": "signedge", "hex": words_to_hex(words)}])
        got = parse_ok(res["signedge"])
        self.assertEqual(got, [(0, v, v) for v in edges])

    # -- malformed instructions -------------------------------------------
    def test_unknown_opcode(self):
        words = ge.packet_words((0, 1, 2)) + [FENCE_WORD, RET_WORD]
        bad = list(words)
        bad[8] ^= 0x5B ^ 0x2B           # custom word opcode 0x7B -> 0x2B
        res = lean_decode_batch([
            {"name": "unkop", "hex": words_to_hex(bad)},
            {"name": "goodctl", "hex": words_to_hex(words)}])
        self.assertTrue(res["unkop"].startswith("ERR"))
        self.assertTrue(res["goodctl"].startswith("OK"))

    def test_bad_custom_fields(self):
        """rs1/rs2 fields, funct3, rd, and funct range are all checked."""
        pw = ge.packet_words((0, 1, 2))
        li, custom = pw[:16], pw[16]     # 8-word li64 x5, li64 x6, custom
        self.assertNotEqual(custom & 0x7F, 0x13)  # sanity: word 16 is custom
        variants = {
            "rs1_not5": (custom & ~(0x1F << 15)) | (4 << 15),
            "rs2_not6": (custom & ~(0x1F << 20)) | (7 << 20),
            "rd_not0": custom | (1 << 7),
            "funct3_not3": (custom & ~0x7000) | (2 << 12),
            "funct7_toobig": (custom & ~(0x7F << 25)) | (7 << 25),
        }
        cases = [{"name": "goodctl",
                  "hex": words_to_hex(li + [custom] + [FENCE_WORD, RET_WORD])}]
        for name, w in variants.items():
            cases.append({"name": name,
                          "hex": words_to_hex(li + [w] + [FENCE_WORD, RET_WORD])})
        res = lean_decode_batch(cases)
        self.assertTrue(res["goodctl"].startswith("OK"))
        for name in variants:
            self.assertTrue(res[name].startswith("ERR"),
                            "%s not rejected: %s" % (name, res[name]))

    def test_uninitialized_register_read(self):
        """A custom packet issued before x5/x6 are written must be rejected."""
        words = [(0 << 25) | (6 << 20) | (5 << 15) | 0x3000 | 0x7B,
                 FENCE_WORD, RET_WORD]
        res = lean_decode_batch([{"name": "uninit", "hex": words_to_hex(words)}])
        self.assertTrue(res["uninit"].startswith("ERR"))

    # -- control structure -------------------------------------------------
    def test_missing_fence_and_return(self):
        words = ge.packet_words((0, 1, 2))
        res = lean_decode_batch([
            {"name": "noret", "hex": words_to_hex(words + [FENCE_WORD])},
            {"name": "nofence", "hex": words_to_hex(words + [RET_WORD])},
            {"name": "empty", "hex": ""}])
        for name in ("noret", "nofence", "empty"):
            self.assertTrue(res[name].startswith("ERR"), name)

    def test_trailing_bytes_after_return(self):
        words = ge.packet_words((0, 1, 2)) + [FENCE_WORD, RET_WORD]
        res = lean_decode_batch([
            {"name": "trailjunk", "hex": words_to_hex(words) + "00000000"},
            {"name": "trailpart", "hex": words_to_hex(words) + "00"},
            {"name": "extrainst", "hex": words_to_hex(words + [FENCE_WORD])}])
        for name in ("trailjunk", "trailpart", "extrainst"):
            self.assertTrue(res[name].startswith("ERR"), name)

    # -- host register discipline -----------------------------------------
    def test_write_outside_scratch_rejected(self):
        """LUI to x10 (a caller register) must be rejected by the decoder."""
        lui_x10 = (0xFFFFF << 12) | (10 << 7) | 0x37
        words = ge.packet_words((0, 1, 2))[:8] + [lui_x10] + \
            ge.packet_words((0, 1, 2))[8:] + [FENCE_WORD, RET_WORD]
        res = lean_decode_batch([{"name": "x10", "hex": words_to_hex(words)}])
        self.assertTrue(res["x10"].startswith("ERR"))

    def test_shift_amount_field_must_be_zero_extended(self):
        """RV64 SLLI requires imm[11:6] == 0; nonzero bits must be rejected."""
        li = ge.packet_words((0, 1, 2))
        bad_slli = (li[2] | (1 << 26))     # imm[11:6] != 0
        words = li[:2] + [bad_slli] + li[3:] + [FENCE_WORD, RET_WORD]
        res = lean_decode_batch([
            {"name": "shval", "hex": words_to_hex(words)},
            {"name": "shok", "hex": words_to_hex(li + [FENCE_WORD, RET_WORD])}])
        self.assertTrue(res["shval"].startswith("ERR"))
        self.assertTrue(res["shok"].startswith("OK"))

    # -- buffer bases -------------------------------------------------------
    def test_base_aliasing_and_overflow(self):
        good = self.enc16["bytes_hex"]
        res = lean_decode_batch([
            {"name": "aliasAB", "hex": good,
             "bases": (0x10000000, 0x10000008, 0x30000000)},
            {"name": "aliasAC", "hex": good,
             "bases": (0x10000000, 0x20000000, 0x10000008),
             "sizes": (16, 16, 16)},
            {"name": "overflow", "hex": good,
             "bases": (ge.U64 - 8, 0x20000000, 0x30000000)},
            {"name": "zerosize", "hex": good, "sizes": (0, 16, 16)},
            {"name": "unaligned", "hex": good,
             "bases": (0x10000002, 0x20000000, 0x30000000)}])
        for name in ("aliasAB", "aliasAC", "overflow", "zerosize", "unaligned"):
            self.assertTrue(res[name].startswith("ERR"),
                            "%s not rejected: %s" % (name, res[name]))

    # -- mutation detection --------------------------------------------------
    def test_immediate_bit_mutation_changes_packets(self):
        """A flipped immediate bit still executes (valid RV64) but must NOT
        reproduce the expected packets: acceptance compares real values."""
        words = list(ge.packet_words((0, 0x1234, 2))) + [FENCE_WORD, RET_WORD]
        mutated = list(words)
        mutated[0] ^= (1 << 17)            # change LUI imm feeding x5
        res = lean_decode_batch([
            {"name": "good", "hex": words_to_hex(words)},
            {"name": "mut", "hex": words_to_hex(mutated)}])
        good = parse_ok(res["good"])
        mut = parse_ok(res["mut"])
        self.assertEqual(good, [(0, 0x1234, 2)])
        self.assertEqual(len(mut), 1)
        self.assertNotEqual(mut, good)
        self.assertTrue(res["mut"].startswith("OK"))  # valid RV64, wrong value

    def test_emit_encoding_matches_declared_schema(self):
        e = self.enc16
        self.assertEqual(e["format"], "veritac_gemmini_bytes_v2")
        self.assertEqual(e["word_size"], 4)
        self.assertEqual(e["endianness"], "little")
        self.assertEqual(e["regs"], {"rs1": 5, "rs2": 6, "temp": 31})
        self.assertEqual(len(e["bytes_hex"]) % 8, 0)
        json.dumps(e)                      # must be JSON-serializable


if __name__ == "__main__":
    unittest.main()
