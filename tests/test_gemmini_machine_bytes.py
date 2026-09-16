"""Independent execution checks for the restricted RV64 kernel body.

This deliberately interprets ordinary RV64 register operations rather than
round-tripping through the production encoder/decoder. It is regression evidence,
not the formal hardware-conformance argument.
"""
import unittest

MASK64 = (1 << 64) - 1


def signed(value, width):
    return value - (1 << width) if value & (1 << (width - 1)) else value


def execute_register_program(code):
    """Return the Gemmini operand packets observed while executing bytes.

    Inputs are byte strings. x1 contains a sentinel return address; all other
    registers start poisoned to detect reliance on unspecified caller values.
    """
    if len(code) % 4:
        raise ValueError("truncated RV64 instruction")
    regs = [0] + [0xDEAD000000000000 + r for r in range(1, 32)]
    packets = []
    returned = False
    for pc in range(0, len(code), 4):
        word = int.from_bytes(code[pc:pc + 4], 'little')
        opcode = word & 127
        rd, f3 = (word >> 7) & 31, (word >> 12) & 7
        r1, r2 = (word >> 15) & 31, (word >> 20) & 31
        imm = signed(word >> 20, 12)
        value = None
        if opcode == 0x13:  # RV64 OP-IMM
            if f3 == 0:
                value = regs[r1] + imm
            elif f3 == 6:
                value = regs[r1] | (imm & MASK64)
            elif f3 == 1 and word >> 26 == 0:
                value = regs[r1] << ((word >> 20) & 63)
            elif f3 == 5 and word >> 26 == 0:
                value = regs[r1] >> ((word >> 20) & 63)
            else:
                raise ValueError("unsupported RV64 OP-IMM")
        elif opcode == 0x33 and f3 == 0 and word >> 25 == 0:  # ADD
            value = regs[r1] + regs[r2]
        elif opcode == 0x37:  # LUI sign extends bit 31 on RV64
            value = signed(word & 0xFFFFF000, 32)
        elif opcode == 0x1B and f3 == 0:  # ADDIW
            value = signed((regs[r1] + imm) & 0xFFFFFFFF, 32)
        elif opcode == 0x7B:  # CUSTOM_3, no destination, both sources enabled
            if rd != 0 or f3 != 3:
                raise ValueError("invalid Gemmini destination/source flags")
            packets.append((word >> 25, regs[r1], regs[r2]))
        elif word == 0x0FF0000F:  # fence iorw,iorw
            packets.append(('fence',))
        elif word == 0x00008067:  # jalr x0,0(x1)
            if pc + 4 != len(code):
                raise ValueError("bytes after return")
            returned = True
        else:
            raise ValueError(f"unsupported instruction {word:#010x}")
        if value is not None and rd != 0:
            regs[rd] = value & MASK64
        regs[0] = 0
    if not returned:
        raise ValueError("missing return")
    return packets


def expected_packets(commands, bases):
    """Pinned gemmini.h macro operands, independent of the byte emitter."""
    def tile(addr, rows=16, cols=16):
        return (rows << 48) | (cols << 32) | addr

    packets = []
    for command in commands:
        c = {'kind': command.kind, **vars(command)} if hasattr(command, 'kind') else command
        kind = c['kind']
        if kind == 'config_ex':
            packets.append((0, (0x3F800000 << 32) | (1 << 16) | (c['dataflow'] << 2), 1 << 48))
        elif kind == 'config_st':
            packets.append((0, 2, (0x3F800000 << 32) | c['stride_bytes']))
        elif kind == 'config_ld':
            packets.append((0, (0x3F800000 << 32) | (16 << 16) | (1 << 8) | (c['slot'] << 3) | 1,
                            c['stride_bytes']))
        elif kind == 'mvin':
            packets.append((2 if c['slot'] == 0 else 1, bases[c['buf']] + c['offset'],
                            tile(c['spad_addr'], c['rows'], c['cols'])))
        elif kind == 'preload':
            packets.append((6, tile(c['bd_spad_addr']), tile(c['out_addr'])))
        elif kind == 'compute':
            packets.append((5 if c['accumulated'] else 4,
                            tile(c['a_spad_addr']), tile(c['bd_spad_addr'])))
        elif kind == 'mvout':
            packets.append((3, bases['C'] + 4 * c['buf_offset'],
                            tile(c['acc_addr'], c['rows'], c['cols'])))
        elif kind == 'fence':
            packets.append(('fence',))
        else:
            raise ValueError('unknown command')
    return packets


class MachineByteExecutionTests(unittest.TestCase):
    def test_generated_program_operands_match_pinned_macros(self):
        from CodeGen import gemmini as g
        from CodeGen import gemmini_encoding as enc
        for shape in ((16, 16, 16), (32, 16, 16), (32, 16, 32), (64, 64, 64)):
            for schedule in ('baseline', 'reuse_b'):
                with self.subTest(shape=shape, schedule=schedule):
                    plan = g.make_plan(*shape, schedule)
                    commands = g.gen_commands(plan)
                    artifact = enc.emit_encoding(plan, commands)
                    actual = execute_register_program(bytes.fromhex(artifact['bytes_hex']))
                    self.assertEqual(actual, expected_packets(commands, artifact['bases']))

    def test_li64_signed_boundaries_and_poisoned_initial_registers(self):
        import random
        from CodeGen import gemmini_encoding as enc
        values = [0, 1, 2047, 2048, 4095, 4096, 0x7FFFFFFF, 0x80000000,
                  0xFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, MASK64]
        rng = random.Random(459017)
        values.extend(rng.getrandbits(64) for _ in range(100))
        for value in values:
            with self.subTest(value=hex(value)):
                words = enc.packet_words((6, value, MASK64 - value)) + [0x0FF0000F, 0x00008067]
                code = b''.join(w.to_bytes(4, 'little') for w in words)
                self.assertEqual(execute_register_program(code),
                                 [(6, value, MASK64 - value), ('fence',)])

    def test_buffer_alias_and_64bit_overflow_are_rejected(self):
        from CodeGen import gemmini as g
        from CodeGen import gemmini_encoding as enc
        plan = g.make_plan(16, 16, 16, 'baseline')
        commands = g.gen_commands(plan)
        for bases in ({'A': 4096, 'B': 4096, 'C': 8192},
                      {'A': (1 << 64) - 4, 'B': 4096, 'C': 8192}):
            with self.assertRaises(ValueError):
                enc.emit_encoding(plan, commands, bases)


if __name__ == '__main__':
    unittest.main()
