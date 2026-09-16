"""Untrusted, direct RV64/Gemmini kernel-body emitter.

The resulting bytes require Lean program/byte validation before acceptance.
The caller must map initialized A/B and writable C at the declared nonoverlapping
bases, enable the pinned Gemmini configuration, make code executable, synchronize
the instruction cache, and call it with a valid return address in x1. x5/x6/x31
are clobbered. No C compiler or assembler constructs the kernel body.

Packing follows gemmini.h and gemmini_params.h pinned by CodeGen.gemmini;
RoCC fields follow IBM/rocc-software commit
fddb795a0b52e82f8f4ce9ead9b1428440a62ab0/src/xcustom.h.
"""

U64 = 1 << 64
REGS = {'rs1': 5, 'rs2': 6, 'temp': 31}
DEFAULT_BASES = {'A': 0x10000000, 'B': 0x20000000, 'C': 0x30000000}
FIELDS = {
    'config_ex': {'dataflow'}, 'config_ld': {'slot', 'stride_bytes', 'scale'},
    'config_st': {'stride_bytes'},
    'mvin': {'slot', 'buf', 'offset', 'spad_addr', 'cols', 'rows'},
    'preload': {'bd_spad_addr', 'out_addr'},
    'compute': {'accumulated', 'a_spad_addr', 'bd_spad_addr'},
    'mvout': {'buf_offset', 'acc_addr', 'cols', 'rows'}, 'fence': set(),
}


def nat(value, bits=64):
    if type(value) is not int or not 0 <= value < (1 << bits):
        raise ValueError('operand must be an unsigned %d-bit integer' % bits)
    return value


def command_dict(command):
    c = dict(command) if isinstance(command, dict) else {'kind': command.kind, **vars(command)}
    kind = c.get('kind')
    if kind not in FIELDS or set(c) != FIELDS[kind] | {'kind'}:
        raise ValueError('unknown command or command fields')
    return c


def check_bases(bases, sizes):
    if set(bases) != {'A', 'B', 'C'} or set(sizes) != {'A', 'B', 'C'}:
        raise ValueError('expected exactly A/B/C buffers')
    intervals = []
    for name in ('A', 'B', 'C'):
        start, length = nat(bases[name]), nat(sizes[name])
        if start % 4 or length == 0 or start + length > U64:
            raise ValueError('unaligned, empty, or overflowing buffer range')
        intervals.append((start, start + length))
    for i, (start, end) in enumerate(intervals):
        for other_start, other_end in intervals[i + 1:]:
            if start < other_end and other_start < end:
                raise ValueError('buffer ranges overlap')


def tile(addr, cols=16, rows=16):
    return (nat(rows, 16) << 48) | (nat(cols, 16) << 32) | nat(addr, 32)


def command_packet(command, bases):
    c = command_dict(command)
    kind = c['kind']
    if kind == 'config_ex':
        if c['dataflow'] != 1 or type(c['dataflow']) is not int:
            raise ValueError('only weight-stationary dataflow is supported')
        return 0, (0x3F800000 << 32) | (1 << 16) | 4, 1 << 48
    if kind == 'config_ld':
        if type(c['scale']) not in (int, float) or c['scale'] != 1.0:
            raise ValueError('only identity load scale is supported')
        slot = nat(c['slot'])
        if slot not in (0, 1):
            raise ValueError('unsupported load slot')
        return 0, (0x3F800000 << 32) | (16 << 16) | (1 << 8) | (slot << 3) | 1, nat(c['stride_bytes'])
    if kind == 'config_st':
        return 0, 2, (0x3F800000 << 32) | nat(c['stride_bytes'], 32)
    if kind == 'mvin':
        if c['buf'] not in ('A', 'B') or type(c['slot']) is not int or c['slot'] not in (0, 1):
            raise ValueError('unsupported input/slot')
        return (2 if c['slot'] == 0 else 1), nat(bases[c['buf']] + nat(c['offset'])), tile(c['spad_addr'], c['cols'], c['rows'])
    if kind == 'preload':
        return 6, tile(c['bd_spad_addr']), tile(c['out_addr'])
    if kind == 'compute':
        if type(c['accumulated']) is not bool:
            raise ValueError('accumulated must be a Boolean')
        return (5 if c['accumulated'] else 4), tile(c['a_spad_addr']), tile(c['bd_spad_addr'])
    if kind == 'mvout':
        return 3, nat(bases['C'] + 4 * nat(c['buf_offset'])), tile(c['acc_addr'], c['cols'], c['rows'])
    return None  # fence has its own RV64 opcode


def _addi(rd, rs, immediate):
    return ((immediate & 4095) << 20) | (rs << 15) | (rd << 7) | 0x13


def _shift(rd, rs, count, funct3):
    return (count << 20) | (rs << 15) | (funct3 << 12) | (rd << 7) | 0x13


def li64(rd, value):
    """Eight RV64 instructions; no assembler pseudoinstruction is trusted."""
    value = nat(value)
    if rd not in (5, 6):
        raise ValueError('operand destination must be x5 or x6')
    high, low = value >> 32, value & 0xFFFFFFFF
    return [
        ((((high + 2048) >> 12) & 0xFFFFF) << 12) | (rd << 7) | 0x37,
        _addi(rd, rd, high & 4095), _shift(rd, rd, 32, 1),
        ((((low + 2048) >> 12) & 0xFFFFF) << 12) | (31 << 7) | 0x37,
        _addi(31, 31, low & 4095), _shift(31, 31, 32, 1),
        _shift(31, 31, 32, 5), (31 << 20) | (rd << 15) | (rd << 7) | 0x33,
    ]


def packet_words(packet):
    funct, rs1, rs2 = packet
    if nat(funct) > 6:
        raise ValueError('unsupported Gemmini funct')
    return li64(5, rs1) + li64(6, rs2) + [(funct << 25) | (6 << 20) | (5 << 15) | 0x3000 | 0x7B]


def emit_encoding(plan, commands, bases=None):
    from CodeGen.gemmini import validate_plan
    ok, reason = validate_plan(plan)
    if not ok:
        raise ValueError(reason)
    bases = dict(DEFAULT_BASES if bases is None else bases)
    sizes = {'A': plan['m'] * plan['k'], 'B': plan['k'] * plan['n'], 'C': 4 * plan['m'] * plan['n']}
    check_bases(bases, sizes)
    commands = [command_dict(c) for c in commands]
    if not commands or commands[-1]['kind'] != 'fence':
        raise ValueError('program must end with fence')
    words = []
    for index, c in enumerate(commands):
        packet = command_packet(c, bases)
        if packet is None:
            if index != len(commands) - 1:
                raise ValueError('only a final fence is supported')
            words.append(0x0FF0000F)
        else:
            words.extend(packet_words(packet))
    words.append(0x00008067)
    code = b''.join(word.to_bytes(4, 'little') for word in words)
    return {'format': 'veritac_gemmini_bytes_v2', 'word_size': 4,
            'endianness': 'little', 'regs': dict(REGS), 'bases': bases,
            'sizes': sizes, 'bytes_hex': code.hex()}
