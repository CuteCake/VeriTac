"""Builders for synthetic Mach-O / HWX fixtures used by the tests.

These construct real byte layouts (headers, load commands, segments,
sections, symbol string tables) so the inspector can be exercised against
valid and malformed inputs without any real ANE artifacts.
"""

import struct

MAGIC_HWX = 0xBEEFFACE
MAGIC_MH64 = 0xFEEDFACF
LC_SEGMENT_64 = 0x19
LC_SEGMENT = 0x1
LC_SYMTAB = 0x2
UNKNOWN_CMD = 0x77777777

HDR_32 = 28
HDR_64 = 32
HDR_HWX = 32


def _name(s, n=16):
    return s.encode("ascii", "replace").ljust(n, b"\x00")[:n]


def build_header(magic, cputype, cpusubtype, filetype, ncmds, sizeofcmds,
                 flags, width=32, reserved=0):
    """Standard Mach-O header: 28 bytes for 32-bit, 32 for 64-bit."""
    h = struct.pack("<IIiiII", magic, cputype, cpusubtype, filetype,
                    ncmds, sizeofcmds)
    h += struct.pack("<I", flags)
    if width == 64:
        h += struct.pack("<I", reserved)
    return h


def build_hwx_header(cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags,
                     reserved=0):
    """HWX shell header is documented as 32 bytes: magic, cputype, cpusubtype,
    filetype, ncmds, sizeofcmds, flags, and a trailing reserved word."""
    h = struct.pack("<IIiiII", MAGIC_HWX, cputype, cpusubtype, filetype,
                    ncmds, sizeofcmds)
    h += struct.pack("<II", flags, reserved)
    return h


def build_section_64(sectname, segname, addr, size, offset, align=0,
                     reloff=0, nreloc=0, flags=0):
    b = _name(sectname) + _name(segname)
    b += struct.pack("<QQIIIII", addr, size, offset, align, reloff, nreloc, flags)
    b += struct.pack("<III", 0, 0, 0)
    return b


def build_section_32(sectname, segname, addr, size, offset, align=0,
                     reloff=0, nreloc=0, flags=0):
    b = _name(sectname) + _name(segname)
    b += struct.pack("<IIIIIII", addr, size, offset, align, reloff, nreloc, flags)
    b += struct.pack("<II", 0, 0)
    return b


def build_segment_64(segname, vmaddr, vmsize, fileoff, filesize, nsects,
                     sections, maxprot=7, initprot=5, flags=0):
    body = _name(segname)
    body += struct.pack("<QQQQ", vmaddr, vmsize, fileoff, filesize)
    body += struct.pack("<iiII", maxprot, initprot, nsects, flags)
    for s in sections:
        body += s
    cmd = LC_SEGMENT_64
    return struct.pack("<II", cmd, 8 + len(body)) + body


def build_unknown_cmd(cmd_type=UNKNOWN_CMD, payload=b"\xde\xad\xbe\xef"):
    body = struct.pack("<I", cmd_type) + payload
    return struct.pack("<II", cmd_type, 8 + len(body)) + body


def build_symtab(symoff, nsyms, stroff, strsize):
    return struct.pack("<IIIIII", LC_SYMTAB, 24, symoff, nsyms, stroff, strsize)


def _assemble(header, commands, trailing=b""):
    return header + b"".join(commands) + trailing


def build_valid_hwx():
    """A synthetic, structurally valid HWX container with:
    - __TEXT segment + one __text section holding a build-info banner
    - __KERN_0 segment with weight bytes (file-backed)
    - __FVMLIB-style window segment (fileoff=0, filesize=0: virtual only)
    - one unknown load command (must be preserved raw)
    - a symbol string table with names
    """
    text_content = (b"zin_ane_compiler v9.509.0 -t h15g "
                    b"--fl2-cache-mode=resident\n"
                    b"com.apple.ANECompilerFramework\n")
    kern_content = b"\x00" * 64 + b"\x3c\x00" * 16
    str_table = (b"\x00" + b"main_ane\x00" + b"t0_ane\x00" +
                 b"K596A4B73_ne_0\x00" + b"t5_ane\x00")

    # Pass 1: build commands with placeholder fileoff to learn sizeofcmds.
    placeholder = 0
    seg_text_dummy = build_segment_64(
        "__TEXT", 0x30010000, 0x1000, placeholder, len(text_content), 1,
        [build_section_64("__text", "__TEXT", 0x30010000, len(text_content),
                          placeholder)])
    seg_kern_dummy = build_segment_64(
        "__KERN_0", 0x30020000, 0x2000, placeholder, len(kern_content), 1,
        [build_section_64("__kern_0", "__KERN_0", 0x30020000, len(kern_content),
                          placeholder)])
    seg_win_dummy = build_segment_64(
        "__FVMLIB", 0x30008000, 0x80, 0, 0, 0, [], maxprot=5, initprot=1)
    unknown = build_unknown_cmd()
    symtab_dummy = build_symtab(0, 0, placeholder, len(str_table))
    commands_dummy = [seg_text_dummy, seg_kern_dummy, seg_win_dummy,
                      unknown, symtab_dummy]
    sizeofcmds = sum(len(c) for c in commands_dummy)

    content_off = HDR_HWX + sizeofcmds
    str_off = content_off + len(text_content) + len(kern_content)

    # Pass 2: rebuild with real file offsets.
    seg_text = build_segment_64(
        "__TEXT", 0x30010000, 0x1000, content_off, len(text_content), 1,
        [build_section_64("__text", "__TEXT", 0x30010000, len(text_content),
                          content_off)])
    seg_kern = build_segment_64(
        "__KERN_0", 0x30020000, 0x2000,
        content_off + len(text_content), len(kern_content), 1,
        [build_section_64("__kern_0", "__KERN_0", 0x30020000, len(kern_content),
                          content_off + len(text_content))])
    seg_win = build_segment_64(
        "__FVMLIB", 0x30008000, 0x80, 0, 0, 0, [], maxprot=5, initprot=1)
    symtab = build_symtab(0, 0, str_off, len(str_table))
    commands = [seg_text, seg_kern, seg_win, unknown, symtab]
    header = build_hwx_header(0x80, 0x4, 2, len(commands), sizeofcmds,
                              0x200000)
    blob = header + b"".join(commands) + text_content + kern_content + str_table
    return blob


def build_macho64():
    """A plain 64-bit Mach-O (MH_MAGIC_64) with a __TEXT segment + section."""
    text_content = b"\x00\x01\x02\x03" * 4
    content_off = HDR_64
    seg = build_segment_64(
        "__TEXT", 0x100000000, 0x2000, content_off, len(text_content), 1,
        [build_section_64("__text", "__TEXT", 0x100000000, len(text_content),
                          content_off)])
    header = build_header(MAGIC_MH64, 0x01000007, 3, 2, 1, len(seg),
                          0x2000, width=64, reserved=0)
    return header + seg + text_content


def build_truncated():
    """Valid header, but the load-command region extends past EOF."""
    header = build_hwx_header(0x80, 0x4, 2, 3, 200, 0x200000)
    # only 8 bytes of commands present; sizeofcmds=200 exceeds file
    return header + b"\x19\x00\x00\x00\x48\x00\x00\x00"


def build_out_of_bounds():
    """A segment command declaring nsects=5 but carrying only one section, so
    the section count does not fit inside the command (count overflow)."""
    sect = build_section_32("__text", "__TEXT", 0, 64, 0)
    segname = _name("__TEXT")
    # vmaddr, vmsize, fileoff, filesize, maxprot, initprot, nsects=5, flags
    segbody = segname + struct.pack("<IIIIIIII", 0, 0x1000, 0, 64, 7, 5, 5, 0)
    body = segbody + sect
    cmd = struct.pack("<II", LC_SEGMENT, 8 + len(body)) + body
    header = build_hwx_header(0x80, 0x4, 2, 1, len(cmd), 0x200000)
    return header + cmd


def build_count_overflow():
    """Header whose ncmds is wildly inconsistent with sizeofcmds."""
    header = build_hwx_header(0x80, 0x4, 2, 0xFFFFF000, 32, 0x200000)
    return header + b"\x00" * 32


def build_fat():
    """A fat header (FAT_MAGIC 0xCAFEBABE, on-disk bytes CA FE BA BE) with 2
    slices.  The magic is stored big-endian; reading it little-endian yields
    0xBEBAFECA, which the inspector must map to FAT_MAGIC, not MH_CIGAM."""
    return struct.pack(">I", 0xCAFEBABE) + struct.pack(">I", 2)


def build_big_endian_macho32():
    """MH_CIGAM big-endian 32-bit Mach-O (must not be mistaken for HWX).
    The on-disk magic bytes are FE ED FA CE."""
    header = b"\xfe\xed\xfa\xce"  # MH_CIGAM literal bytes
    header += struct.pack(">iiIIII", 0x80, 0x4, 2, 1, 56, 0x200000)
    segname = _name("__TEXT")
    # vmaddr, vmsize, fileoff, filesize, maxprot, initprot, nsects=0, flags
    segbody = segname + struct.pack(">IIIIIIII", 0, 0x1000, 0, 64, 7, 5, 0, 0)
    cmd = struct.pack(">II", LC_SEGMENT, 8 + len(segbody)) + segbody
    return header + cmd + b"\x00" * 64


def build_short_known_command():
    """A known command (LC_SYMTAB) whose cmdsize is below its minimum, so it
    cannot be a self-consistent symtab (would borrow from the next command)."""
    # LC_SYMTAB with cmdsize=8 (header only) -- below min 24.
    cmd = struct.pack("<II", LC_SYMTAB, 8)
    header = build_hwx_header(0x80, 0x4, 2, 1, len(cmd), 0x200000)
    return header + cmd


S_ZEROFILL = 0x1

def build_zerofill_macho64():
    """A valid 64-bit Mach-O whose __DATA segment has filesize 0 and holds a
    __bss section of nonzero *virtual* size with the S_ZEROFILL section type.
    The inspector must not treat the virtual size as file bytes."""
    vm = 0x100000000
    seg = build_segment_64(
        "__DATA", vm, 0x100000, 0, 0, 1,
        [build_section_64("__bss", "__DATA", vm, 0x100000, 0,
                          flags=S_ZEROFILL)],
        maxprot=7, initprot=3)
    # The S_ZEROFILL section occupies virtual memory only.
    header = build_header(MAGIC_MH64, 0x01000007, 3, 2, 1, len(seg), 0x2000,
                          width=64, reserved=0)
    return header + seg


def build_bad_regular_section():
    """A 64-bit Mach-O whose regular (non-zero-fill) __text section declares a
    file range beyond EOF; this must be rejected as out_of_bounds."""
    text_content = b"\x90\x90\x90\x90"
    content_off = HDR_64
    seg = build_segment_64(
        "__TEXT", 0x100000000, 0x2000, content_off, len(text_content), 1,
        [build_section_64("__text", "__TEXT", 0x100000000, 0x2000,
                          0x100000, flags=0)],  # offset beyond EOF
        maxprot=7, initprot=5)
    header = build_header(MAGIC_MH64, 0x01000007, 3, 2, 1, len(seg), 0x2000,
                          width=64, reserved=0)
    return header + seg + text_content
