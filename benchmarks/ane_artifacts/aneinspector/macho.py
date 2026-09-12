"""Bounded generic Mach-O / HWX container parsing.

The engine hardware container is Mach-O-shaped with magic ``0xbeefface``
(on-disk little-endian bytes ``CE FA EF BE``).  This module parses only the
container shell: magic, header, load-command census, and (where a command is
self-consistent with the standard ABI) segment / section metadata and a
symbol table.

It deliberately does NOT interpret register-write / task-descriptor opcodes.
Field offsets, register addresses, and command layouts are silicon-generation
specific (M1 v10 / H13 values are NOT assumed valid on the M3 Ultra), so any
load command that is not confidently a standard Mach-O command is preserved
as raw type / offset / size / hash.  For HWX even the standard-numbered
commands are treated as *provisional* generic interpretations, because the
HWX command ABI is not guaranteed by type number and size alone.

Bounds are validated throughout: header must fit, ncmds*8 <= sizeofcmds,
each known command meets its own minimum cmdsize, segments/sections that are
file-backed must lie within the file (true zero-fill sections are allowed),
and section counts must fit inside their command.
"""

import hashlib
import struct

from .reader import ReadLimitExceeded, read_bounded

# Magic word detected by reading the first 4 bytes as a little-endian u32.
# Each entry: (LE u32, canonical name, variant, cpu width, endian, header len).
_MAGIC_BY_LE = {
    0xFEEDFACE: ("MH_MAGIC", "macho32", 32, "le", 28),
    0xFEEDFACF: ("MH_MAGIC_64", "macho64", 64, "le", 32),
    0xCEFAEDFE: ("MH_CIGAM", "macho_be32", 32, "be", 28),
    0xCFFAEDFE: ("MH_CIGAM_64", "macho_be64", 64, "be", 32),
    0xBEEFFACE: ("HWX_ENGINE_EXEC", "hwx", 32, "le", 32),
    0xBEBAFECA: ("FAT_MAGIC", "fat", None, "be", 8),
    0xBFBAFECA: ("FAT_MAGIC_64", "fat", None, "be", 8),
}

# Load command types we classify.  Names are from the public Mach-O ABI.
LC_SEGMENT = 0x1
LC_SYMTAB = 0x2
LC_DYSYMTAB = 0xB
LC_LOAD_DYLIB = 0xC
LC_ID_DYLIB = 0xD
LC_UUID = 0x1B
LC_SEGMENT_64 = 0x19
LC_SOURCE_VERSION = 0x2A
LC_BUILD_VERSION = 0x32
LC_VERSION_MIN_MACOSX = 0x24
LC_VERSION_MIN_IPHONEOS = 0x25

LC_NAME = {
    LC_SEGMENT: "LC_SEGMENT",
    LC_SEGMENT_64: "LC_SEGMENT_64",
    LC_SYMTAB: "LC_SYMTAB",
    LC_DYSYMTAB: "LC_DYSYMTAB",
    LC_LOAD_DYLIB: "LC_LOAD_DYLIB",
    LC_ID_DYLIB: "LC_ID_DYLIB",
    LC_UUID: "LC_UUID",
    LC_SOURCE_VERSION: "LC_SOURCE_VERSION",
    LC_BUILD_VERSION: "LC_BUILD_VERSION",
    LC_VERSION_MIN_MACOSX: "LC_VERSION_MIN_MACOSX",
    LC_VERSION_MIN_IPHONEOS: "LC_VERSION_MIN_IPHONEOS",
}

# Minimum self-consistent cmdsize per known command, so a short or forged
# command cannot borrow bytes from its successor.
LC_MIN_SIZE = {
    LC_SEGMENT: 56,
    LC_SEGMENT_64: 72,
    LC_SYMTAB: 24,
    LC_DYSYMTAB: 80,
    LC_LOAD_DYLIB: 24,
    LC_ID_DYLIB: 24,
    LC_UUID: 24,
    LC_SOURCE_VERSION: 16,
    LC_BUILD_VERSION: 24,
    LC_VERSION_MIN_MACOSX: 16,
    LC_VERSION_MIN_IPHONEOS: 16,
}

# Structural bounds (defense in depth against malicious headers).
MAX_NCMDS = 1 << 20
MAX_SIZEOFCMDS = 64 << 20  # 64 MiB
MAX_SECTIONS = 1 << 18
MAX_SEGMENT_FILESIZE = 1 << 34  # 16 GiB guard

# Section type is the low 8 bits of the section flags.  Zero-fill sections
# carry a nonzero *virtual* size but occupy no file bytes; they must not be
# treated as file-backed.
SECTION_TYPE_MASK = 0xFF
S_ZEROFILL = 0x1
S_GB_ZEROFILL = 0xC
S_THREAD_LOCAL_ZEROFILL = 0x12
ZEROFILL_TYPES = (S_ZEROFILL, S_GB_ZEROFILL, S_THREAD_LOCAL_ZEROFILL)


class ParseError(Exception):
    """Structured container parse failure.  .reason is a stable token."""

    def __init__(self, reason, detail=""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


class _View:
    """Bounded little/big-endian reader over a bytes region."""

    def __init__(self, data, endian):
        self.data = data
        self.prefix = "<" if endian == "le" else ">"

    def _check(self, off, size):
        if off < 0 or off + size > len(self.data):
            raise ParseError("out_of_bounds", "read past end of region")

    def u32(self, off):
        self._check(off, 4)
        return struct.unpack(self.prefix + "I", self.data[off:off + 4])[0]

    def i32(self, off):
        self._check(off, 4)
        return struct.unpack(self.prefix + "i", self.data[off:off + 4])[0]

    def u64(self, off):
        self._check(off, 8)
        return struct.unpack(self.prefix + "Q", self.data[off:off + 8])[0]

    def name(self, off, length):
        self._check(off, length)
        raw = self.data[off:off + length]
        return raw.split(b"\x00", 1)[0].decode("ascii", "replace")


def detect_magic(fh, file_len):
    """Return (name, variant, width, endian, header_len) or (None,)*5."""
    if file_len < 4:
        return None, None, None, None, None
    data, _ = read_bounded(fh, 0, 4, 4)
    if data is None:
        return None, None, None, None, None
    le = struct.unpack("<I", data)[0]
    return _MAGIC_BY_LE.get(le, (None,) * 5)


def parse_header(fh, file_len, magic_info):
    name, variant, width, endian, hdr_len = magic_info
    if file_len < hdr_len:
        raise ParseError("truncated", "file shorter than container header")
    data, _ = read_bounded(fh, 0, hdr_len, hdr_len)
    if data is None:
        raise ParseError("truncated", "cannot read container header")
    view = _View(data, endian)

    header = {
        "magic_name": name,
        "variant": variant,
        "endian": endian,
        "header_width": width,
        "header_length": hdr_len,
        "cputype": view.i32(4),
        "cpusubtype": view.i32(8),
        "filetype": view.u32(12),
        "ncmds": view.u32(16),
        "sizeofcmds": view.u32(20),
        "flags": view.u32(24),
    }
    if hdr_len >= 32:
        header["reserved"] = view.u32(28)

    ncmds = header["ncmds"]
    sizeofcmds = header["sizeofcmds"]

    if ncmds > MAX_NCMDS:
        raise ParseError("count_overflow", "ncmds=%d exceeds bound" % ncmds)
    if sizeofcmds > MAX_SIZEOFCMDS:
        raise ParseError("count_overflow",
                         "sizeofcmds=%d exceeds bound" % sizeofcmds)
    # Every load command is >= 8 bytes, so ncmds*8 must fit in sizeofcmds.
    if ncmds * 8 > sizeofcmds:
        raise ParseError("count_overflow",
                         "ncmds=%d inconsistent with sizeofcmds=%d"
                         % (ncmds, sizeofcmds))
    if hdr_len + sizeofcmds > file_len:
        raise ParseError("truncated", "load-command region exceeds file length")
    return header


def _parse_segment(view, cmd_off, is_64):
    seg = {"segname": view.name(cmd_off + 8, 16)}
    if is_64:
        seg.update(vmaddr=view.u64(cmd_off + 24), vmsize=view.u64(cmd_off + 32),
                   fileoff=view.u64(cmd_off + 40), filesize=view.u64(cmd_off + 48),
                   maxprot=view.i32(cmd_off + 56), initprot=view.i32(cmd_off + 60),
                   nsects=view.u32(cmd_off + 64), flags=view.u32(cmd_off + 68))
        sect_hdr_len = 80
    else:
        seg.update(vmaddr=view.u32(cmd_off + 24), vmsize=view.u32(cmd_off + 28),
                   fileoff=view.u32(cmd_off + 32), filesize=view.u32(cmd_off + 36),
                   maxprot=view.i32(cmd_off + 40), initprot=view.i32(cmd_off + 44),
                   nsects=view.u32(cmd_off + 48), flags=view.u32(cmd_off + 52))
        sect_hdr_len = 68
    if seg["nsects"] > MAX_SECTIONS:
        raise ParseError("count_overflow", "nsects=%d exceeds bound" % seg["nsects"])
    if seg["filesize"] > MAX_SEGMENT_FILESIZE:
        raise ParseError("count_overflow", "segment %r filesize=%d exceeds guard"
                         % (seg["segname"], seg["filesize"]))
    return seg, sect_hdr_len


def _parse_section(view, off, is_64):
    sect = {"sectname": view.name(off, 16), "segname": view.name(off + 16, 16)}
    if is_64:
        sect.update(addr=view.u64(off + 32), size=view.u64(off + 40),
                    offset=view.u32(off + 48), align=view.u32(off + 52),
                    reloff=view.u32(off + 56), nreloc=view.u32(off + 60),
                    flags=view.u32(off + 64), reserved1=view.u32(off + 68),
                    reserved2=view.u32(off + 72), reserved3=view.u32(off + 76))
    else:
        sect.update(addr=view.u32(off + 32), size=view.u32(off + 36),
                    offset=view.u32(off + 40), align=view.u32(off + 44),
                    reloff=view.u32(off + 48), nreloc=view.u32(off + 52),
                    flags=view.u32(off + 56), reserved1=view.u32(off + 60),
                    reserved2=view.u32(off + 64))
    return sect


def _parse_symtab(view, cmd_off):
    return {"symoff": view.u32(cmd_off + 8), "nsyms": view.u32(cmd_off + 12),
            "stroff": view.u32(cmd_off + 16), "strsize": view.u32(cmd_off + 20)}


def _parse_build_version(view, cmd_off):
    return {"platform": view.u32(cmd_off + 8), "minos": view.u32(cmd_off + 12),
            "sdk": view.u32(cmd_off + 16), "ntools": view.u32(cmd_off + 20)}


def _validate_file_backed(file_len, name, fileoff, filesize):
    """Allow true zero-fill (filesize==0) regions; require file-backed
    regions to lie within the file.  Callers must NOT call this for virtual
    sections (zero-fill section type, or a section of a zero-fill segment)."""
    if fileoff < 0 or filesize < 0:
        raise ParseError("out_of_bounds", "%s negative offset/size" % name)
    if filesize == 0:
        return
    if fileoff + filesize > file_len:
        raise ParseError("out_of_bounds",
                         "%s file range %d..%d exceeds file length %d"
                         % (name, fileoff, fileoff + filesize, file_len))


def parse_container(fh, file_len, magic_info):
    """Parse the Mach-O / HWX shell.  Raises ParseError on malformed input.
    Returns a container dict (for macho/hwx) or a fat-header dict."""
    name, variant, width, endian, hdr_len = magic_info

    if variant == "fat":
        # Fat binaries are containers of arch slices; we only report the
        # header census (magic + nfat_arch), not parse slices.
        if file_len < 8:
            raise ParseError("truncated", "fat header shorter than 8 bytes")
        data, _ = read_bounded(fh, 0, 8, 8)
        if data is None:
            raise ParseError("truncated", "cannot read fat header")
        view = _View(data, endian)
        return {"header": {
                    "magic_name": name, "variant": variant, "endian": endian,
                    "header_length": hdr_len,
                },
                "nfat_arch": view.u32(4),
                "note": "fat container; arch slices not parsed"}

    header = parse_header(fh, file_len, magic_info)

    is_64 = width == 64
    sizeofcmds = header["sizeofcmds"]
    region, region_capped = read_bounded(fh, hdr_len, sizeofcmds, MAX_SIZEOFCMDS)
    if region is None:
        raise ParseError("truncated", "cannot read load-command region")
    view = _View(region, endian)

    provisional = variant.startswith("hwx")
    commands = []
    segments = []
    sections = []
    symtab = None
    build_version = None
    offset = 0
    ncmds_parsed = 0
    region_truncated = region_capped or len(region) < sizeofcmds

    for i in range(header["ncmds"]):
        if offset + 8 > len(region):
            if region_capped:
                break
            raise ParseError("truncated", "load command %d beyond region" % i)
        cmd = view.u32(offset)
        cmdsize = view.u32(offset + 4)
        if cmdsize < 8:
            raise ParseError("malformed",
                             "load command %d has cmdsize=%d < 8" % (i, cmdsize))
        if offset + cmdsize > len(region):
            if region_capped:
                break
            raise ParseError("out_of_bounds",
                             "load command %d extends past command region" % i)
        if offset + cmdsize > sizeofcmds:
            raise ParseError("out_of_bounds",
                             "load command %d exceeds sizeofcmds" % i)

        cmd_bytes = region[offset:offset + cmdsize]
        body = {
            "type": cmd,
            "name": LC_NAME.get(cmd, "UNKNOWN"),
            "offset": hdr_len + offset,
            "size": cmdsize,
            "sha256": hashlib.sha256(cmd_bytes).hexdigest(),
        }

        # Only apply a generic interpretation when the command number maps to
        # a known ABI AND meets that command's own minimum cmdsize.  For HWX
        # the result is explicitly provisional.
        if cmd in LC_MIN_SIZE:
            if cmdsize < LC_MIN_SIZE[cmd]:
                raise ParseError("malformed",
                                 "load command %d (%s) cmdsize=%d below min %d"
                                 % (i, LC_NAME.get(cmd, cmd), cmdsize,
                                    LC_MIN_SIZE[cmd]))
            if cmd == LC_SEGMENT or cmd == LC_SEGMENT_64:
                seg_cmd = cmd == LC_SEGMENT_64
                seg_hdr = 72 if seg_cmd else 56
                seg, sect_hdr_len = _parse_segment(view, offset, seg_cmd)
                nsects = seg["nsects"]
                if seg_hdr + nsects * sect_hdr_len > cmdsize:
                    raise ParseError("count_overflow",
                                     "segment %r nsects=%d exceeds command size"
                                     % (seg["segname"], nsects))
                _validate_file_backed(file_len, "segment %r" % seg["segname"],
                                      seg["fileoff"], seg["filesize"])
                if seg["filesize"] == 0:
                    seg["virtual"] = True
                body["segment"] = seg
                if provisional:
                    body["interpretation"] = "provisional"
                segments.append(seg)
                s_off = offset + seg_hdr
                for _ in range(nsects):
                    sect = _parse_section(view, s_off, seg_cmd)
                    sect_type = sect["flags"] & SECTION_TYPE_MASK
                    sect["section_type"] = sect_type
                    # HWX uses virtual tensor windows. Ordinary Mach-O
                    # sections require an actual zero-fill section type.
                    virtual = (sect_type in ZEROFILL_TYPES) or (
                        provisional and seg["filesize"] == 0)
                    if virtual:
                        sect["virtual"] = True
                    else:
                        _validate_file_backed(
                            file_len, "section %r" % sect["sectname"],
                            sect["offset"], sect["size"])
                    sections.append(sect)
                    s_off += sect_hdr_len
            elif cmd == LC_SYMTAB:
                symtab = _parse_symtab(view, offset)
                body["symtab"] = symtab
                if provisional:
                    body["interpretation"] = "provisional"
            elif cmd == LC_BUILD_VERSION:
                build_version = _parse_build_version(view, offset)
                body["build_version"] = build_version
                if provisional:
                    body["interpretation"] = "provisional"

        commands.append(body)
        ncmds_parsed += 1
        offset += cmdsize

    result = {
        "header": header,
        "load_commands": commands,
        "load_command_count": ncmds_parsed,
        "region_truncated": region_truncated,
        "segments": segments,
        "sections": sections,
        "symtab": symtab,
        "build_version": build_version,
    }
    if provisional:
        result["interpretation"] = {
            "level": "provisional",
            "note": ("HWX load-command ABI is not guaranteed by type number "
                     "and size; generic Mach-O-shaped interpretations are "
                     "provisional and carry no semantic claim."),
        }
    return result


def extract_symbol_names(fh, file_len, container, cap=8 << 20):
    """Read bounded strings from the symtab string table, if present."""
    st = container.get("symtab")
    if not st:
        return []
    stroff = st["stroff"]
    strsize = st["strsize"]
    if stroff < 0 or strsize < 0 or stroff + strsize > file_len:
        return []
    data, _ = read_bounded(fh, stroff, strsize, cap)
    if data is None:
        return []
    blob = data
    names = []
    pos = 0
    while pos < len(blob):
        end = blob.find(b"\x00", pos)
        if end == -1:
            break
        raw = blob[pos:end]
        pos = end + 1
        if raw:
            names.append(raw.decode("utf-8", "replace"))
    return names
