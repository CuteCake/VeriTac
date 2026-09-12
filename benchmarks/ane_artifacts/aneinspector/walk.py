"""Explicit-input walker with symlink-escape rejection.

Only reads within the explicit input root.  Any symlink that resolves outside
the root is rejected (reported as a security note, not followed).  Directory
symlinks that stay inside the root are traversed; those outside are skipped.
Never globs user/system caches -- only the explicit directory is walked.
"""

import os


def _within(target_real, root_real):
    target_real = os.path.realpath(target_real)
    try:
        return os.path.commonpath([target_real, root_real]) == root_real
    except ValueError:
        return False


def collect(root):
    """Return (files, security_notes, root_real).  Raises ValueError if the
    root does not exist.  ``root`` may be a single file or a directory."""
    if not os.path.exists(root):
        raise ValueError("input path does not exist: %s" % root)

    root_real = os.path.realpath(root)
    files = []
    notes = []

    if os.path.isfile(root):
        # An explicitly named file is the input root itself.
        return [root], notes, root_real

    if os.path.islink(root) and not os.path.isdir(root):
        raise ValueError("input root is a symlink to a non-directory")

    # Manual recursive walk so we can reject escaping symlinks.
    def _walk(directory):
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError as exc:
            notes.append("cannot read directory %s: %s" % (directory, exc))
            return
        for entry in entries:
            entry_path = entry.path
            try:
                is_link = entry.is_symlink()
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                continue
            real = os.path.realpath(entry_path)
            if not _within(real, root_real):
                notes.append(
                    "rejected symlink escaping input root: %s -> %s"
                    % (entry_path, real))
                continue
            if is_dir:
                _walk(entry_path)
            elif is_file:
                files.append(entry_path)
            # non-regular files (sockets, devices, FIFOs) are ignored

    if os.path.isdir(root):
        _walk(root)
    return files, notes, root_real
