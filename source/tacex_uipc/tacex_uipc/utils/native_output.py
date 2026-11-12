"""Isolation of native (C/C++) libraries that write to the console and the working directory.

Some native libraries (e.g. wildmeshing/fTetWild) print to stdout/stderr regardless of their log level and write debug
files (``__tracked_surface.stl``) into the current working directory. Python-level redirection
(``contextlib.redirect_stdout``) does not catch native output, so file descriptors 1/2 are redirected instead.

Only standard library code, so it can be used/tested without Isaac Sim.
"""

from __future__ import annotations

import contextlib
import ctypes
import datetime
import os
import pathlib
import shutil
import sys
import tempfile

_LIBC = ctypes.CDLL(None)


def _flush_all():
    """Flushes Python and C stdio buffers (C++ ``std::cout`` is synced with C stdio by default)."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    with contextlib.suppress(Exception):
        _LIBC.fflush(None)


@contextlib.contextmanager
def isolate_native_output(name: str, log_dir: str | None = None, tail_on_error: int = 30):
    """Runs the enclosed code in a temporary working directory with stdout/stderr redirected to a log file.

    On exit, the working directory and the console are restored and the temporary directory (incl. files the native
    code wrote into the working directory) is deleted.

    Args:
        name: Name used for the temporary directory and the log file (``<name>.log``).
        log_dir: If given, the log is kept as ``<log_dir>/<name>_<timestamp>.log``. Otherwise it is discarded.
        tail_on_error: Number of trailing log lines printed to stderr if the enclosed code raises.

    Note:
        Changes the process-wide working directory and file descriptors 1/2 while active, so other threads writing
        to the console in the meantime end up in the log. Meant for short blocking calls (e.g. mesh generation during
        scene setup).
    """
    tmp_dir = tempfile.mkdtemp(prefix=f"tacex_{name}_")
    log_path = os.path.join(tmp_dir, f"{name}.log")
    cwd = os.getcwd()

    _flush_all()
    saved_fds = (os.dup(1), os.dup(2))
    log_file = open(log_path, "w")
    failed = False
    try:
        os.dup2(log_file.fileno(), 1)
        os.dup2(log_file.fileno(), 2)
        os.chdir(tmp_dir)
        yield tmp_dir
    except BaseException:
        failed = True
        raise
    finally:
        _flush_all()
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        for fd in saved_fds:
            os.close(fd)
        log_file.close()
        os.chdir(cwd)

        if failed and tail_on_error > 0:
            with contextlib.suppress(Exception):
                lines = pathlib.Path(log_path).read_text(errors="replace").splitlines()[-tail_on_error:]
                print(f"[{name}] last {len(lines)} log lines:\n" + "\n".join(lines), file=sys.stderr, flush=True)
        if log_dir is not None:
            with contextlib.suppress(Exception):
                pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
                stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                shutil.copyfile(log_path, os.path.join(log_dir, f"{name}_{stamp}.log"))
        shutil.rmtree(tmp_dir, ignore_errors=True)
