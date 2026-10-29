"""Single-key console input on Windows and POSIX; restores terminal on exit."""

import os
import sys
from contextlib import contextmanager


@contextmanager
def keyboard():
    if not sys.stdin.isatty():
        raise RuntimeError("Collection requires an interactive terminal")
    if os.name == "nt":
        import msvcrt

        def read_key():
            return msvcrt.getwch() if msvcrt.kbhit() else ""

        yield read_key
    else:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)

            def read_key():
                return os.read(fd, 1).decode("ascii", errors="ignore") if select.select([fd], [], [], 0)[0] else ""

            yield read_key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
