"""Opt-in terminal execution for AI agents running under Termux.

The terminal is deliberately disabled by default because commands can modify the
host system. Enable it explicitly with ``terminal.enable()`` or by passing
``enabled=True`` when constructing a Terminal instance.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional, Sequence, Union

Command = Union[str, Sequence[str]]


class TerminalDisabledError(RuntimeError):
    """Raised when terminal execution is attempted while disabled."""


class TerminalUnavailableError(RuntimeError):
    """Raised when Termux is not available on the current system."""


@dataclass
class TerminalResult:
    command: Command
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class Terminal:
    """Controlled interface to the local Termux shell.

    Safety defaults:
      * disabled=True by default
      * shell=False by default for string commands
      * configurable timeout
      * no automatic sudo/root escalation
      * Termux detection before execution
    """

    def __init__(self, enabled: bool = False, timeout: float = 30.0,
                 require_termux: bool = True):
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        self.enabled = bool(enabled)
        self.timeout = float(timeout)
        self.require_termux = bool(require_termux)

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("TERMUX_VERSION")) or shutil.which("termux-info") is not None

    @staticmethod
    def is_termux() -> bool:
        return Terminal.available() or os.environ.get("PREFIX", "").startswith("/data/data/com.termux")

    def enable(self):
        self.enabled = True
        return self

    def disable(self):
        self.enabled = False
        return self

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "available": self.available(),
            "is_termux": self.is_termux(),
            "platform": platform.platform(),
            "shell": os.environ.get("SHELL"),
        }

    def run(self, command: Command, *, timeout: Optional[float] = None,
            shell: bool = False, cwd: Optional[str] = None,
            env: Optional[dict] = None, check: bool = False) -> TerminalResult:
        if not self.enabled:
            raise TerminalDisabledError("Terminal tool is disabled; call terminal.enable() explicitly")
        if self.require_termux and not self.is_termux():
            raise TerminalUnavailableError("Termux was not detected")
        if not command:
            raise ValueError("command must not be empty")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        if isinstance(command, str):
            args = command if shell else command.split()
        else:
            args = list(command)
            if not args or not all(isinstance(x, str) and x for x in args):
                raise ValueError("command sequence must contain non-empty strings")

        timed_out = False
        try:
            completed = subprocess.run(
                args,
                shell=shell,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout if timeout is None else timeout,
                check=False,
            )
            result = TerminalResult(command, completed.stdout, completed.stderr,
                                    completed.returncode)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes): stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes): stderr = stderr.decode(errors="replace")
            result = TerminalResult(command, stdout, stderr, -1, timed_out=True)

        if check and not result.ok:
            raise subprocess.CalledProcessError(result.returncode, command,
                                                  output=result.stdout,
                                                  stderr=result.stderr)
        return result


terminal = Terminal()
