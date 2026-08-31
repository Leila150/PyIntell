"""Safe, opt-in local terminal execution."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Sequence, Union

Command = Union[str, Sequence[str]]

class TerminalDisabledError(RuntimeError):
    """Raised when terminal execution is disabled."""

class TerminalUnavailableError(RuntimeError):
    """Raised when the requested shell is unavailable."""

@dataclass
class TerminalResult:
    command: Command
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    timed_out: bool = False
    duration: float = 0.0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def as_dict(self):
        return {"command": self.command, "stdout": self.stdout, "stderr": self.stderr,
                "returncode": self.returncode, "timed_out": self.timed_out,
                "duration": self.duration, "ok": self.ok, "truncated": self.truncated}

class Terminal:
    """Controlled interface to the host terminal/shell."""
    def __init__(self, enabled=False, timeout=30.0, require_termux=False,
                 shell_path=None, max_output=1_000_000):
        if timeout <= 0: raise ValueError("timeout must be greater than 0")
        if max_output <= 0: raise ValueError("max_output must be greater than 0")
        self.enabled = bool(enabled)
        self.timeout = float(timeout)
        self.require_termux = bool(require_termux)
        self.shell_path = shell_path
        self.max_output = int(max_output)

    @staticmethod
    def is_termux():
        return (bool(os.environ.get("TERMUX_VERSION")) or
                os.environ.get("PREFIX", "").startswith("/data/data/com.termux") or
                shutil.which("termux-info") is not None)

    @staticmethod
    def available(shell=None):
        if shell: return shutil.which(shell) is not None
        return (shutil.which("bash") is not None or shutil.which("sh") is not None or
                shutil.which("pwsh") is not None or os.name == "nt")

    def enable(self): self.enabled = True; return self
    def disable(self): self.enabled = False; return self

    def configure(self, **values):
        for key, value in values.items():
            if not hasattr(self, key): raise ValueError(f"Unknown terminal option: {key}")
            setattr(self, key, value)
        if self.timeout <= 0 or self.max_output <= 0: raise ValueError("invalid terminal limits")
        return self

    def status(self):
        shell = self.shell_path or os.environ.get("SHELL") or shutil.which("bash") or shutil.which("sh")
        return {"enabled": self.enabled, "available": self.available(), "is_termux": self.is_termux(),
                "platform": platform.platform(), "shell": shell,
                "shell_name": os.path.basename(shell) if shell else None}

    @staticmethod
    def _text(value):
        if value is None: return ""
        if isinstance(value, bytes): return value.decode(errors="replace")
        return str(value)

    def run(self, command: Command, *, timeout=None, shell=False, cwd=None,
            env=None, check=False, stdin=None, max_output=None) -> TerminalResult:
        if not self.enabled: raise TerminalDisabledError("Terminal execution is disabled")
        if self.require_termux and not self.is_termux(): raise TerminalUnavailableError("Termux was not detected")
        if command is None or (isinstance(command, str) and not command.strip()): raise ValueError("command must not be empty")
        limit = self.max_output if max_output is None else int(max_output)
        effective_timeout = self.timeout if timeout is None else float(timeout)
        if limit <= 0 or effective_timeout <= 0: raise ValueError("invalid execution limits")

        if isinstance(command, str):
            if shell:
                args = command
            else:
                args = command.split()
        else:
            args = list(command)
            if not args or not all(isinstance(x, str) and x for x in args): raise ValueError("command sequence must contain non-empty strings")

        run_env = os.environ.copy()
        if env: run_env.update({str(k): str(v) for k, v in env.items()})
        started = time.monotonic()
        try:
            completed = subprocess.run(args, shell=shell, cwd=cwd, env=run_env, input=stdin,
                                       capture_output=True, text=True, timeout=effective_timeout, check=False)
            stdout, stderr = self._text(completed.stdout), self._text(completed.stderr)
            truncated = len(stdout) > limit or len(stderr) > limit
            result = TerminalResult(command, stdout[:limit], stderr[:limit], completed.returncode,
                                    False, time.monotonic() - started, truncated)
        except FileNotFoundError as exc:
            raise TerminalUnavailableError(str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = self._text(exc.stdout), self._text(exc.stderr)
            result = TerminalResult(command, stdout[:limit], stderr[:limit], -1, True,
                                    time.monotonic() - started, len(stdout) > limit or len(stderr) > limit)
        if check and not result.ok:
            raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
        return result

terminal = Terminal()

def _terminal_tool(command: str, **kwargs):
    return terminal.run(command, **kwargs).as_dict()

def _enable_terminal(): terminal.enable()
def _disable_terminal(): terminal.disable()

try:
    from .tools import tool
    tool.add(_terminal_tool, name="terminal",
             description="Execute a local terminal command and return structured output.",
             trusted=True, aliases=("shell_terminal",), category="development",
             dangerous=True, requires_explicit_enable=True)
    entry = tool.get_tool("terminal")
    entry._on_enable = _enable_terminal
    entry._on_disable = _disable_terminal
    tool.disable("terminal")
except Exception:
    pass
