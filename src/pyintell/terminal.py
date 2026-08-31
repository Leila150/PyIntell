"""Safe, opt-in local terminal execution.

Terminal commands are executed through a shell by default when supplied as a
string, which matches normal terminal behavior. Argument sequences remain
shell-free for safer direct process execution. Disabled by default.
"""
from __future__ import annotations

import os
import platform
import shutil
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Union

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
        return {
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration": self.duration,
            "ok": self.ok,
            "truncated": self.truncated,
        }

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

    def _shell(self):
        configured = self.shell_path
        if configured and os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
        return (shutil.which("bash") or shutil.which("sh") or
                shutil.which("pwsh") or (os.environ.get("COMSPEC") if os.name == "nt" else None))

    def enable(self): self.enabled = True; return self
    def disable(self): self.enabled = False; return self

    def configure(self, **values):
        for key, value in values.items():
            if not hasattr(self, key): raise ValueError(f"Unknown terminal option: {key}")
            setattr(self, key, value)
        return self

    def status(self):
        shell = self._shell()
        return {
            "enabled": self.enabled,
            "available": self.available(),
            "is_termux": self.is_termux(),
            "platform": platform.platform(),
            "shell": shell,
            "shell_name": os.path.basename(shell) if shell else None,
        }

    @staticmethod
    def _text(value):
        if value is None: return ""
        if isinstance(value, bytes): return value.decode(errors="replace")
        return str(value)

    def run(self, command: Command, *, timeout=None, shell=None, cwd=None,
            env=None, check=False, stdin=None, max_output=None) -> TerminalResult:
        if not self.enabled:
            raise TerminalDisabledError("Terminal execution is disabled")
        if self.require_termux and not self.is_termux():
            raise TerminalUnavailableError("Termux was not detected")
        if command is None or (isinstance(command, str) and not command.strip()):
            raise ValueError("command must not be empty")
        limit = self.max_output if max_output is None else int(max_output)
        if limit <= 0: raise ValueError("max_output must be greater than 0")
        effective_timeout = self.timeout if timeout is None else float(timeout)
        if effective_timeout <= 0: raise ValueError("timeout must be greater than 0")

        # Normal terminal strings behave like commands typed into a shell.
        # Sequences are direct process invocations unless shell=True is explicit.
        if shell is None:
            shell = isinstance(command, str)

        executable = None
        if shell:
            if isinstance(command, str):
                args = command
            else:
                args = " ".join(shlex.quote(x) for x in command)
            executable = self._shell()
            if executable is None:
                raise TerminalUnavailableError("No usable shell was found on this host")
        else:
            if isinstance(command, str):
                args = shlex.split(command)
                if not args: raise ValueError("command must not be empty")
            else:
                args = list(command)
            if not args or not all(isinstance(x, str) and x for x in args):
                raise ValueError("command sequence must contain non-empty strings")

        run_env = os.environ.copy()
        if env: run_env.update({str(k): str(v) for k, v in env.items()})
        started = time.monotonic()
        try:
            completed = subprocess.run(args, shell=shell, executable=executable,
                                       cwd=cwd, env=run_env, input=stdin,
                                       capture_output=True, text=True,
                                       timeout=effective_timeout, check=False)
            raw_stdout = self._text(completed.stdout)
            raw_stderr = self._text(completed.stderr)
            truncated = len(raw_stdout) > limit or len(raw_stderr) > limit
            result = TerminalResult(command, raw_stdout[:limit], raw_stderr[:limit],
                                    completed.returncode, False,
                                    time.monotonic() - started, truncated)
        except FileNotFoundError as exc:
            raise TerminalUnavailableError(str(exc)) from exc
        except subprocess.TimeoutExpired as exc:
            raw_stdout = self._text(exc.stdout)
            raw_stderr = self._text(exc.stderr)
            truncated = len(raw_stdout) > limit or len(raw_stderr) > limit
            result = TerminalResult(command, raw_stdout[:limit], raw_stderr[:limit],
                                    -1, True, time.monotonic() - started, truncated)
        if check and not result.ok:
            raise subprocess.CalledProcessError(result.returncode, command,
                                                 output=result.stdout, stderr=result.stderr)
        return result

terminal = Terminal()

def _terminal_tool(command, **kwargs):
    return terminal.run(command, **kwargs).as_dict()

try:
    from .tools import tool
    tool.add(_terminal_tool, name="terminal",
             description="Execute a local terminal command and return structured output.",
             trusted=True, aliases=("shell_terminal",), category="development",
             dangerous=True, requires_explicit_enable=True)
    _tool = tool.get_tool("terminal")
    _tool._on_enable = terminal.enable
    _tool._on_disable = terminal.disable
    tool.disable("terminal")
except Exception:
    pass
