"""Opt-in Bash tool built on the host shell, not Termux."""
from __future__ import annotations

import os
import shutil
from typing import Optional

from .terminal import Terminal, TerminalResult

class Bash:
    """Controlled Bash runner. Disabled by default."""
    def __init__(self, enabled=False, timeout=30.0, max_output=1_000_000):
        self.terminal = Terminal(enabled=enabled, timeout=timeout,
                                 require_termux=False, shell_path=shutil.which("bash"),
                                 max_output=max_output)

    @property
    def enabled(self): return self.terminal.enabled

    def enable(self): self.terminal.enable(); return self
    def disable(self): self.terminal.disable(); return self

    @staticmethod
    def available(): return shutil.which("bash") is not None

    def status(self):
        info = self.terminal.status()
        info["bash_available"] = self.available()
        return info

    def run(self, command: str, *, timeout: Optional[float] = None,
            cwd=None, env=None, stdin=None, check=False, max_output=None):
        if not self.available():
            raise RuntimeError("Bash is not installed on this host")
        return self.terminal.run(command, timeout=timeout, shell=True, cwd=cwd,
                                 env=env, stdin=stdin, check=check,
                                 max_output=max_output)

    def execute(self, command: str, **kwargs):
        return self.run(command, **kwargs).as_dict()

bash = Bash()

# First-class AI tool. Explicitly disabled by default.
def _bash_tool(command: str, **kwargs):
    return bash.execute(command, **kwargs)

try:
    from .tools import tool
    tool.add(_bash_tool, name="bash",
             description="Execute a Bash command on the local host and return structured output.",
             trusted=True, aliases=("bash_execute",), category="development",
             dangerous=True, requires_explicit_enable=True)
    tool.disable("bash")
except Exception:
    pass
