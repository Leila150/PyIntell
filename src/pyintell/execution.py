"""Universal development-code execution engine.

Execution is an explicit capability, not a built-in AI tool. The engine is
runtime-aware: it discovers interpreters/compilers installed on the host and
never claims support merely because a language is registered.
"""
from dataclasses import dataclass, field
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from .languages import Language, get_language, detect_language

class ExecutionDisabledError(RuntimeError):
    """Raised when execution is disabled by policy."""

class RuntimeUnavailableError(RuntimeError):
    """Raised when no runtime/compiler for a language is installed."""

@dataclass
class ExecutionResult:
    language: str
    command: list[str]
    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration: float = 0.0
    ok: bool = False
    runtime_available: bool = True
    file: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def output(self):
        return self.stdout

    @property
    def error(self):
        return self.stderr

    def as_dict(self):
        return {
            "language": self.language, "command": list(self.command),
            "returncode": self.returncode, "stdout": self.stdout,
            "stderr": self.stderr, "timed_out": self.timed_out,
            "duration": self.duration, "ok": self.ok,
            "runtime_available": self.runtime_available, "file": self.file,
            "metadata": dict(self.metadata),
        }

class ExecutionPolicy:
    def __init__(self, enabled=False, default_timeout=30, max_output=1_000_000,
                 cwd=None, env=None, allowed_languages=None, allow_network=True):
        self.enabled = bool(enabled)
        self.default_timeout = max(0.1, float(default_timeout))
        self.max_output = max(1, int(max_output))
        self.cwd = cwd
        self.env = dict(env or {})
        self.allowed_languages = ({str(x).lower() for x in allowed_languages}
                                  if allowed_languages else None)
        self.allow_network = bool(allow_network)

    def allow(self, language):
        return (self.allowed_languages is None or
                language.name.lower() in self.allowed_languages or
                any(a.lower() in self.allowed_languages for a in language.aliases))

class CodeExecutor:
    def __init__(self, policy=None):
        self.policy = policy or ExecutionPolicy()

    def enable(self):
        self.policy.enabled = True
        return self

    def disable(self):
        self.policy.enabled = False
        return self

    def configure(self, **values):
        for key, value in values.items():
            if not hasattr(self.policy, key):
                raise ValueError(f"Unknown execution policy option: {key}")
            setattr(self.policy, key, value)
        return self

    def _runtime(self, language):
        for command in language.commands:
            path = shutil.which(command)
            if path:
                return command, path
        return None, None

    def runtime_info(self, language=None):
        lang = get_language(language or "python")
        command, path = self._runtime(lang)
        return {"language": lang.name, "available": path is not None,
                "command": command, "path": path, "extensions": list(lang.extensions),
                "gui": lang.gui, "frameworks": list(lang.frameworks)}

    def _prepare(self, language, code, filename=None, cwd=None):
        suffix = Path(filename).suffix if filename else (language.extensions[0] if language.extensions else ".txt")
        root = Path(cwd or self.policy.cwd or tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, prefix="pyintell_",
                                             dir=str(root), delete=False, encoding="utf-8")
        handle.write(str(code)); handle.close()
        return Path(handle.name)

    def _command(self, language, runtime, path, args=()):
        args = [str(x) for x in args]
        name = language.name
        if name in {"C", "C++", "Fortran", "Pascal", "Zig"}:
            output = str(path.with_suffix(".pyintell_bin"))
            return [runtime, str(path), "-o", output, *args], output
        if name == "Rust" and runtime == "rustc":
            output = str(path.with_suffix(".pyintell_bin"))
            return [runtime, str(path), "-o", output, *args], output
        if name == "Go":
            return [runtime, "run", str(path), *args], None
        if name == "Java" and runtime == "javac":
            return [runtime, str(path), *args], None
        if name == "Kotlin" and runtime == "kotlinc":
            output = str(path.with_suffix(".jar"))
            return [runtime, str(path), "-include-runtime", "-d", output, *args], output
        if name == "WebAssembly":
            return [runtime, str(path), *args], None
        return [runtime, str(path), *args], None

    def _run_command(self, command, *, cwd, env, timeout, stdin=None):
        started = time.monotonic()
        try:
            completed = subprocess.run(command, cwd=cwd, env=env, input=stdin,
                                       capture_output=True, text=True, timeout=timeout,
                                       shell=False)
            return completed.returncode, completed.stdout or "", completed.stderr or "", False, time.monotonic() - started
        except subprocess.TimeoutExpired as exc:
            return None, str(exc.stdout or ""), str(exc.stderr or ""), True, time.monotonic() - started

    def run(self, code=None, language=None, *, filename=None, timeout=None, cwd=None,
            env=None, args=(), stdin=None, keep_file=False, compile=True):
        if not self.policy.enabled:
            raise ExecutionDisabledError("Code execution is disabled")
        if code is None and not filename:
            raise ValueError("provide code or filename")
        lang = (get_language(language, filename) if language else
                detect_language(filename) if filename else get_language("python"))
        if not self.policy.allow(lang):
            raise PermissionError(f"Language '{lang.name}' is not permitted")
        runtime, runtime_path = self._runtime(lang)
        if not runtime:
            raise RuntimeUnavailableError(f"No installed runtime found for {lang.name}: {lang.commands}")
        path = Path(filename) if filename else self._prepare(lang, code, filename, cwd)
        command, artifact = self._command(lang, runtime, path, args)
        run_env = os.environ.copy()
        run_env.update(self.policy.env)
        run_env.update(env or {})
        limit = self.policy.max_output
        try:
            # Compilers produce an artifact first, then the artifact is executed.
            if compile and artifact:
                rc, out, err, timed, duration = self._run_command(
                    command, cwd=cwd or self.policy.cwd, env=run_env,
                    timeout=float(timeout if timeout is not None else self.policy.default_timeout), stdin=stdin)
                if timed or rc != 0:
                    return ExecutionResult(lang.name, command, rc, out[:limit], err[:limit], timed,
                                           duration, False, True, str(path),
                                           {"runtime": runtime, "runtime_path": runtime_path,
                                            "gui": lang.gui, "frameworks": list(lang.frameworks), "phase": "compile"})
                run_command = [artifact, *[str(x) for x in args]]
                rc, run_out, run_err, timed, run_duration = self._run_command(
                    run_command, cwd=cwd or self.policy.cwd, env=run_env,
                    timeout=float(timeout if timeout is not None else self.policy.default_timeout), stdin=stdin)
                return ExecutionResult(lang.name, run_command, rc, (out + run_out)[:limit],
                                       (err + run_err)[:limit], timed, duration + run_duration,
                                       rc == 0 and not timed, True, str(path),
                                       {"runtime": runtime, "runtime_path": runtime_path,
                                        "gui": lang.gui, "frameworks": list(lang.frameworks), "phase": "run",
                                        "artifact": artifact})
            rc, out, err, timed, duration = self._run_command(
                command, cwd=cwd or self.policy.cwd, env=run_env,
                timeout=float(timeout if timeout is not None else self.policy.default_timeout), stdin=stdin)
            return ExecutionResult(lang.name, command, rc, out[:limit], err[:limit], timed,
                                   duration, rc == 0 and not timed, True, str(path),
                                   {"runtime": runtime, "runtime_path": runtime_path,
                                    "gui": lang.gui, "frameworks": list(lang.frameworks)})
        finally:
            if not keep_file and not filename:
                try: path.unlink()
                except OSError: pass
            if artifact:
                try: Path(artifact).unlink()
                except OSError: pass

    def run_file(self, filename, language=None, **kwargs):
        return self.run(None, language, filename=filename, **kwargs)

    def available_languages(self, gui_only=False):
        from .languages import list_languages
        return list_languages(available_only=True, gui_only=gui_only)

executor = CodeExecutor()

def code_execute(code, language="python", **kwargs):
    return executor.run(code, language, **kwargs)

def run_code(code, language="python", **kwargs):
    return code_execute(code, language, **kwargs)
