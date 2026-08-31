"""Universal development-code execution engine.

Code execution is opt-in and is exposed as the ``code_execution`` tool by the
PyIntell tool registry. Runtime availability is discovered on the host; a
registered language is never treated as installed automatically.
"""
from dataclasses import dataclass, field
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from .languages import get_language, detect_language

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
            "language": self.language,
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration": self.duration,
            "ok": self.ok,
            "runtime_available": self.runtime_available,
            "file": self.file,
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
        # Kept as policy metadata. Network isolation is platform-dependent and
        # is deliberately NOT falsely advertised as enforced by this runner.
        self.allow_network = bool(allow_network)

    def allow(self, language):
        if self.allowed_languages is None:
            return True
        allowed = {x.lower() for x in self.allowed_languages}
        return language.name.lower() in allowed or any(a.lower() in allowed for a in language.aliases)

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

    def _prepare(self, language, code, cwd=None):
        root = Path(cwd or self.policy.cwd or tempfile.gettempdir()).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        suffix = language.extensions[0] if language.extensions else ".txt"
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, prefix="pyintell_",
                                             dir=str(root), delete=False, encoding="utf-8")
        try:
            handle.write(str(code))
        finally:
            handle.close()
        return Path(handle.name)

    def _commands(self, language, runtime, path, args=()):
        args = [str(x) for x in args]
        name = language.name
        if name in {"C", "C++"}:
            artifact = str(path.with_suffix(".pyintell_bin"))
            return [runtime, str(path), "-o", artifact], [artifact, *args], artifact
        if name == "Rust" and runtime == "rustc":
            artifact = str(path.with_suffix(".pyintell_bin"))
            return [runtime, str(path), "-o", artifact], [artifact, *args], artifact
        if name == "Fortran":
            artifact = str(path.with_suffix(".pyintell_bin"))
            return [runtime, str(path), "-o", artifact], [artifact, *args], artifact
        if name == "Pascal":
            artifact = str(path.with_suffix(".pyintell_bin"))
            return [runtime, str(path), "-o", artifact], [artifact, *args], artifact
        if name == "Go":
            return [runtime, "run", str(path), *args], None, None
        if name == "Java":
            if runtime != "javac":
                raise RuntimeUnavailableError("Java execution requires javac")
            source = path.with_name("Main.java")
            if path != source:
                source.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                path.unlink()
                path = source
            return [runtime, str(path)], ["java", "-cp", str(path.parent), "Main", *args], None
        if name == "Kotlin":
            if runtime != "kotlinc":
                raise RuntimeUnavailableError("Kotlin execution requires kotlinc")
            artifact = str(path.with_suffix(".jar"))
            return [runtime, str(path), "-include-runtime", "-d", artifact], ["java", "-jar", artifact, *args], artifact
        if name == "Zig":
            return [runtime, "run", str(path)], None, None
        if name == "GDScript":
            return [runtime, "--headless", "--script", str(path), *args], None, None
        if name == "Shell":
            return [runtime, str(path), *args], None, None
        if name == "PowerShell":
            return [runtime, "-File", str(path), *args], None, None
        if name == "WebAssembly":
            return [runtime, str(path), *args], None, None
        return [runtime, str(path), *args], None, None

    @staticmethod
    def _decode(value):
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return str(value)

    def _run_command(self, command, *, cwd, env, timeout, stdin=None):
        started = time.monotonic()
        try:
            completed = subprocess.run(command, cwd=cwd, env=env, input=stdin,
                                       capture_output=True, text=True, timeout=timeout,
                                       shell=False, check=False)
            return completed.returncode, completed.stdout or "", completed.stderr or "", False, time.monotonic() - started
        except subprocess.TimeoutExpired as exc:
            return None, self._decode(exc.stdout), self._decode(exc.stderr), True, time.monotonic() - started

    def run(self, code=None, language=None, *, filename=None, timeout=None, cwd=None,
            env=None, args=(), stdin=None, keep_file=False, compile=True):
        if not self.policy.enabled:
            raise ExecutionDisabledError("Code execution is disabled")
        if code is None and not filename:
            raise ValueError("provide code or filename")
        if filename is not None and code is not None:
            raise ValueError("provide either code or filename, not both")

        lang = (get_language(language, filename) if language else
                detect_language(filename) if filename else get_language("python"))
        if not self.policy.allow(lang):
            raise PermissionError(f"Language '{lang.name}' is not permitted")
        runtime, runtime_path = self._runtime(lang)
        if not runtime:
            raise RuntimeUnavailableError(f"No installed runtime/compiler found for {lang.name}: {lang.commands}")

        generated = filename is None
        path = self._prepare(lang, code, cwd) if generated else Path(filename).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(str(path))
        run_cwd = str(Path(cwd).expanduser()) if cwd else str(path.parent.resolve())
        Path(run_cwd).mkdir(parents=True, exist_ok=True)
        run_env = os.environ.copy()
        run_env.update(self.policy.env)
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})
        limit = self.policy.max_output
        total_timeout = float(timeout if timeout is not None else self.policy.default_timeout)
        if total_timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        started = time.monotonic()
        artifact = None
        try:
            compile_command, run_command, artifact = self._commands(lang, runtime, path, args)
            if compile and run_command is not None and artifact is not None:
                remaining = max(0.1, total_timeout - (time.monotonic() - started))
                rc, out, err, timed, duration = self._run_command(
                    compile_command, cwd=run_cwd, env=run_env, timeout=remaining, stdin=None)
                if timed or rc != 0:
                    return ExecutionResult(lang.name, compile_command, rc, out[:limit], err[:limit], timed,
                                           time.monotonic() - started, False, True, str(path),
                                           {"runtime": runtime, "runtime_path": runtime_path,
                                            "gui": lang.gui, "frameworks": list(lang.frameworks), "phase": "compile"})
                remaining = max(0.1, total_timeout - (time.monotonic() - started))
                rc, run_out, run_err, timed, run_duration = self._run_command(
                    run_command, cwd=run_cwd, env=run_env, timeout=remaining, stdin=stdin)
                return ExecutionResult(lang.name, run_command, rc, (out + run_out)[:limit],
                                       (err + run_err)[:limit], timed, time.monotonic() - started,
                                       rc == 0 and not timed, True, str(path),
                                       {"runtime": runtime, "runtime_path": runtime_path,
                                        "gui": lang.gui, "frameworks": list(lang.frameworks),
                                        "phase": "run", "artifact": artifact})

            command = compile_command
            remaining = max(0.1, total_timeout - (time.monotonic() - started))
            rc, out, err, timed, duration = self._run_command(
                command, cwd=run_cwd, env=run_env, timeout=remaining, stdin=stdin)
            return ExecutionResult(lang.name, command, rc, out[:limit], err[:limit], timed,
                                   time.monotonic() - started, rc == 0 and not timed, True, str(path),
                                   {"runtime": runtime, "runtime_path": runtime_path,
                                    "gui": lang.gui, "frameworks": list(lang.frameworks)})
        finally:
            if generated and not keep_file:
                try:
                    path.unlink()
                except OSError:
                    pass
            if artifact:
                try:
                    Path(artifact).unlink()
                except OSError:
                    pass

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
