"""Universal development-code execution engine.

The executor is language-agnostic and data-driven. It discovers runtimes already
installed on the host, supports source files and snippets, captures output,
handles timeouts, and exposes GUI/framework metadata without forcing GUI
libraries into PyIntell's dependencies.
"""
from dataclasses import dataclass, field
import os, shlex, shutil, subprocess, tempfile
from pathlib import Path
from typing import Any, Optional

from .languages import Language, get_language, detect_language

class ExecutionDisabledError(RuntimeError): pass
class RuntimeUnavailableError(RuntimeError): pass

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

    def as_dict(self):
        return {"language": self.language, "command": self.command,
                "returncode": self.returncode, "stdout": self.stdout,
                "stderr": self.stderr, "timed_out": self.timed_out,
                "duration": self.duration, "ok": self.ok,
                "runtime_available": self.runtime_available, "file": self.file,
                "metadata": dict(self.metadata)}

class ExecutionPolicy:
    def __init__(self, enabled=True, default_timeout=30, max_output=1_000_000,
                 cwd=None, env=None, allowed_languages=None):
        self.enabled = bool(enabled)
        self.default_timeout = float(default_timeout)
        self.max_output = int(max_output)
        self.cwd = cwd
        self.env = dict(env or {})
        self.allowed_languages = set(x.lower() for x in allowed_languages) if allowed_languages else None

    def allow(self, language):
        return self.allowed_languages is None or language.name.lower() in self.allowed_languages

class CodeExecutor:
    def __init__(self, policy=None):
        self.policy = policy or ExecutionPolicy(enabled=True)

    def enable(self): self.policy.enabled = True; return self
    def disable(self): self.policy.enabled = False; return self

    def _runtime(self, language):
        for command in language.commands:
            path = shutil.which(command)
            if path:
                return command, path
        return None, None

    def _prepare(self, language, code, filename=None, cwd=None):
        suffix = (Path(filename).suffix if filename else (language.extensions[0] if language.extensions else ".txt"))
        root = Path(cwd or self.policy.cwd or tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, prefix="pyintell_", dir=str(root), delete=False, encoding="utf-8")
        handle.write(code); handle.close()
        return Path(handle.name)

    def run(self, code=None, language=None, *, filename=None, timeout=None,
            cwd=None, env=None, args=(), shell=False, keep_file=False):
        if not self.policy.enabled:
            raise ExecutionDisabledError("Code execution is disabled")
        if code is None and not filename:
            raise ValueError("provide code or filename")
        lang = get_language(language, filename) if language else detect_language(filename) if filename else get_language("python")
        if not self.policy.allow(lang):
            raise PermissionError(f"Language '{lang.name}' is not permitted")
        runtime, runtime_path = self._runtime(lang)
        if not runtime:
            raise RuntimeUnavailableError(f"No installed runtime found for {lang.name}: {lang.commands}")
        path = Path(filename) if filename else self._prepare(lang, code, filename, cwd)
        command = self._command(lang, runtime, path, args)
        started = __import__("time").monotonic()
        run_env = os.environ.copy(); run_env.update(self.policy.env); run_env.update(env or {})
        try:
            completed = subprocess.run(command, cwd=cwd or self.policy.cwd,
                env=run_env, capture_output=True, text=True,
                timeout=float(timeout if timeout is not None else self.policy.default_timeout),
                shell=shell)
            timed_out = False
            rc = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True; rc = None
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        finally:
            duration = __import__("time").monotonic() - started
            if not keep_file and not filename:
                try: path.unlink()
                except OSError: pass
        stdout, stderr = stdout[:self.policy.max_output], stderr[:self.policy.max_output]
        return ExecutionResult(lang.name, command, rc, stdout, stderr, timed_out,
                               duration, rc == 0 and not timed_out, True, str(path),
                               {"runtime": runtime, "runtime_path": runtime_path, "gui": lang.gui,
                                "frameworks": list(lang.frameworks)} )

    def _command(self, language: Language, runtime: str, path: Path, args):
        if language.name == "C":
            out = str(path.with_suffix("")); return [runtime, str(path), "-o", out, *map(str,args)]
        if language.name == "C++":
            out = str(path.with_suffix("")); return [runtime, str(path), "-o", out, *map(str,args)]
        if language.name in {"Java", "Kotlin"} and runtime in {"javac", "kotlinc"}:
            return [runtime, str(path), *map(str,args)]
        if language.name == "Rust" and runtime == "cargo":
            return [runtime, "run", *map(str,args)]
        if language.name in {"Go"}:
            return [runtime, "run", str(path), *map(str,args)]
        return [runtime, str(path), *map(str,args)]

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
