"""Dependency-free crash and runtime logging for PyIntell."""
from __future__ import annotations
import json, os, platform, sys, time, traceback
from pathlib import Path

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

class CrashLogger:
    def __init__(self, directory=None, level="INFO", enabled=True):
        self.directory = Path(directory or (Path.home() / ".pyintell" / "logs"))
        self.level = str(level).upper(); self.enabled = bool(enabled)
        if self.enabled: self.directory.mkdir(parents=True, exist_ok=True)
    def _write(self, level, message, **context):
        if not self.enabled or _LEVELS.get(level, 20) < _LEVELS.get(self.level, 20): return None
        record = {"timestamp": time.time(), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": level, "message": str(message), "context": context}
        path = self.directory / "pyintell.log"
        with path.open("a", encoding="utf-8") as f: f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return path
    def log(self, level, message, **context): return self._write(str(level).upper(), message, **context)
    def debug(self, message, **context): return self._write("DEBUG", message, **context)
    def info(self, message, **context): return self._write("INFO", message, **context)
    def warning(self, message, **context): return self._write("WARNING", message, **context)
    def error(self, message, **context): return self._write("ERROR", message, **context)
    def crash(self, exception=None, *, message=None, **context):
        exc = exception or sys.exc_info()[1]
        if exc is None: return self.error(message or "Crash reported without an exception", **context)
        record = {"timestamp": time.time(), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "CRITICAL", "message": message or str(exc), "exception_type": type(exc).__name__, "exception": str(exc), "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), "python": sys.version, "platform": platform.platform(), "context": context}
        path = self.directory / f"crash-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        self._write("CRITICAL", message or str(exc), exception_type=type(exc).__name__, crash_file=str(path), **context)
        return path
    def install_excepthook(self):
        previous = sys.excepthook
        def hook(exc_type, exc, tb):
            self.crash(exc)
            previous(exc_type, exc, tb)
        sys.excepthook = hook
        return hook

_default_logger = CrashLogger()
def get_logger(): return _default_logger
def log(level, message, **context): return _default_logger.log(level, message, **context)
def crash(exception=None, **context): return _default_logger.crash(exception, **context)
