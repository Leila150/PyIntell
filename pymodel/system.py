"""System resource inspection used by model builders."""

import os
import shutil

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def ram():
    """Return total and available system RAM in bytes."""
    if psutil is None:
        return {"total": None, "available": None, "used": None}
    info = psutil.virtual_memory()
    return {"total": info.total, "available": info.available, "used": info.used}


def storage_info(path="."):
    """Return storage statistics for the filesystem containing path."""
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "free": usage.free, "used": usage.used}


def memory_info():
    """Return a normalized memory report."""
    return {"ram": ram(), "process": psutil.Process(os.getpid()).memory_info().rss if psutil else None}


def system_info():
    """Return basic hardware/resource information without requiring GPU packages."""
    import platform
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram": ram(),
        "storage": storage_info(),
    }
