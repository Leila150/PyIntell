"""System resource inspection used by model builders."""

import os
import platform
import shutil

try:
    import psutil
except ImportError:
    psutil = None


def ram():
    """Return total, available, and used system RAM in bytes."""
    if psutil is not None:
        try:
            info = psutil.virtual_memory()
            return {"total": info.total, "available": info.available, "used": info.used}
        except Exception:
            pass

    # Android/Pydroid may not expose virtual memory through psutil.
    try:
        values = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as file:
            for line in file:
                key, value = line.split(":", 1)
                parts = value.strip().split()
                if parts:
                    values[key] = int(parts[0]) * 1024

        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if available is None:
            available = values.get("MemFree", 0) + values.get("Buffers", 0) + values.get("Cached", 0)
        used = total - available if total is not None and available is not None else None
        return {"total": total, "available": available, "used": used}
    except (OSError, ValueError, IndexError):
        return {"total": None, "available": None, "used": None}


def cpu_info():
    return {"name": platform.processor(), "count": os.cpu_count(), "architecture": platform.machine()}


def storage_info(path="."):
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "free": usage.free, "used": usage.used}


def memory_info():
    process = None
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            pass
    return {"ram": ram(), "process": process}


def device_info():
    return {"cpu": cpu_info(), "gpu": gpu_info()}


def device_count():
    return os.cpu_count() or 1


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return {"available": True, "count": torch.cuda.device_count(), "name": torch.cuda.get_device_name(0)}
    except Exception:
        pass
    return {"available": False, "count": 0, "name": None}


def is_gpu_available():
    return bool(gpu_info()["available"])


def cuda():
    return is_gpu_available()


def cpu():
    return True


def clear_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def free_memory():
    return ram().get("available")


def gpu_memory():
    return gpu_info()


def cpu_memory():
    return ram()


def system_info():
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram": ram(),
        "storage": storage_info(),
        "gpu": gpu_info(),
    }
