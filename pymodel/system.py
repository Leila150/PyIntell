"""System resource inspection used by model builders."""

import os
import platform
import shutil

try:
    import psutil
except ImportError:
    psutil = None


def ram():
    if psutil is None: return {"total": None, "available": None, "used": None}
    info = psutil.virtual_memory(); return {"total": info.total, "available": info.available, "used": info.used}

def cpu_info():
    return {"name": platform.processor(), "count": os.cpu_count(), "architecture": platform.machine()}

def storage_info(path="."):
    usage = shutil.disk_usage(path); return {"total": usage.total, "free": usage.free, "used": usage.used}

def memory_info():
    return {"ram": ram(), "process": psutil.Process(os.getpid()).memory_info().rss if psutil else None}

def device_info(): return {"cpu": cpu_info(), "gpu": gpu_info()}
def device_count(): return os.cpu_count() or 1

def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return {"available": True, "count": torch.cuda.device_count(), "name": torch.cuda.get_device_name(0)}
    except Exception:
        pass
    return {"available": False, "count": 0, "name": None}

def is_gpu_available(): return bool(gpu_info()["available"])
def cuda(): return is_gpu_available()
def cpu(): return True

def clear_cache():
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        pass

def free_memory(): return ram().get("available")
def gpu_memory(): return gpu_info()
def cpu_memory(): return ram()

def system_info():
    return {"platform": platform.platform(), "system": platform.system(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(), "ram": ram(),
            "storage": storage_info(), "gpu": gpu_info()}
