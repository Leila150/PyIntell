"""Trusted-package metadata and documentation registry for PyIntell."""
from dataclasses import dataclass, asdict
import importlib.util

@dataclass(frozen=True)
class PackageInfo:
    name: str
    import_name: str
    category: str
    documentation: str
    description: str = ""
    trusted: bool = True
    installed: bool = False

_DEFAULTS = [
    ("random", "random", "stdlib", "https://docs.python.org/3/library/random.html", "Pseudo-random number generation"),
    ("time", "time", "stdlib", "https://docs.python.org/3/library/time.html", "Time access and conversions"),
    ("sqlite3", "sqlite3", "stdlib", "https://docs.python.org/3/library/sqlite3.html", "SQLite database interface"),
    ("json", "json", "stdlib", "https://docs.python.org/3/library/json.html", "JSON encoding and decoding"),
    ("os", "os", "stdlib", "https://docs.python.org/3/library/os.html", "Operating-system interfaces"),
    ("pathlib", "pathlib", "stdlib", "https://docs.python.org/3/library/pathlib.html", "Object-oriented filesystem paths"),
    ("numpy", "numpy", "scientific", "https://numpy.org/doc/stable/", "Numerical computing"),
    ("pytorch", "torch", "deep-learning", "https://pytorch.org/docs/stable/", "Tensor and deep-learning framework"),
    ("flask", "flask", "web", "https://flask.palletsprojects.com/", "Lightweight WSGI web framework"),
    ("fastapi", "fastapi", "web", "https://fastapi.tiangolo.com/", "Modern Python API framework"),
    ("pyintell", "pyintell", "ai", "https://github.com/Leila150/PyIntell", "PyIntell AI framework"),
    ("requests", "requests", "networking", "https://requests.readthedocs.io/", "HTTP client"),
    ("httpx", "httpx", "networking", "https://www.python-httpx.org/", "Async and sync HTTP client"),
    ("discord.py", "discord", "bots", "https://discordpy.readthedocs.io/", "Discord API wrapper"),
    ("kivy", "kivy", "gui", "https://kivy.org/doc/stable/", "Cross-platform GUI framework"),
    ("tkinter", "tkinter", "gui", "https://docs.python.org/3/library/tkinter.html", "Python Tk GUI bindings"),
    ("customtkinter", "customtkinter", "gui", "https://customtkinter.tomschimansky.com/documentation/", "Modern Tkinter widgets"),
    ("pandas", "pandas", "data", "https://pandas.pydata.org/docs/", "Data analysis"),
    ("scipy", "scipy", "scientific", "https://docs.scipy.org/doc/scipy/", "Scientific computing"),
    ("Pillow", "PIL", "media", "https://pillow.readthedocs.io/", "Image processing"),
]

TRUSTED_PACKAGES = {}
for row in _DEFAULTS:
    name, imp, category, docs, desc = row
    TRUSTED_PACKAGES[name] = PackageInfo(name, imp, category, docs, desc, True, importlib.util.find_spec(imp) is not None)

def register_package(name, import_name=None, documentation=None, category="external", description="", trusted=False):
    if not documentation:
        raise ValueError("A documentation URL is required for registered packages")
    imp = import_name or name
    TRUSTED_PACKAGES[name] = PackageInfo(name, imp, category, documentation, description, trusted, importlib.util.find_spec(imp) is not None)
    return TRUSTED_PACKAGES[name]

def get_package(name):
    return TRUSTED_PACKAGES[name]

def list_packages(installed_only=False, trusted_only=False):
    values = list(TRUSTED_PACKAGES.values())
    if installed_only: values = [p for p in values if p.installed]
    if trusted_only: values = [p for p in values if p.trusted]
    return [asdict(p) for p in values]

def package_installed(name):
    return get_package(name).installed
