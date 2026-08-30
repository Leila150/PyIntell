"""Language and development-runtime registry.

The registry is intentionally data-driven: PyIntell does not pretend that every
compiler exists on every machine. It discovers installed runtimes and lets users
register additional languages/frameworks without changing the executor.
"""
from dataclasses import dataclass, field
import shutil
from typing import Dict, Iterable, Optional

@dataclass(frozen=True)
class Language:
    name: str
    extensions: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    run_template: tuple[str, ...] = ()
    compile_template: tuple[str, ...] = ()
    gui: bool = False
    frameworks: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def available(self) -> bool:
        return any(shutil.which(c) for c in self.commands)

LANGUAGES: Dict[str, Language] = {}

def register_language(language: Language):
    if not isinstance(language, Language):
        raise TypeError("language must be a Language")
    LANGUAGES[language.name.lower()] = language
    for alias in language.aliases:
        LANGUAGES[alias.lower()] = language
    return language

def _add(name, ext, commands, run=(), compile=(), gui=False, frameworks=(), aliases=()):
    return register_language(Language(name, tuple(ext), tuple(commands), tuple(run), tuple(compile), gui, tuple(frameworks), tuple(aliases)))

# Core/general-purpose languages and common build systems.
_add("Python", (".py",), ("python", "python3", "py"), ("{runtime}", "{file}"), frameworks=("Tkinter", "CustomTkinter", "Kivy", "PySide6", "PyQt6", "wxPython", "Pygame", "Flask", "FastAPI"), aliases=("py",))
_add("JavaScript", (".js", ".mjs", ".cjs"), ("node", "bun", "deno"), ("{runtime}", "{file}"), frameworks=("Electron", "React", "Vue", "Svelte"), aliases=("js", "nodejs"))
_add("TypeScript", (".ts", ".tsx"), ("tsx", "ts-node", "bun", "deno"), ("{runtime}", "{file}"), frameworks=("React", "Angular", "Vue", "Svelte"), aliases=("ts",))
_add("C", (".c",), ("gcc", "clang"), ("{file}" ,), ("{compiler}", "{file}", "-o", "{output}"))
_add("C++", (".cpp", ".cc", ".cxx", ".hpp", ".hh"), ("g++", "clang++"), compile=("{compiler}", "{file}", "-o", "{output}"), aliases=("cpp", "cxx"))
_add("C#", (".cs",), ("dotnet", "csc"), frameworks=(".NET", "Avalonia", "WinUI", "MAUI"), aliases=("csharp", "cs"))
_add("Java", (".java",), ("java", "javac"), aliases=("jvm",))
_add("Kotlin", (".kt", ".kts"), ("kotlinc", "kotlin"), frameworks=("Compose Multiplatform",), aliases=("kt",))
_add("Swift", (".swift",), ("swift", "swiftc"), gui=True, frameworks=("SwiftUI", "UIKit"))
_add("Objective-C", (".m", ".mm"), ("clang",), gui=True, frameworks=("Cocoa", "UIKit"), aliases=("objc",))
_add("Go", (".go",), ("go",))
_add("Rust", (".rs",), ("rustc", "cargo"))
_add("Ruby", (".rb",), ("ruby",), frameworks=("Rails", "Sinatra"))
_add("PHP", (".php",), ("php",), frameworks=("Laravel", "Symfony"))
_add("Perl", (".pl", ".pm"), ("perl",))
_add("R", (".r", ".R"), ("Rscript", "R"))
_add("Dart", (".dart",), ("dart",), gui=True, frameworks=("Flutter",))
_add("Lua", (".lua",), ("lua", "luajit"), aliases=("luajit",))
_add("Julia", (".jl",), ("julia",))
_add("Haskell", (".hs",), ("ghc", "runhaskell"))
_add("OCaml", (".ml", ".mli"), ("ocaml", "ocamlc", "dune"))
_add("Elixir", (".ex", ".exs"), ("elixir", "mix"))
_add("Erlang", (".erl",), ("erl", "erlc"))
_add("Scala", (".scala",), ("scala", "scalac"))
_add("Groovy", (".groovy",), ("groovy",))
_add("Zig", (".zig",), ("zig",))
_add("Nim", (".nim",), ("nim",))
_add("D", (".d",), ("dmd", "ldc2", "gdc"))
_add("Fortran", (".f", ".f90", ".f95", ".f03", ".f08"), ("gfortran",))
_add("Pascal", (".pas",), ("fpc", "ppc"))
_add("Assembly", (".asm", ".s"), ("nasm", "as", "gas"), aliases=("asm", "x86 asm"))
_add("Shell", (".sh", ".bash", ".zsh"), ("bash", "sh", "zsh"), aliases=("bash", "sh", "shell"))
_add("PowerShell", (".ps1",), ("pwsh", "powershell"), aliases=("powershell", "ps"))
_add("SQL", (".sql",), ("sqlite3",), gui=False, frameworks=("SQLite", "PostgreSQL", "MySQL"))
_add("HTML", (".html", ".htm"), (), gui=True, frameworks=("CSS", "JavaScript", "WebAssembly"))
_add("CSS", (".css", ".scss", ".sass", ".less"), (), gui=True, frameworks=("Tailwind", "Bootstrap"))
_add("WebAssembly", (".wat", ".wasm"), ("wasmtime", "wasmer", "wat2wasm"), aliases=("wasm", "webassembly"))
_add("MATLAB", (".m",), ("matlab",), aliases=("octave",))
_add("VHDL", (".vhd", ".vhdl"), ("ghdl",))
_add("Verilog", (".v", ".sv"), ("iverilog", "verilator"))
_add("Prolog", (".plg", ".pro"), ("swipl",), aliases=("swi-prolog",))
_add("Clojure", (".clj", ".cljs"), ("clojure",))
_add("F#", (".fs", ".fsx"), ("dotnet",), aliases=("fsharp",))
_add("Crystal", (".cr",), ("crystal",))
_add("Raku", (".raku", ".rakumod"), ("raku",), aliases=("perl6",))

# Development-oriented GUI/framework registry. These are metadata, not hard dependencies.
GUI_FRAMEWORKS = {
    "Kivy": "Python", "Tkinter": "Python", "CustomTkinter": "Python",
    "PySide6": "Python", "PyQt6": "Python", "wxPython": "Python", "Pygame": "Python",
    "Toga": "Python", "Dear PyGui": "Python", "Flet": "Python",
    "Flutter": "Dart", "SwiftUI": "Swift", "UIKit": "Swift",
    "Jetpack Compose": "Kotlin", "Compose Multiplatform": "Kotlin",
    "JavaFX": "Java", "Swing": "Java", "Avalonia": "C#", "MAUI": "C#",
    "WinUI": "C#", "WPF": "C#", "Electron": "JavaScript",
    "React": "JavaScript", "React Native": "JavaScript", "Vue": "JavaScript",
    "Angular": "TypeScript", "Svelte": "JavaScript", "Qt": "C++", "GTK": "C/C++",
}

def get_language(value: str, filename: Optional[str] = None) -> Language:
    key = str(value or "").lower()
    if key in LANGUAGES:
        return LANGUAGES[key]
    if filename:
        low = filename.lower()
        for language in {id(x): x for x in LANGUAGES.values()}.values():
            if any(low.endswith(ext.lower()) for ext in language.extensions):
                return language
    raise KeyError(f"Unknown language: {value!r}")

def list_languages(available_only=False, gui_only=False):
    seen = set(); result = []
    for language in LANGUAGES.values():
        if id(language) in seen: continue
        seen.add(id(language))
        if available_only and not language.available(): continue
        if gui_only and not language.gui: continue
        result.append(language)
    return sorted(result, key=lambda x: x.name.lower())

def detect_language(filename: str) -> Language:
    return get_language("", filename=filename)

def register(name: str, extensions: Iterable[str], commands: Iterable[str], **kwargs):
    return register_language(Language(name, tuple(extensions), tuple(commands), **kwargs))
