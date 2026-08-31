"""Data-driven language/compiler/runtime and GUI framework registry."""
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
    def available(self): return any(shutil.which(c) for c in self.commands)
    def runtime(self):
        for c in self.commands:
            p = shutil.which(c)
            if p: return p
        return None

LANGUAGES: Dict[str, Language] = {}
def register_language(language: Language):
    if not isinstance(language, Language): raise TypeError("language must be a Language")
    LANGUAGES[language.name.lower()] = language
    for alias in language.aliases: LANGUAGES[alias.lower()] = language
    return language

def _add(name, ext=(), commands=(), run=(), compile=(), gui=False, frameworks=(), aliases=(), **metadata):
    return register_language(Language(name, tuple(ext), tuple(commands), tuple(run), tuple(compile), gui, tuple(frameworks), tuple(aliases), metadata))

_add("Python", (".py",), ("python", "python3", "py"), frameworks=("Tkinter", "CustomTkinter", "Kivy", "PySide6", "PyQt6", "wxPython", "Pygame", "Flask", "FastAPI", "Django", "Toga", "Dear PyGui", "Flet"), aliases=("py",))
_add("JavaScript", (".js", ".mjs", ".cjs"), ("node", "bun", "deno"), gui=True, frameworks=("Electron", "React", "React Native", "Vue", "Svelte"), aliases=("js", "nodejs"))
_add("TypeScript", (".ts", ".tsx"), ("tsx", "ts-node", "bun", "deno"), gui=True, frameworks=("React", "Angular", "Vue", "Svelte", "Next.js"), aliases=("ts",))
_add("C", (".c",), ("gcc", "clang", "cc"), compile=("{compiler}", "{file}", "-o", "{output}"))
_add("C++", (".cpp", ".cc", ".cxx"), ("g++", "clang++", "c++"), compile=("{compiler}", "{file}", "-o", "{output}"), gui=True, frameworks=("Qt", "GTK", "wxWidgets", "SDL", "SFML"), aliases=("cpp", "cxx"))
_add("C#", (".cs",), ("dotnet", "csc"), gui=True, frameworks=(".NET", "Avalonia", "WinUI", "WPF", "MAUI"), aliases=("csharp", "cs"))
_add("Java", (".java",), ("java", "javac"), gui=True, frameworks=("JavaFX", "Swing", "Android"), aliases=("jvm",))
_add("Kotlin", (".kt", ".kts"), ("kotlinc", "kotlin"), gui=True, frameworks=("Compose Multiplatform", "Jetpack Compose", "Android"), aliases=("kt",))
_add("Swift", (".swift",), ("swift", "swiftc"), gui=True, frameworks=("SwiftUI", "UIKit", "AppKit"))
_add("Objective-C", (".m", ".mm"), ("clang",), gui=True, frameworks=("Cocoa", "UIKit"), aliases=("objc",))
_add("Go", (".go",), ("go",), gui=True, frameworks=("Fyne", "Wails"))
_add("Rust", (".rs",), ("rustc", "cargo"), gui=True, frameworks=("egui", "iced", "Tauri"))
_add("Ruby", (".rb",), ("ruby",), frameworks=("Rails", "Sinatra"))
_add("PHP", (".php",), ("php",), frameworks=("Laravel", "Symfony"))
_add("Perl", (".pl", ".pm"), ("perl",))
_add("R", (".r", ".R"), ("Rscript", "R"))
_add("Dart", (".dart",), ("dart",), gui=True, frameworks=("Flutter",))
_add("Lua", (".lua",), ("lua", "luajit"), gui=True, frameworks=("LÖVE", "Solar2D"), aliases=("luajit",))
_add("Luau", (".luau",), ("luau",), gui=True, frameworks=("Roblox",), aliases=("roblox luau",))
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
_add("Assembly", (".asm", ".s", ".S"), ("nasm", "as", "gcc"), aliases=("asm", "x86 asm"))
_add("Shell", (".sh", ".bash", ".zsh", ".fish"), ("bash", "sh", "zsh", "fish"), aliases=("bash", "sh", "shell", "zsh", "fish"))
_add("PowerShell", (".ps1",), ("pwsh", "powershell"), aliases=("powershell", "ps"))
_add("SQL", (".sql",), ("sqlite3",), frameworks=("SQLite", "PostgreSQL", "MySQL", "MariaDB"))
_add("HTML", (".html", ".htm"), (), gui=True, frameworks=("CSS", "JavaScript", "WebAssembly"))
_add("CSS", (".css", ".scss", ".sass", ".less"), (), gui=True, frameworks=("Tailwind", "Bootstrap"))
_add("WebAssembly", (".wat", ".wasm"), ("wasmtime", "wasmer", "wasm3", "wat2wasm"), aliases=("wasm", "webassembly"))
_add("MATLAB", (".m",), ("matlab", "octave"), aliases=("octave",))
_add("VHDL", (".vhd", ".vhdl"), ("ghdl",))
_add("Verilog", (".v", ".sv"), ("iverilog", "verilator"))
_add("Prolog", (".plg", ".pro"), ("swipl",), aliases=("swi-prolog",))
_add("Clojure", (".clj", ".cljs", ".cljc"), ("clojure",))
_add("F#", (".fs", ".fsx", ".fsi"), ("dotnet",), aliases=("fsharp",))
_add("Crystal", (".cr",), ("crystal",))
_add("Raku", (".raku", ".rakumod"), ("raku",), aliases=("perl6",))
_add("GDScript", (".gd",), ("godot",), gui=True, frameworks=("Godot",), aliases=("gdscript",))
_add("Solidity", (".sol",), ("solc", "forge"), aliases=("sol",))
_add("V", (".v",), ("v", "vlang"), aliases=("vlang",))
_add("CoffeeScript", (".coffee",), ("coffee", "node"), aliases=("coffee",))

GUI_FRAMEWORKS = {
    "Kivy": "Python", "Tkinter": "Python", "CustomTkinter": "Python", "PySide6": "Python", "PyQt6": "Python", "wxPython": "Python", "Pygame": "Python", "Toga": "Python", "Dear PyGui": "Python", "Flet": "Python",
    "Flutter": "Dart", "SwiftUI": "Swift", "UIKit": "Swift", "AppKit": "Swift", "Jetpack Compose": "Kotlin", "Compose Multiplatform": "Kotlin", "JavaFX": "Java", "Swing": "Java",
    "Avalonia": "C#", "MAUI": "C#", "WinUI": "C#", "WPF": "C#", "Electron": "JavaScript", "React": "JavaScript", "React Native": "JavaScript", "Vue": "JavaScript", "Angular": "TypeScript", "Svelte": "JavaScript", "Qt": "C++", "GTK": "C/C++", "SDL": "C/C++", "SFML": "C++", "Fyne": "Go", "Wails": "Go", "egui": "Rust", "iced": "Rust", "Tauri": "Rust", "LÖVE": "Lua", "Roblox": "Luau", "Godot": "GDScript",
}

def get_language(value: str, filename: Optional[str] = None) -> Language:
    key = str(value or "").lower()
    if key and key in LANGUAGES: return LANGUAGES[key]
    if filename:
        low = str(filename).lower()
        candidates = []
        seen = set()
        for language in LANGUAGES.values():
            if id(language) in seen: continue
            seen.add(id(language))
            for ext in language.extensions:
                if low.endswith(ext.lower()): candidates.append(language); break
        if candidates:
            candidates.sort(key=lambda x: max((len(e) for e in x.extensions if low.endswith(e.lower())), default=0), reverse=True)
            return candidates[0]
    raise KeyError(f"Unknown language: {value!r}")

def list_languages(available_only=False, gui_only=False):
    seen, result = set(), []
    for language in LANGUAGES.values():
        if id(language) in seen: continue
        seen.add(id(language))
        if available_only and not language.available(): continue
        if gui_only and not language.gui: continue
        result.append(language)
    return sorted(result, key=lambda x: x.name.lower())

def detect_language(filename: str) -> Language: return get_language("", filename=filename)
def register(name: str, extensions: Iterable[str], commands: Iterable[str], **kwargs):
    return register_language(Language(name, tuple(extensions), tuple(commands), **kwargs))
