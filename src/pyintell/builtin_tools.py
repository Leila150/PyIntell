"""Built-in PyIntell tools.

The command-execution tools share the same controlled Terminal backend. The
Terminal backend is disabled by default, so ``bash()`` is also disabled until
it is explicitly enabled through ``terminal.enable()``.
"""
import json
import sys
import urllib.parse
import urllib.request

from .tools import tool
from .terminal import terminal

_USER_AGENT = "PyIntell/0.2.0 (+https://github.com/Leila150/PyIntell)"


def _request_json(url, timeout=15):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@tool.add(name="web_search", description="Search the web using DuckDuckGo Instant Answer data.", trusted=True)
def web_search(query, max_results=8):
    """Return web result candidates for a query without scraping arbitrary pages."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    })
    data = _request_json(url)
    results = []
    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", query),
            "url": data.get("AbstractURL", ""),
            "snippet": data["AbstractText"],
            "source": "DuckDuckGo",
        })

    def walk(items):
        for item in items:
            if isinstance(item, dict) and item.get("FirstURL"):
                results.append({
                    "title": item.get("Text", ""),
                    "url": item.get("FirstURL", ""),
                    "snippet": item.get("Text", ""),
                    "source": "DuckDuckGo",
                })
            if isinstance(item, dict) and item.get("Topics"):
                walk(item["Topics"])

    walk(data.get("RelatedTopics", []))
    return results[:max(1, int(max_results))]


@tool.add(name="pypi_search", description="Search PyPI package metadata and documentation links.", trusted=True)
def pypi_search(query, limit=10):
    """Look up exact PyPI project names and return package/documentation metadata."""
    q = str(query).strip().lower()
    if not q:
        raise ValueError("query must be non-empty")

    candidates = [q]
    if " " in q:
        candidates.extend(q.split())
    found = {}
    for name in candidates:
        try:
            data = _request_json(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
        except Exception:
            continue
        info = data.get("info", {})
        package_name = info.get("name", name)
        found[package_name] = {
            "name": package_name,
            "version": info.get("version"),
            "summary": info.get("summary"),
            "home_page": info.get("home_page"),
            "documentation": info.get("project_urls", {}).get("Documentation") or info.get("docs_url"),
            "pypi": f"https://pypi.org/project/{package_name}/",
        }
    return list(found.values())[:max(1, int(limit))]


@tool.add(name="pip", description="Run pip for the current Python interpreter.", trusted=True)
def pip(*args, timeout=120):
    """Execute an allowed pip subcommand with the current Python interpreter."""
    allowed = {"install", "uninstall", "show", "list", "check", "download", "index", "cache"}
    if not args:
        raise ValueError("pip requires arguments")
    if str(args[0]) not in allowed:
        raise PermissionError(f"pip subcommand '{args[0]}' is not allowed")
    command = [sys.executable, "-m", "pip", *map(str, args)]
    result = terminal.run(command, timeout=timeout)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
        "ok": result.ok,
    }


@tool.add(name="bash", description="Execute a Bash command through PyIntell's controlled terminal.", trusted=False)
def bash(command, timeout=30, cwd=None, allow_failure=False):
    """Run Bash through the shared Terminal backend.

    Bash is intentionally disabled by default. Call ``terminal.enable()`` first.
    This keeps all local command execution under one timeout/capture/detection
    implementation instead of maintaining a second subprocess path.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")

    result = terminal.run(
        command,
        timeout=timeout,
        cwd=cwd,
        shell=True,
    )
    output = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
        "ok": result.ok,
        "timed_out": result.timed_out,
    }
    if result.returncode and not allow_failure:
        raise RuntimeError(output)
    return output
