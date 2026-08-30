"""Built-in tools: web search, PyPI search, pip, and bash.

Network and process execution are explicit tool calls and can be disabled through
PyIntell's tool registry. Bash is intentionally opt-in at call time.
"""
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from .tools import tool

_USER_AGENT = "PyIntell/0.2.0 (+https://github.com/Leila150/PyIntell)"

def _request_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

@tool.add(name="web_search", description="Search the web using DuckDuckGo Instant Answer data.", trusted=True)
def web_search(query, max_results=8):
    """Return web result candidates for a query without scraping arbitrary pages."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
    data = _request_json(url)
    results = []
    if data.get("AbstractText"):
        results.append({"title": data.get("Heading", query), "url": data.get("AbstractURL", ""), "snippet": data["AbstractText"], "source": "DuckDuckGo"})
    def walk(items):
        for item in items:
            if isinstance(item, dict) and item.get("FirstURL"):
                results.append({"title": item.get("Text", ""), "url": item.get("FirstURL", ""), "snippet": item.get("Text", ""), "source": "DuckDuckGo"})
            if isinstance(item, dict) and item.get("Topics"):
                walk(item["Topics"])
    walk(data.get("RelatedTopics", []))
    return results[:max(1, int(max_results))]

@tool.add(name="pypi_search", description="Search PyPI package metadata and documentation links.", trusted=True)
def pypi_search(query, limit=10):
    """Search PyPI's public JSON index through its search-compatible XML-RPC API."""
    # PyPI's legacy search endpoint is no longer reliable, so use the public index
    # project API for exact names and return a useful package/documentation record.
    q = query.strip().lower()
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
        found[info.get("name", name)] = {"name": info.get("name", name), "version": info.get("version"), "summary": info.get("summary"), "home_page": info.get("home_page"), "documentation": info.get("project_urls", {}).get("Documentation") or info.get("docs_url"), "pypi": f"https://pypi.org/project/{info.get('name', name)}/"}
    return list(found.values())[:max(1, int(limit))]

@tool.add(name="pip", description="Run pip for the current Python interpreter.", trusted=True)
def pip(*args, timeout=120):
    """Execute pip with the current interpreter and return structured output."""
    allowed = {"install", "uninstall", "show", "list", "check", "download", "index", "cache"}
    if not args:
        raise ValueError("pip requires arguments")
    if str(args[0]) not in allowed:
        raise PermissionError(f"pip subcommand '{args[0]}' is not allowed")
    command = [sys.executable, "-m", "pip", *map(str, args)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "command": command}

@tool.add(name="bash", description="Execute a shell command. Disabled by default for safety-sensitive use.", trusted=False)
def bash(command, timeout=30, cwd=None, allow_failure=False):
    """Run a shell command with explicit execution and timeout controls."""
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    proc = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    result = {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "command": command}
    if proc.returncode and not allow_failure:
        raise RuntimeError(result)
    return result
