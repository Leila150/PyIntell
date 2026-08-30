"""Source verification helpers for web-derived knowledge."""
import hashlib, urllib.parse, urllib.request

TRUSTED_DOMAINS = {
    "docs.python.org", "pypi.org", "numpy.org", "pytorch.org", "fastapi.tiangolo.com",
    "flask.palletsprojects.com", "kivy.org", "github.com", "developer.mozilla.org",
}

def domain(url): return urllib.parse.urlparse(url).netloc.lower().split(":")[0]
def is_trusted_domain(url):
    host = domain(url)
    return host in TRUSTED_DOMAINS or any(host.endswith("." + d) for d in TRUSTED_DOMAINS)

def verify_url(url, timeout=10):
    if not url.startswith(("http://", "https://")): return {"valid": False, "reason": "unsupported scheme"}
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "PyIntell/0.2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"valid": True, "status": r.status, "url": r.geturl(), "trusted_domain": is_trusted_domain(r.geturl())}
    except Exception as exc:
        return {"valid": False, "error": str(exc), "trusted_domain": is_trusted_domain(url)}

def fingerprint(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()

def rank_sources(sources):
    """Rank sources by domain trust, reachability and duplicate content fingerprints."""
    ranked = []
    seen = set()
    for source in sources:
        url = source.get("url", "") if isinstance(source, dict) else str(source)
        score = 2 if is_trusted_domain(url) else 0
        check = verify_url(url)
        score += 2 if check.get("valid") else 0
        key = url.split("#")[0].rstrip("/")
        score += 1 if key not in seen else -2
        seen.add(key)
        ranked.append({**(source if isinstance(source, dict) else {"url": url}), "verification": check, "score": score})
    return sorted(ranked, key=lambda x: x["score"], reverse=True)
