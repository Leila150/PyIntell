"""Evidence and source verification primitives."""
import hashlib, urllib.parse, urllib.request

TRUSTED_DOMAINS={
 "docs.python.org","pypi.org","numpy.org","pytorch.org","fastapi.tiangolo.com","flask.palletsprojects.com","kivy.org","github.com","developer.mozilla.org","docs.rs","go.dev","rust-lang.org","nodejs.org","typescriptlang.org","docs.oracle.com","learn.microsoft.com","developer.apple.com","dart.dev","flutter.dev","lua.org","sqlite.org"
}

def domain(url): return urllib.parse.urlparse(url).netloc.lower().split(":")[0]
def is_trusted_domain(url):
    host=domain(url); return bool(host) and (host in TRUSTED_DOMAINS or any(host.endswith("."+d) for d in TRUSTED_DOMAINS))
def fingerprint(text): return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

def verify_url(url,timeout=8):
    if not isinstance(url,str) or urllib.parse.urlparse(url).scheme not in {"http","https"}: return {"valid":False,"reason":"unsupported scheme"}
    try:
        req=urllib.request.Request(url,method="HEAD",headers={"User-Agent":"PyIntell/0.2.0"})
        with urllib.request.urlopen(req,timeout=timeout) as r:
            final=r.geturl(); return {"valid":True,"status":getattr(r,"status",200),"url":final,"trusted_domain":is_trusted_domain(final),"domain":domain(final)}
    except Exception as exc:
        # Some sites reject HEAD. A small GET is a safer fallback than declaring the source dead.
        try:
            req=urllib.request.Request(url,method="GET",headers={"User-Agent":"PyIntell/0.2.0"})
            with urllib.request.urlopen(req,timeout=timeout) as r:
                final=r.geturl(); return {"valid":True,"status":getattr(r,"status",200),"url":final,"trusted_domain":is_trusted_domain(final),"domain":domain(final),"head_error":str(exc)}
        except Exception as fallback:
            return {"valid":False,"error":str(fallback),"trusted_domain":is_trusted_domain(url),"domain":domain(url)}

def rank_sources(sources,check=True):
    ranked=[]; seen=set()
    for source in sources:
        item=dict(source) if isinstance(source,dict) else {"url":str(source)}
        url=item.get("url",""); key=url.split("#")[0].rstrip("/")
        score=0
        if is_trusted_domain(url): score+=5
        if key not in seen: score+=2
        else: score-=3
        seen.add(key)
        verification=verify_url(url) if check else {"trusted_domain":is_trusted_domain(url)}
        if verification.get("valid"): score+=2
        item["verification"]=verification; item["score"]=score; ranked.append(item)
    return sorted(ranked,key=lambda x:x["score"],reverse=True)
