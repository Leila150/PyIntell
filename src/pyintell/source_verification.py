"""Evidence-oriented source verification primitives."""
from dataclasses import dataclass, field
import hashlib, re, time
from urllib.parse import urlparse
import urllib.request

@dataclass
class Evidence:
    url: str
    title: str = ""
    snippet: str = ""
    domain: str = ""
    score: float = 0.0
    reachable: bool = False
    official: bool = False
    notes: list[str] = field(default_factory=list)

@dataclass
class VerificationReport:
    claim: str
    evidence: list[Evidence] = field(default_factory=list)
    verdict: str = "unknown"
    confidence: float = 0.0
    contradictions: list[str] = field(default_factory=list)

_TRUSTED_SUFFIXES=(".gov", ".edu", ".ac.uk", ".mil")
_OFFICIAL_HINTS=("docs.", "developer.", "api.", "support.", "pypi.org", "github.com")

def domain(url): return urlparse(url).netloc.lower().split(":")[0]

def source_score(url, title="", snippet=""):
    d=domain(url); score=0.2; notes=[]
    if d.endswith(_TRUSTED_SUFFIXES): score+=0.35; notes.append("trusted institutional suffix")
    if any(x in d for x in _OFFICIAL_HINTS): score+=0.2; notes.append("official/developer-style domain")
    if urlparse(url).scheme=="https": score+=0.1; notes.append("HTTPS")
    if title and snippet: score+=0.1
    return min(score,1.0),notes

def verify_url(url, timeout=8):
    req=urllib.request.Request(url, method="HEAD", headers={"User-Agent":"PyIntell/0.2"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: reachable=200 <= r.status < 400
    except Exception: reachable=False
    score,notes=source_score(url)
    return Evidence(url=url, domain=domain(url), score=score+(0.15 if reachable else 0), reachable=reachable, official=score>=0.5, notes=notes)

def verify_sources(claim, sources, min_agreement=0.6):
    """Rank evidence; never treats a single source as proof by itself."""
    evidence=[]
    for src in sources:
        if isinstance(src,str): src={"url":src}
        url=src.get("url","")
        if not url: continue
        score,notes=source_score(url,src.get("title",""),src.get("snippet",""))
        evidence.append(Evidence(url,src.get("title",""),src.get("snippet",""),domain(url),score,notes=notes))
    evidence.sort(key=lambda x:x.score, reverse=True)
    usable=[e for e in evidence if e.snippet]
    normalized=[]
    for e in usable:
        text=re.sub(r"\W+"," ",e.snippet.lower()).strip()
        normalized.append(set(text.split()))
    agreement=0.0
    if len(normalized)>=2:
        pairs=[]
        for i in range(len(normalized)):
            for j in range(i+1,len(normalized)):
                a,b=normalized[i],normalized[j]; pairs.append(len(a&b)/max(1,len(a|b)))
        agreement=sum(pairs)/len(pairs)
    confidence=min(1.0, (evidence[0].score if evidence else 0)*0.6 + agreement*0.4)
    verdict="supported" if len(usable)>=2 and confidence>=min_agreement else "insufficient_evidence"
    return VerificationReport(claim,evidence,verdict,confidence)

def fingerprint(text): return hashlib.sha256(str(text).encode()).hexdigest()
