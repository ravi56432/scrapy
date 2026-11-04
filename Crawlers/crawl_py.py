#!/usr/bin/env python3
"""
sitecrawl.py — high-performance, robots-aware site crawler & link mapper.
No heavy browser, no content harvesting mid-crawl: it VISITS every page,
reads only what it needs (title / meta / links / a small text sample),
then classifies, deduplicates, and reports the whole link graph.

Usage
-----
    python sitecrawl.py https://example.com
    python sitecrawl.py https://example.com -c 32 --max-pages 500 --report both
    python sitecrawl.py https://example.com --no-robots --render   # JS-heavy sites
    python sitecrawl.py https://a.com https://b.com -o outdir

Key ideas
---------
* asyncio + httpx (HTTP/2, connection pooling) with N concurrent workers.
* Per-host politeness: global worker cap, per-host cap + token-bucket delay,
  and crawl-delay taken from robots.txt.
* Streaming: workers push results straight into the frontier/state, so memory
  stays flat no matter how big the site is; progress is checkpointed to disk.
* Link classification: internal / external / ignored (mailto, tel, js, data),
  page vs. file-asset, plus a semantic category guessed from the URL path
  (blog, docs, product, auth, legal, careers, ...) so output is easy to scan.
* No bulk content download: HTML is streamed, parsed once and dropped, keeping
  only title / meta / link graph / a --sample-chars text sample. Binary files
  are never downloaded — just HEAD-like metadata (2 KB probe at most).
* Outputs per site: report.html, report.md, pages.csv, links.csv, links.txt,
  crawl_index.json (resumable, written every ~15 s) and sitemap_crawled.xml.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import gzip
import heapq
import html
import json
import os
import random
import re
import signal
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import (
    parse_qsl,
    quote,
    unquote_plus,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

try:
    import httpx
except ImportError:  # pragma: no cover
    sys.exit("missing dependency: pip install 'httpx[http2]'")

try:                                    # HTTP/2 rides on the optional h2 package
    import h2
    _H2_OK = bool(h2)
except ImportError:                     # degrade to HTTP/1.1 instead of crashing
    _H2_OK = False

try:                                    # TLS/JA3 impersonation transport
    from curl_cffi import requests as curl_requests
    _CURL_OK = True
except ImportError:                     # plain httpx still works
    curl_requests = None                # type: ignore[assignment]
    _CURL_OK = False

try:
    import playwright
    _PW_OK = bool(playwright)
except ImportError:
    _PW_OK = False

try:
    from selectolax.lexbor import LexborHTMLParser as HTMLParser
    _PARSER = "lexbor"
except ImportError:  # pragma: no cover
    try:
        from selectolax.parser import HTMLParser
        _PARSER = "modest"
    except ImportError:
        HTMLParser = None
        _PARSER = "regex"  # degraded but still works

try:
    from protego import Protego
except ImportError:
    Protego = None

try:
    import tldextract

    # Offline / no network at import time: use the bundled snapshot of the
    # public suffix list so subdomain detection never blocks a crawl.
    _TLD = tldextract.TLDExtract(suffix_list_urls=())
except ImportError:  # pragma: no cover
    _TLD = None

try:
    import orjson

    def _dumps(obj: Any, indent: bool = False) -> str:
        return orjson.dumps(
            obj, option=orjson.OPT_INDENT_2 if indent else 0
        ).decode()
except ImportError:  # pragma: no cover
    def _dumps(obj: Any, indent: bool = False) -> str:
        return json.dumps(obj, indent=2 if indent else None, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════
#  configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class Config:
    start_urls: list[str] = field(default_factory=list)
    concurrency: int = 16           # global in-flight requests
    per_host: int = 4               # concurrent requests per host
    max_pages: int = 0              # 0 = unlimited
    max_depth: int = 100            # link-levels from the seed
    delay: float = 0.0              # base politeness delay, per host (s)
    jitter: float = 0.35            # ± random spread added to every delay
    timeout: float = 20.0
    max_retries: int = 2
    idle_grace: float = 12.0            # s to wait for the frontier to refill
    circuit_break: int = 4              # consecutive transport failures → skip host
    max_body_bytes: int = 3_000_000       # per-page buffering cap
    asset_probe_bytes: int = 2048         # binary assets: read this much, keep none
    sample_chars: int = 1200            # visible text kept per page (for summaries)
    max_links_per_page: int = 3000
    robots: bool = True
    sitemap: bool = True
    include_subdomains: bool = True
    only_host: bool = False
    respect_nofollow: bool = False      # rel=nofollow = "don't crawl", off by default
    verify_tls: bool = True             # --insecure flips this off
    # ── anti-bot ────────────────────────────────────────────────────
    impersonate: str = "chrome"         # curl_cffi TLS/JA3 profile, '' = plain httpx
    timezone: str = "Asia/Kolkata"      # browser session timezone
    auto_render: bool = True            # escalate walls/SPAs to the browser
    stealth: bool = True                # browser stealth shims
    block_retries: int = 2              # extra attempts after a block signal
    block_backoff: float = 6.0          # base seconds for block backoff
    solve_challenge: bool = True        # sit through JS challenges in-browser
    challenge_wait: float = 18.0        # max seconds to wait one out
    slow_host_backoff: float = 4.0      # multiplier for hosts that push back
    proxy_file: str = ""                # file with one proxy URL per line
    proxy_rotate: str = "per-request"   # per-request | per-host | off
    max_block_rate: float = 0.5         # >50% blocked on a host → slow to a crawl
    escalate_limit: int = 2             # failed browser attempts before giving up per host
    rotate_profiles: bool = True        # rotate TLS fingerprints on walls
    # ── depth ───────────────────────────────────────────────────────
    deep: bool = False                  # seed from search engines + archives
    archive_seed: bool = False          # include the Wayback CDX index
    search_queries: list[str] = field(default_factory=list)
    sitemap_only: bool = False
    follow_external: bool = False
    query_crawl: bool = False           # crawl URLs that differ only by ?query
    assets: bool = False                # crawl same-origin non-HTML files too
    render: bool = False                # Playwright fallback for JS-only pages
    render_limit: int = 200             # max pages sent to the renderer
    user_agent: str = (
        "Mozilla/5.0 (compatible; sitecrawl/2.0; +https://example.com/bot)"
    )
    out_dir: str = "crawl_output"
    checkpoint_every: float = 15.0
    report: str = "both"                # html | md | both | none
    resume: bool = False                # pick up an interrupted crawl_index.json
    verbose: bool = False


# URL path ↔ semantic category. First match wins, so order matters.
CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("auth",       re.compile(r"/(log[-_]?in|sign[-_]?in|sign[-_]?up|register|auth|oauth|sso|password|account|session)\b", re.I)),
    ("checkout",   re.compile(r"/(cart|checkout|basket|payment|order|billing|invoice|subscribe)\b", re.I)),
    ("product",    re.compile(r"/(products?|shop|store|item|catalog|collections?|pricing|plans?|buy)\b", re.I)),
    ("blog",       re.compile(r"/(blog|news|articles?|posts?|stories|insights|journal|press|magazine|updates?)\b", re.I)),
    ("docs",       re.compile(r"/(docs?|documentation|manual|guide|reference|api|sdk|wiki|help|kb|support|faq|tutorials?|how[-_]?to)\b", re.I)),
    ("changelog",  re.compile(r"/(changelog|releases?|whats[-_]?new|versions?|download)\b", re.I)),
    ("legal",      re.compile(r"/(privacy|terms?|tos|cookies?|legal|imprint|disclaimer|gdpr|dpa|licen[cs]e|accessibility|refund|shipping|returns?)\b", re.I)),
    ("about",      re.compile(r"/(about|company|who[-_]?we[-_]?are|our[-_]?story|mission|values|team|people|leadership|press|investors?)\b", re.I)),
    ("careers",    re.compile(r"/(careers?|jobs?|hiring|work[-_]?with[-_]?us|join[-_]?us|vacanc(y|ies)|positions?)\b", re.I)),
    ("contact",    re.compile(r"/(contact|get[-_]?in[-_]?touch|sales|demo|book|schedule|quote)\b", re.I)),
    ("media",      re.compile(r"/(gallery|photos?|images?|videos?|podcasts?|webinars?|events?|press[-_]?kit|media)\b", re.I)),
    ("community",  re.compile(r"/(community|forum|discuss|threads?|comments?|reviews?|testimonials?|customers?|case[-_]?stud(y|ies)|use[-_]?cases?)\b", re.I)),
    ("integrations", re.compile(r"/(integrations?|apps?|plugins?|marketplace|extensions?|partners?)\b", re.I)),
    ("dashboard",  re.compile(r"/(dashboard|admin|settings?|profile|console|panel|portal|my[-_]?account)\b", re.I)),
    ("search",     re.compile(r"/(search|filter|tags?|topics?|categories?|archive|page/\d+)\b", re.I)),
    ("locale",     re.compile(r"/([a-z]{2}([-_][a-z]{2})?|en[-_]us|fr|de|es|hi|ja|zh)(/|$)", re.I)),
]

ASSET_EXT = frozenset(
    "pdf doc docx xls xlsx ppt pptx odt rtf txt csv tsv json xml rss atom yml yaml "
    "zip rar 7z tar tgz gz bz2 xz zst dmg exe msi apk ipa deb rpm pkg whl "
    "iso img bin jar war kdbx "
    "jpg jpeg png gif webp avif svg ico bmp tiff heic eps "
    "mp4 webm mov avi mkv m4v ogv flv "
    "mp3 wav ogg oga opus flac m4a aac "
    "woff woff2 ttf otf eot "
    "css js mjs cjs map wasm".split()
)
PAGE_EXT = frozenset("html htm xhtml php asp aspx jsp cfm shtml do".split())

# Third-party plumbing that is never a useful destination link.
SKIP_HOSTS = re.compile(
    r"(^|\.)(google-analytics\.com|googletagmanager\.com|doubleclick\.net|"
    r"googleadservices\.com|googlesyndication\.com|facebook\.com|fb\.me|"
    r"fonts\.googleapis\.com|fonts\.gstatic\.com|gstatic\.com|"
    r"cdn\.jsdelivr\.net|unpkg\.com|cdnjs\.cloudflare\.com|"
    r"segment\.io|segment\.com|mixpanel\.com|amplitude\.com|hotjar\.com|"
    r"sentry\.io|newrelic\.com|nr-data\.net|cloudflareinsights\.com|"
    r"clarity\.ms|bat\.bing\.com|t\.co|bit\.ly)$",
    re.I,
)

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_name", "gclid", "dclid", "fbclid", "msclkid", "mc_cid",
    "mc_eid", "igshid", "_ga", "_gl", "ref_src", "ref_url", "spm", "scm",
    "yclid", "twclid", "li_fat_id", "vero_id", "wickedid", "oly_anon_id",
    "oly_enc_id", "s_kwcid", "elqTrackId", "elqTrack", "assetType",
    "assetId", "campaignId", "icid", "ncid", "cmpid", "trk", "trackingId",
}
SORT_PARAMS = {"sort", "order", "orderby", "dir", "view", "layout", "per_page", "page_size"}


# ══════════════════════════════════════════════════════════════════════
#  small utilities
# ══════════════════════════════════════════════════════════════════════

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def registrable(host: str) -> str:
    """example.co.uk from blog.example.co.uk (offline PSL snapshot)."""
    if not host:
        return host
    if _TLD is not None:
        try:
            ext = _TLD(host)
            if ext.suffix:
                return f"{ext.domain}.{ext.suffix}"
        except Exception:
            pass
    parts = host.lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def norm_host(url: str) -> str:
    """Hostname for scope checks: lowercase, no www., port kept when non-default."""
    u = urlsplit(url)
    host = (u.hostname or "").lower().removeprefix("www.")
    try:
        port = u.port
    except ValueError:
        port = None
    if port and port not in (80, 443):
        host = f"{host}:{port}"
    return host


def normalize_url(url: str, *, drop_query: bool = False) -> str | None:
    """Canonical form used for dedup: lowercase host, sorted+cleaned query,
    no fragment, no trailing slash on non-root paths, no duplicate slashes."""
    url = html.unescape(url.strip())
    if not url or url.startswith(("mailto:", "tel:", "javascript:", "data:", "#")):
        return None
    try:
        s = urlsplit(url)
    except ValueError:
        return None
    scheme = s.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    host = (s.hostname or "").lower()
    if not host or "." not in host:
        return None
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:                       # IPv6 literal
        host = f"[{host}]"
    try:
        port = s.port
    except ValueError:
        return None
    if port and port not in (80, 443):
        host = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", s.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
        path = quote(path, safe="/:@!$&'()*+,;=-._~%")
    else:
        path = "/"
    query = ""
    if s.query and not drop_query:
        pairs = [
            (k, v)
            for k, v in parse_qsl(s.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS
        ]
        pairs.sort(key=lambda kv: (kv[0] not in SORT_PARAMS, kv[0]))
        if pairs:
            query = urlencode(pairs, doseq=True, safe="/:,")
    return urlunsplit(("https" if scheme == "http" and port == 443 else scheme, host, path, query, ""))


def url_category(url: str) -> tuple[str, str]:
    """(section, kind) — kind separates assets from real pages."""
    s = urlsplit(url)
    path = s.path or "/"
    ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if ext in ASSET_EXT:
        kind = "asset"
    elif ext in PAGE_EXT or not ext:
        kind = "page"
    else:
        kind = "page"
    if path in ("", "/"):
        return "home", kind
    for name, rx in CATEGORY_RULES:
        if rx.search(path):
            return name, kind
    # fallback: first "meaningful" path segment (skip versions, ids, 1-char junk)
    for seg in (p for p in path.split("/") if p):
        stem = seg.rsplit(".", 1)[0].lower()
        if (
            len(stem) >= 3
            and not re.fullmatch(r"v?\d+([._-]\d+)*", stem)
            and not re.fullmatch(r"[0-9a-f]{8,}", stem)
            and stem not in ("index", "default", "home", "en", "www")
        ):
            return re.sub(r"[^a-z0-9._-]+", "-", stem)[:24], kind
    host = urlsplit(url).hostname or ""
    return ("home" if path in ("", "/") else f"other:{host[:20]}"), kind


def _decompress(raw: bytes) -> bytes:
    """Safety net for bodies httpx could not decode itself (br / zstd / gzip)."""
    if raw[:2] == b"\x1f\x8b":
        with contextlib.suppress(Exception):
            return gzip.decompress(raw)
    if raw[:4] == b"\x28\xb5\x2f\xfd":
        try:
            import zstandard as zstd
            return zstd.ZstdDecompressor().decompressobj().decompress(raw)
        except Exception:
            pass
    if raw[:1] == b"\x78":
        with contextlib.suppress(Exception):
            return zlib.decompress(raw)
    # brotli has no magic number: sniff for a plausible br stream
    for mod in ("brotli", "brotlicffi"):
        try:
            m = __import__(mod)
            out = m.decompress(raw)
            if out[:200].lstrip()[:5].lower().startswith((b"<!doc", b"<html", b"<?xml")):
                return out
        except Exception:
            continue
    return raw


_CAMEL = re.compile(r"([a-z0-9])([A-Z])")


def _fixup_host(url: str) -> str:
    """Normalisation hurts some real URLs; repair the ones we can reconstruct.

    ``docs.python.org`` treats the host case-sensitively, so a lowercased host
    starts returning 404s. CamelCase paths that were lowercased are equally
    unrecoverable, so the alphabetical "the..." article names are left alone.
    """
    p = urlsplit(url)
    host = p.netloc
    if not host or host == host.lower():
        return url
    segs = [sg for sg in p.path.split("/") if sg]
    if not segs:
        return url
    first = segs[0]
    if first and first[0].isupper() or _CAMEL.search(first) or " " not in first and any(
        ch.isupper() for ch in first
    ):
        return url  # looks like it needs its original casing — leave it
    return urlunsplit((p.scheme, host.lower(), p.path, p.query, ""))


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1:.1f}{unit}"
        n /= 1024
    return f"{n}B"


def short(s: str, n: int = 70) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


# ══════════════════════════════════════════════════════════════════════
#  HTML extraction (fast; Lexbor, one pass over the DOM)
# ══════════════════════════════════════════════════════════════════════

BLOCK = {"p", "div", "section", "article", "header", "footer", "li", "td", "th",
         "h1", "h2", "h3", "h4", "h5", "h6", "tr", "br", "hr", "blockquote",
         "pre", "figcaption", "dt", "dd", "table", "ul", "ol"}
SKIP_TEXT = {"script", "style", "noscript", "template", "svg", "iframe", "head"}
ROLE_SKIP = {"navigation", "banner", "contentinfo", "search", "menu", "menubar"}


def _clean(s: str) -> str:
    return " ".join(s.replace("\xa0", " ").split())


def _node_text(node) -> str:
    """Visible text of a subtree, block-level newlines preserved (iterative)."""
    out: list[str] = []
    stack = [node]
    while stack:
        n = stack.pop()
        tag = n.tag
        if tag in SKIP_TEXT:
            continue
        if tag == "-text":
            out.append(n.text_content)
            continue
        children = []
        c = n.child
        while c is not None:
            children.append(c)
            c = c.next
        if tag in BLOCK:
            out.append("\n")
        stack.extend(reversed(children))
    raw = "".join(out).replace("\xa0", " ")
    lines = [" ".join(l.split()) for l in raw.split("\n")]
    return "\n".join(l for l in lines if l).strip()


def _first(parser, *selectors: str):
    for sel in selectors:
        n = parser.css_first(sel)
        if n is not None:
            return n
    return None


def _attr(node, *names: str) -> str:
    if node is None:
        return ""
    a = node.attributes
    for n in names:
        v = a.get(n)
        if v:
            return _clean(v)
    return ""


def extract_head(parser) -> dict[str, Any]:
    """Cheap <head> pass — enough to classify a page before the body arrives."""
    title = _clean(_first(parser, "title") .text() if _first(parser, "title") else "")
    meta: dict[str, str] = {}
    for m in parser.css("meta"):
        a = m.attributes
        key = (a.get("property") or a.get("name") or a.get("http-equiv") or "").lower()
        val = a.get("content")
        if key and val and key not in meta:
            meta[key] = _clean(val)
    canon = _attr(_first(parser, 'link[rel="canonical"]'), "href")
    lang = ""
    h = _first(parser, "html")
    if h is not None:
        lang = (h.attributes.get("lang") or "")[:12]
    generator = meta.get("generator", "")
    return {
        "title": title,
        "description": meta.get("description") or meta.get("og:description", ""),
        "keywords": meta.get("keywords", "")[:300],
        "canonical": canon,
        "lang": lang,
        "og_title": meta.get("og:title", ""),
        "og_type": meta.get("og:type", ""),
        "og_image": meta.get("og:image", ""),
        "robots_meta": meta.get("robots", ""),
        "generator": generator,
        "framework": guess_framework(parser, generator),
    }


def guess_framework(parser, generator: str) -> str:
    g = generator.lower()
    for needle, name in (
        ("wordpress", "WordPress"), ("drupal", "Drupal"), ("joomla", "Joomla"),
        ("wix", "Wix"), ("squarespace", "Squarespace"), ("shopify", "Shopify"),
        ("webflow", "Webflow"), ("ghost", "Ghost"), ("hugo", "Hugo"),
        ("gatsby", "Gatsby"), ("next.js", "Next.js"), ("nuxt", "Nuxt"),
        ("docusaurus", "Docusaurus"), ("gitbook", "GitBook"),
    ):
        if needle in g:
            return name
    try:
        html_start = parser.root.inner_html[:4000].lower()
    except Exception:
        return ""
    for needle, name in (
        ("/_next/", "Next.js"), ("__nuxt", "Nuxt"), ("__gatsby", "Gatsby"),
        ("data-reactroot", "React"), ("ng-version", "Angular"), ("vue.", "Vue"),
        ("svelte", "Svelte"), ("astro-island", "Astro"), ("_sveltekit", "SvelteKit"),
        ("cdn.shopify.com", "Shopify"), ("wp-content", "WordPress"),
        ("drupal-settings", "Drupal"),
    ):
        if needle in html_start:
            return name
    return ""


def extract_links(parser, base_url: str, cap: int) -> list[tuple[str, str, str, bool]]:
    """All navigable links: (absolute_url, anchor_text, rel, nofollow)."""
    out: list[tuple[str, str, str, bool]] = []
    seen: set[str] = set()
    for a in parser.css("a[href]"):
        href = (a.attributes.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:", "data:")):
            continue
        absu = urljoin(base_url, href)
        if absu in seen:
            continue
        seen.add(absu)
        rel = _attr(a, "rel").lower()
        text = _clean(a.text())[:160]
        out.append((absu, text, rel, "nofollow" in rel))
        if len(out) >= cap:
            break
    return out


def sample_text(parser, limit: int) -> str:
    body = _first(parser, "main", "article", '[role="main"]', "#content", "#main", "body")
    if body is None:
        return ""
    txt = _node_text(body)
    return txt[:limit] if limit else ""


def find_meta_refresh(parser) -> str:
    n = _first(parser, 'meta[http-equiv="refresh"]')
    if n is None:
        return ""
    m = re.search(r"url\s*=\s*['\"]?([^'\";]+)", n.attributes.get("content") or "", re.I)
    return m.group(1).strip() if m else ""



# ══════════════════════════════════════════════════════════════════════
#  anti-bot: detection, adaptive pacing, transport impersonation
# ══════════════════════════════════════════════════════════════════════

# Signals that an HTTP response is an anti-bot wall rather than the real page.
# Checked against the first ~12 KB of the body plus header names, cheap enough
# to run on every response. Deliberately split from "just a 403" because the
# remedy differs: a Cloudflare interstitial wants a browser + JS challenge, a
# rate limiter wants patience.
BLOCK_SIGNALS: list[tuple[str, re.Pattern[str]]] = [
    ("cloudflare",   re.compile(r"cf-browser-verification|cf_chl_opt|"
                                r"__cf_chl_|challenge-platform|cf-challenge|"
                                r"attention required! \| cloudflare", re.I)),
    ("cloudflare",   re.compile(r"just a moment\.\.\.|checking your browser before accessing", re.I)),
    ("akamai",       re.compile(r"akamai bot manager|ak_bmsc|_abck|sec-if-cpt|"
                                r"reference #\d+\.[0-9a-f]+", re.I)),
    ("datadome",     re.compile(r"datadome|dd_cookie_test|geo\.captcha-delivery", re.I)),
    ("perimeterx",   re.compile(r"perimeterx|_px2|px-captcha|_pxhd", re.I)),
    ("imperva",      re.compile(r"incapsula|_incap_|visid_incap", re.I)),
    ("sucuri",       re.compile(r"sucuri website firewall|sucuri cloudproxy", re.I)),
    ("shape",        re.compile(r"f5 networks|bobcmn|shape security|bigipserver", re.I)),
    ("aws-waf",      re.compile(r"aws waf|captcha-container|aws-waf-token", re.I)),
    ("kasada",       re.compile(r"kasada|kpsdk|x-kpsdk", re.I)),
]

CHALLENGE_TITLES = re.compile(
    r"^(just a moment\.\.\.|attention required! \| cloudflare|checking your browser"
    r"|please wait|verifying you are human|one more step|are you a robot"
    r"|access denied|ddos-guard|security check|human verification"
    r"|请稍候|un instant|einen moment)", re.I)

BLOCK_STATUS = frozenset({401, 403, 405, 406, 409, 418, 429, 503})


def detect_block(status: int, headers: dict[str, str], head_text: str,
                 title: str = "") -> str:
    """Return a blocker name when this looks like an anti-bot wall, else ''."""
    h = {k.lower(): v for k, v in headers.items()}
    server = h.get("server", "").lower()
    if status in (200, 201):
        # a 200 can still be a challenge page
        if title and CHALLENGE_TITLES.search(title.strip()):
            return "challenge"
        if status == 200 and not head_text:
            return ""
    elif status not in BLOCK_STATUS and status < 400:
        return ""
    for name, rx in BLOCK_SIGNALS:
        if rx.search(head_text or ""):
            return name
    if "cloudflare" in server and status in BLOCK_STATUS:
        return "cloudflare"
    if "akamai" in server and status in BLOCK_STATUS:
        return "akamai"
    if "datadome" in h.get("x-datadome", "") or "datadome" in h.get("set-cookie", "").lower():
        return "datadome"
    if status == 429:
        return "rate-limit"
    if status in (401, 403) and ("captcha" in (head_text or "").lower()):
        return "captcha"
    return ""


# Navigator/screen shims injected before any page script runs. Covers the
# cheap checks that headless Chrome fails by default: webdriver flag, empty
# plugin and mime arrays, missing window.chrome, permission query mismatch,
# absent WebGL vendor strings, and the Notification/CSS media asymmetries.
STEALTH_INIT_JS = r"""
(() => {
  const undef = (obj, prop, val) => {
    try { Object.defineProperty(obj, prop, {get: () => val, configurable: true}); }
    catch (e) {}
  };
  undef(navigator, 'webdriver', undefined);
  undef(navigator, 'deviceMemory', 8);
  undef(navigator, 'hardwareConcurrency', 8);
  undef(navigator, 'language', 'en-US');
  undef(navigator, 'languages', Object.freeze(['en-US', 'en']));
  if (!window.chrome) { window.chrome = {}; }
  if (!window.chrome.runtime) {
    window.chrome.runtime = {connect: function(){}, sendMessage: function(){},
                            id: undefined};
  }
  if (!window.chrome.app) { window.chrome.app = {isInstalled: false}; }
  if (!window.chrome.csi) { window.chrome.csi = function(){ return {}; }; }
  if (!window.chrome.loadTimes) { window.chrome.loadTimes = function(){ return {}; }; }
  // plugins / mimeTypes as array-like
  const mkPlugin = (name, filename, desc) => {
    const p = Object.create(Plugin.prototype);
    Object.defineProperty(p, 'name', {get: () => name});
    Object.defineProperty(p, 'filename', {get: () => filename});
    Object.defineProperty(p, 'description', {get: () => desc});
    Object.defineProperty(p, 'length', {get: () => 1});
    return p;
  };
  const plugins = [
    mkPlugin('PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
    mkPlugin('Chrome PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
    mkPlugin('Chromium PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
    mkPlugin('Microsoft Edge PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
    mkPlugin('WebKit built-in PDF', 'internal-pdf-viewer', 'Portable Document Format'),
  ];
  plugins.item = (i) => plugins[i] || null;
  plugins.namedItem = (n) => plugins.find(p => p.name === n) || null;
  plugins.refresh = () => {};
  undef(navigator, 'plugins', plugins);
  undef(navigator, 'mimeTypes', plugins.map(p => p));
  // permissions.query must resolve like a real browser
  if (navigator.permissions && navigator.permissions.query) {
    const orig = navigator.permissions.query.bind(navigator.permissions);
    navigator.permissions.query = (params) => (
      params && params.name === 'notifications'
        ? Promise.resolve({state: Notification.permission, onchange: null})
        : orig(params)
    );
  }
  // WebGL vendor/renderer
  for (const Ctx of [window.WebGLRenderingContext, window.WebGL2RenderingContext]) {
    if (!Ctx || !Ctx.prototype.getParameter) continue;
    const gp = Ctx.prototype.getParameter;
    Ctx.prototype.getParameter = function (p) {
      if (p === 37445) return 'Intel Inc.';
      if (p === 37446) return 'Intel Iris Xe Graphics';
      if (p === 7936) return 'WebKit';
      if (p === 7937) return 'WebKit WebGL';
      return gp.call(this, p);
    };
  }
  // headless chrome reports no codecs through canPlayType oddly
  const CPT = HTMLMediaElement.prototype.canPlayType;
  HTMLMediaElement.prototype.canPlayType = function (t) {
    if (!t) return '';
    if (/^(audio|video)\/(mp4|webm|ogg)/.test(t)) return 'probably';
    return CPT.call(this, t);
  };
  // iframe contentWindow sanity + CDP detection
  if (window.RTCPeerConnection) {
    const RTC = window.RTCPeerConnection;
    window.RTCPeerConnection = function (...a) {
      const pc = new RTC(...a);
      pc.createDataChannel = () => ({close(){}, send(){}, addEventListener(){}});
      return pc;
    };
  }
  // Date/screen coherence
  undef(screen, 'colorDepth', 24);
  undef(screen, 'pixelDepth', 24);
})();
"""

# Headless "switches" that leak automation through fingerprinting APIs.
HEADLESS_HINTS = ("--headless=new", "--disable-blink-features=AutomationControlled",
                  "--disable-features=IsolateOrigins,site-per-process",
                  "--no-first-run", "--no-default-browser-check",
                  "--disable-infobars", "--disable-dev-shm-usage",
                  "--disable-setuid-sandbox", "--no-sandbox")

# Note: distributing requests across egress IPs is how a rate-limit wall is
# actually beaten; --proxy accepts a file of them so the crawler rotates
# per request and quarantines proxies that get challenged.
PROXY_SCHEMES = ("http://", "https://", "socks4://", "socks5://", "socks5h://")


# ══════════════════════════════════════════════════════════════════════
#  crawl state
# ══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class Page:
    url: str                 # normalized
    final_url: str           # after redirects
    depth: int
    status: int
    content_type: str
    elapsed_ms: int
    bytes_read: int
    title: str = ""
    description: str = ""
    canonical: str = ""
    lang: str = ""
    framework: str = ""
    section: str = "home"
    kind: str = "page"
    sample: str = ""
    redirect_to: str = ""
    error: str = ""
    links_out: int = 0
    internal_out: int = 0
    external_out: int = 0
    render_mode: str = "http"     # http | curl | render
    blocker: str = ""             # '' | cloudflare | akamai | datadome | rate-limit ...
    proxy: str = ""               # egress used for this request
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class LinkIndex:
    """Every distinct URL we ever saw, with where it came from and what it is."""

    __slots__ = ("rows",)

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def add(
        self,
        url: str,
        *,
        scope: str,               # internal | external | ignore
        category: str,
        kind: str,
        source: str | None,
        depth: int,
        anchor: str = "",
        rel: str = "",
        nofollow: bool = False,
        scheme: str = "",
        raw: str = "",
    ) -> None:
        row = self.rows.get(url)
        if row is None:
            row = {
                "url": url,
                "scope": scope,
                "category": category,
                "kind": kind,
                "scheme": scheme or urlsplit(url).scheme or "other",
                "host": urlsplit(url).hostname or "",
                "sources": [],
                "first_depth": depth,
                "anchor": anchor,
                "rel": rel,
                "nofollow": nofollow,
                "raw": raw or url,
                "visits": 0,
                "status": None,
                "title": "",
            }
            self.rows[url] = row
        else:
            if scope == "internal" and row["scope"] != "internal":
                row["scope"] = scope
            if depth < row["first_depth"]:
                row["first_depth"] = depth
            if anchor and not row["anchor"]:
                row["anchor"] = anchor
        if raw and raw != url and row.get("raw") == row["url"]:
            row["raw"] = raw
        if source and source != url and len(row["sources"]) < 8:
            if source not in row["sources"]:
                row["sources"].append(source)
        if nofollow:
            row["nofollow"] = True

    def mark_visited(self, url: str, page: Page) -> None:
        row = self.rows.get(url)
        if row is None:
            self.add(url, scope="internal", category=page.section, kind=page.kind,
                     source=None, depth=page.depth)
            row = self.rows[url]
        row["visits"] += 1
        row["status"] = page.status or None
        row["title"] = page.title or (page.error[:120] if page.error else "")
        if page.final_url and page.final_url != url:
            row["final_url"] = page.final_url

    def to_rows(self) -> list[dict[str, Any]]:
        return list(self.rows.values())


@dataclass(slots=True)
class Stats:
    requested: int = 0
    ok: int = 0
    redirects: int = 0
    client_err: int = 0
    server_err: int = 0
    failed: int = 0
    robots_blocked: int = 0
    skipped_dupe: int = 0
    bytes: int = 0
    started: float = 0.0

    def record(self, page: Page) -> None:
        self.requested += 1
        self.bytes += page.bytes_read
        s = page.status
        if page.error:
            self.failed += 1
        elif 200 <= s < 300:
            self.ok += 1
        elif 300 <= s < 400:
            self.redirects += 1
        elif 400 <= s < 500:
            self.client_err += 1
        elif s >= 500:
            self.server_err += 1


# ══════════════════════════════════════════════════════════════════════
#  the crawler
# ══════════════════════════════════════════════════════════════════════

class Crawler:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.stats = Stats(started=time.perf_counter())
        self.links = LinkIndex()
        self.pages: dict[str, Page] = {}
        self.queue: list[tuple[int, int, str]] = []      # (depth, seq, url) min-heap
        self.seen: set[str] = set()
        self.visited: set[str] = set()
        self.request_url: dict[str, str] = {}   # canonical -> original href
        self.seq = 0
        self.skipped: Counter[str] = Counter()
        self.robots: dict[str, Any] = {}
        self.robots_agents: dict[str, str] = {}
        self.host_gate: dict[str, asyncio.Semaphore] = {}
        self.host_next: dict[str, float] = {}
        self.host_delay: dict[str, float] = {}
        self.sitemap_urls: set[str] = set()
        self.blocked: set[str] = set()
        self.blocked_hosts: Counter[str] = Counter()   # host → wall hits
        self.blocked_pages = 0        # walls we could not get past
        self.recovered = 0            # walls the browser got past
        self.escalations: Counter[str] = Counter()   # host → failed browser attempts
        self.escalated_hosts: set[str] = set()
        self.walled: set[str] = set()      # hosts that beat both transports
        self._profile_i = 0
        self._warned_pw = False
        self.host_ok: Counter[str] = Counter()         # host → clean pages
        self.render_host: set[str] = set()             # hosts that need a browser
        self._curl = None                              # impersonated session
        self._attempts: dict[int, int] = {}
        self._rendered_raw: dict[str, bytes] = {}   # url → html from the browser
        self.retry_after = 0.0
        self.proxies = ProxyPool(cfg)
        self._httpx_client: Any = None
        self.host_fails: dict[str, int] = {}     # consecutive transport failures
        self.host_down: set[str] = set()         # circuit broken → skip host
        self.external_hosts: Counter[str] = Counter()
        self._stop = asyncio.Event()
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._last_work = time.monotonic()
        self._lock = asyncio.Lock()
        self._render_sem: asyncio.Semaphore | None = None
        self._renderer = None
        self._rendered = 0
        self._last_ckpt = time.monotonic()
        self.t0 = time.perf_counter()

    # ── seeds, robots, sitemaps ──────────────────────────────────────
    def compute_scope(self) -> None:
        self.seeds: list[str] = []
        self.roots: set[str] = set()
        self.seed_hosts: set[str] = set()
        for raw in self.cfg.start_urls:
            u = normalize_url(raw)
            if not u:
                sys.exit(f"invalid start url: {raw!r}")
            self.seeds.append(u)
            self.roots.add(registrable(norm_host(u)))
            self.seed_hosts.add(norm_host(u))
            self.links.add(u, scope="internal", category="home", kind="page",
                           source=None, depth=0)

    def in_scope(self, url: str) -> str | None:
        host = norm_host(url)
        if not host:
            return "ignore"
        if host in self.seed_hosts:
            return "internal"
        root = registrable(host)
        if root in self.roots:
            # another subdomain of the same registrable domain
            if self.cfg.only_host or not self.cfg.include_subdomains:
                return "external"
            return "internal"
        return "external"


    async def load_robots(self, client: httpx.AsyncClient, origin: str) -> None:
        if not self.cfg.robots or origin in self.robots:
            return
        if Protego is None:
            self.robots[origin] = None
            return
        try:
            r = await client.get(origin.rstrip("/") + "/robots.txt", timeout=self.cfg.timeout)
            if r.status_code == 200 and len(r.content) < 1_000_000:
                rp = Protego.parse(r.text)
                self.robots[origin] = rp
                cd = rp.crawl_delay(self.cfg.user_agent) or rp.crawl_delay("*")
                if cd:
                    self.host_delay[norm_host(origin)] = max(self.cfg.delay, float(cd))
                if self.cfg.sitemap:
                    grab = getattr(rp, "sitemaps", None)
                    maps = list(grab) if grab else re.findall(
                        r"(?im)^\s*sitemap:\s*(\S+)", r.text)
                    for sm in maps[:12]:
                        asyncio.create_task(self.fetch_sitemap(client, sm, 0))
                if self.cfg.verbose:
                    print(f"  robots: {origin}/robots.txt parsed"
                          + (f" (crawl-delay {cd}s)" if cd else ""))
            else:
                self.robots[origin] = None
        except Exception:
            self.robots[origin] = None

    def origin_of(self, url: str) -> str:
        s = urlsplit(url)
        return f"{s.scheme}://{s.netloc}"

    def robots_allowed(self, url: str) -> bool:
        if not self.cfg.robots:
            return True
        rp = self.robots.get(self.origin_of(url))
        if rp is None:
            return True
        try:
            return bool(rp.can_fetch(url, self.cfg.user_agent))
        except TypeError:
            return bool(rp.can_fetch(url))

    async def fetch_sitemap(self, client: httpx.AsyncClient, url: str, depth: int) -> None:
        if depth > 2 or len(self.sitemap_urls) > 50_000:
            return
        try:
            r = await client.get(url, timeout=self.cfg.timeout)
            if r.status_code != 200:
                return
            text = r.text
        except Exception:
            return
        if "<sitemapindex" in text:
            for loc in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text)[:50]:
                await self.fetch_sitemap(client, html.unescape(loc), depth + 1)
            return
        for loc in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text):
            u = normalize_url(html.unescape(loc))
            if u and self.in_scope(u) == "internal":
                self.sitemap_urls.add(u)
                self.enqueue(u, 1, source=None, anchor="", rel="sitemap")
        if self.cfg.verbose and self.sitemap_urls:
            print(f"  sitemap: {url} → {len(self.sitemap_urls)} urls known")

    # ── deep discovery ───────────────────────────────────────────────
    async def deep_seed(self, client: httpx.AsyncClient) -> None:
        """Find URLs the link graph cannot reach.

        Deep crawling is normally limited by what is *linked*. These sources
        break that ceiling:

        * sitemap.xml / robots.txt Sitemap: lines (already handled), including
          gzipped and indexed sitemaps
        * the Wayback CDX index, which lists URLs the site itself may have
          dropped, and is served by a third party — so an anti-bot wall on the
          target does not hide it
        * a path dictionary probed with cheap requests, which surfaces pages
          that no longer appear in any nav
        * optional search-engine queries (needs an API key or a scrape-friendly
          endpoint, so it's opt-in)
        """
        cfg = self.cfg
        origin = self.origin_of(self.seeds[0])
        hosts = sorted({norm_host(u) for u in self.seeds})
        found: set[str] = set()

        if cfg.sitemap:
            for seed in self.seeds:
                for cand in self._sitemap_candidates(seed):
                    await self.fetch_sitemap(client, cand, 0)
            found |= self.sitemap_urls
            if self.sitemap_urls:
                print(f"  sitemaps: {len(self.sitemap_urls)} urls")

        if cfg.archive_seed:
            for host in hosts:
                urls = await self._wayback_urls(host)
                for u in urls:
                    nu = normalize_url(u)
                    if nu and self.in_scope(nu) == "internal":
                        found.add(nu)
            print(f"  wayback: {len(found) - len(self.sitemap_urls)} extra urls")

        dict_hits = await self._probe_dictionary(client, origin)
        found |= dict_hits
        if dict_hits:
            print(f"  dictionary: {len(dict_hits)} live paths")

        for q in cfg.search_queries or [f"site:{hosts[0]}"]:
            for u in await self._search_urls(q):
                nu = normalize_url(u)
                if nu and self.in_scope(nu) == "internal":
                    found.add(nu)

        added = 0
        for u in sorted(found):
            if self.enqueue(u, 1, source=None, anchor="", rel="deep"):
                added += 1
        if added:
            print(f"  deep seed: +{added} urls queued")

    def _sitemap_candidates(self, seed: str) -> list[str]:
        o = self.origin_of(seed)
        return [f"{o}/sitemap.xml", f"{o}/sitemap_index.xml", f"{o}/sitemap-index.xml",
                f"{o}/sitemap1.xml", f"{o}/wp-sitemap.xml", f"{o}/sitemap.xml.gz",
                f"{o}/news-sitemap.xml", f"{o}/hreflang-sitemap.xml"]

    async def _wayback_urls(self, host: str, limit: int = 40_000) -> list[str]:
        """Pull historical URLs from the Wayback CDX API (JSON, paginated)."""
        out: list[str] = []
        # The CDX API is flaky under load and rejects some query shapes with a
        # bare 400, so try the shapes that are known to answer and detect its
        # "Temporarily Offline" HTML page rather than treating it as no data.
        variants = [
            f"url={quote(host + '/*', safe='*/')}&fl=original&collapse=urlkey&limit={limit}",
            f"url={quote(host, safe='')}&matchType=domain&fl=original&collapse=urlkey"
            f"&limit={limit}",
            f"url={quote(host + '/*', safe='*/')}&fl=original&collapse=urlkey"
            f"&filter=statuscode:200&limit={limit}",
        ]
        text = ""
        for q in variants:
            url = f"https://web.archive.org/cdx/search/cdx?{q}&output=json"
            try:
                sess = await self._curl_session()
                r = await sess.get(url, timeout=max(30.0, self.cfg.timeout)) if sess \
                    else await self.client_get(url)
            except Exception:
                continue
            body = r.text or ""
            if r.status_code == 200 and body.lstrip().startswith("["):
                text = body
                break
            if "Temporarily Offline" in body:
                if self.cfg.verbose:
                    print("  · wayback CDX is offline right now — skipping")
                return out
        if not text:
            if self.cfg.verbose:
                print(f"  · wayback returned no usable data for {host}")
            return out
        with contextlib.suppress(Exception):
            rows = json.loads(text)
            for row in rows[1:]:
                u = row[0] if isinstance(row, list) and row else ""
                if isinstance(u, str) and u.startswith("http"):
                    out.append(u)
            return out
        for line in text.splitlines():          # plain-text fallback
            line = line.strip()
            if line.startswith("http"):
                out.append(line)
        return out

    async def _probe_dictionary(self, client, origin: str,
                                cap: int = 300) -> set[str]:
        """Probe well-known paths to resurrect unlinked but live pages."""
        words = ("about contact pricing blog docs support faq careers jobs privacy "
                 "terms login signin signup register account dashboard admin api "
                 "status changelog releases download downloads products product "
                 "solutions services company team customers partners legal "
                 "security sitemap news events gallery press media investors "
                 "integrations marketplace search help community forum").split()
        found: set[str] = set()
        sem = asyncio.Semaphore(6)

        async def probe(word: str) -> None:
            url = f"{origin}/{word}"
            async with sem:
                try:
                    page, _ = await self.fetch(client, url, probe_only=True)
                except Exception:
                    return
                # 2xx/3xx = live; a soft-404 page ("Sorry, not found") is not
                if page.status < 400 and not page.blocker and not self._soft_404(page):
                    nu = normalize_url(page.final_url or url)
                    if nu and self.in_scope(nu) == "internal":
                        found.add(nu)

        await asyncio.gather(*(probe(w) for w in words[:cap]))
        return found

    @staticmethod
    def _soft_404(page: Page) -> bool:
        """Many sites answer 200 with a 'not found' page; treat it as a 404."""
        t = (page.title or "").lower()
        return any(k in t for k in ("404", "not found", "page not found",
                                    "doesn't exist", "does not exist", "no longer"))

    async def _search_urls(self, query: str, limit: int = 200) -> list[str]:
        """Best-effort search-engine discovery (no API key required)."""
        out: list[str] = []
        endpoints = [
            "https://html.duckduckgo.com/html/?q=" + quote(query),
            "https://lite.duckduckgo.com/lite/?q=" + quote(query),
        ]
        for ep in endpoints:
            try:
                sess = await self._curl_session()
                r = await sess.get(ep, timeout=max(20.0, self.cfg.timeout)) if sess \
                    else await self.client_get(ep)
                text = r.text
            except Exception:
                continue
            # DDG wraps outbound results as /l/?uddg=<encoded>
            for m in re.finditer(r"uddg=([^&\"']+)", text):
                u = unquote_plus(m.group(1))
                if u.startswith("http"):
                    out.append(u)
            for m in re.finditer(r'<a[^>]+href="(https?://[^"]+)"', text):
                u = html.unescape(m.group(1))
                if "duckduckgo.com" not in u:
                    out.append(u)
            if out:
                break
        if not out and self.cfg.verbose:
            print(f"  · search returned nothing for {query!r} "
                  f"(the engine may be walling this IP)")
        return out[:limit]

    async def client_get(self, url: str):
        """GET through whatever transport is live (curl_cffi or httpx)."""
        sess = await self._curl_session()
        if sess is not None:
            return await sess.get(url)
        return await self._httpx_client.get(url)

    # ── frontier ─────────────────────────────────────────────────────
    def enqueue(self, url: str, depth: int, *, source: str | None,
                anchor: str = "", rel: str = "", nofollow: bool = False,
                raw: str = "") -> bool:
        if url in self.seen or url in self.visited:
            self.stats.skipped_dupe += 1
            return False
        if self.cfg.max_pages and len(self.seen) >= self.cfg.max_pages:
            self.skipped["max_pages"] += 1
            return False
        if depth > self.cfg.max_depth:
            self.skipped["max_depth"] += 1
            return False
        host = norm_host(url)
        if host and SKIP_HOSTS.search(host):
            self.links.add(url, scope="ignore", category="tracker", kind="page",
                           source=source, depth=depth, anchor=anchor, rel=rel,
                           nofollow=nofollow, raw=raw)
            self.external_hosts[host] += 1
            return False
        scope = self.in_scope(url)
        section, kind = url_category(url)
        if scope == "external" and not self.cfg.follow_external:
            self.links.add(url, scope="external", category=section, kind=kind,
                           source=source, depth=depth, anchor=anchor, rel=rel,
                           nofollow=nofollow, raw=raw)
            self.external_hosts[norm_host(url)] += 1
            return False
        self.links.add(url, scope=scope, category=section, kind=kind, source=source,
                       depth=depth, anchor=anchor, rel=rel, nofollow=nofollow, raw=raw)
        if kind == "asset" and not self.cfg.assets:
            self.skipped["asset"] += 1
            return False
        if nofollow and scope == "internal" and self.cfg.respect_nofollow:
            self.skipped["nofollow"] += 1
            return False
        if self.cfg.sitemap_only and rel not in ("sitemap", "deep", "seed"):
            self.skipped["sitemap_only"] += 1
            return False
        if norm_host(url) in self.host_down:
            self.skipped["host down"] = self.skipped.get("host down", 0) + 1
            return False
        if not self.robots_allowed(url):
            if url not in self.blocked:
                self.blocked.add(url)
                self.stats.robots_blocked += 1
            if self.cfg.verbose:
                print(f"  · robots.txt disallows {url}")
            return False
        self.seen.add(url)
        if raw and raw != url:
            self.request_url.setdefault(url, raw)
        self.seq += 1
        heapq.heappush(self.queue, (depth, self.seq, url))
        return True

    async def next_job(self) -> tuple[int, str] | None:
        """Pop the shallowest URL; block while others are in flight.

        A worker only gives up after the frontier has stayed empty for
        `idle_grace` seconds AND nothing is in flight — otherwise a single very
        slow request could make every other worker exit early.
        """
        idle_since: float | None = None
        while True:
            if self.queue:
                depth, _, url = heapq.heappop(self.queue)
                self._last_work = time.monotonic()
                return depth, url
            if self._stop.is_set():
                return None
            if self._inflight == 0:
                # nothing running: give late callbacks a moment, then stop
                if idle_since is None:
                    idle_since = time.monotonic()
                if time.monotonic() - idle_since >= min(self.cfg.idle_grace, 2.0):
                    return None
                await asyncio.sleep(0.05)
                continue
            idle_since = None
            self._idle.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._idle.wait(), timeout=1.0)

    # ── politeness ───────────────────────────────────────────────────
    def gate_for(self, host: str) -> asyncio.Semaphore:
        sem = self.host_gate.get(host)
        if sem is None:
            sem = asyncio.Semaphore(max(1, min(self.cfg.per_host, self.cfg.concurrency)))
            self.host_gate[host] = sem
        return sem

    def adapt_pacing(self, host: str) -> None:
        """Hosts that push back get slower; clean hosts keep full speed.

        Block rate is measured on *responses*, so the pacing converges quickly:
        three clean pages restore normal speed, two walls in a row multiply the
        delay. This is what keeps a full crawl alive against a soft rate limiter
        instead of getting the whole IP banned halfway through.
        """
        hits = self.blocked_hosts.get(host, 0)
        ok = self.host_ok.get(host, 0)
        if not hits:
            self.host_delay.pop(host, None)
            return
        rate = hits / max(1, hits + ok)
        base = max(self.cfg.delay, 0.05)
        if rate > 0.6:
            self.host_delay[host] = min(base * self.cfg.slow_host_backoff * 8, 30.0)
        elif rate > 0.3:
            self.host_delay[host] = min(base * self.cfg.slow_host_backoff, 15.0)
        else:
            self.host_delay[host] = min(base * 2, 5.0)

    async def polite_wait(self, host: str) -> None:
        delay = self.host_delay.get(host, self.cfg.delay)
        if delay <= 0:
            return
        loop = asyncio.get_running_loop()
        ready = self.host_next.get(host, 0.0)
        now = loop.time()
        wait = max(0.0, ready - now)
        self.host_next[host] = max(now, ready) + delay + random.uniform(0, delay * self.cfg.jitter + 0.05)
        if wait:
            await asyncio.sleep(min(wait, 30.0))

    # ── fetching ─────────────────────────────────────────────────────
    # ── transports ───────────────────────────────────────────────────
    async def _curl_session(self) -> Any:
        """One impersonated session (JA3/H2/UA coherent) per crawler."""
        if self._curl is None:
            if not _CURL_OK:
                return None
            self._curl = curl_requests.AsyncSession(
                impersonate=self.cfg.impersonate,
                timeout=self.cfg.timeout,
                max_redirects=10,
                verify=self.cfg.verify_tls,
                default_headers=True,
                allow_redirects=True,
            )
        return self._curl

    async def fetch(self, client: httpx.AsyncClient, url: str,
                    *, referer: str = "", probe_only: bool = False
                    ) -> tuple[Page, bytes]:
        """Visit one URL. Prefers the impersonated TLS transport when available.

        Impersonation is not cosmetic: Cloudflare/Akamai/DataDome all fingerprint
        the TLS ClientHello (JA3/JA4) before they ever look at HTML. A stock
        `httpx` handshake is a known "python-httpx" fingerprint that gets 403'd
        on sight, whereas curl_cffi replays Chrome's exact cipher order.
        """
        cfg = self.cfg
        page = Page(url=url, final_url=url, depth=-1, status=0,
                    content_type="", elapsed_ms=0, bytes_read=0)
        t0 = time.perf_counter()
        hk = norm_host(url)
        if hk in self.host_down:
            page.error = f"skipped: {hk} unreachable (circuit open)"
            return page, b""
        kind = url_category(url)[1]
        is_binary = kind == "asset" and not url.lower().split("?")[0].endswith(
            (".html", ".htm", ".php", ".xml", ".txt", ".json"))
        budget = (cfg.asset_probe_bytes if is_binary or probe_only
                  else cfg.max_body_bytes)
        page.proxy = self.proxies.pick(hk) if self.proxies.enabled else ""
        self._attempts[id(page)] = 0

        use_curl = bool(cfg.impersonate) and _CURL_OK
        walled_host = hk in self.walled
        for attempt in range(0 if walled_host else cfg.max_retries + 1):
            self._attempts[id(page)] = attempt
            try:
                if attempt:
                    page.bytes_read = 0
                if use_curl:
                    got = await self._fetch_curl(url, page, budget,
                                                 is_binary or probe_only,
                                                 referer=referer)
                else:
                    got = await self._fetch_httpx(client, url, page, budget,
                                                  is_binary or probe_only,
                                                  referer=referer)
                if got is not None:
                    page.elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    return page, got
            except Exception as e:  # transport failure of any flavour
                page.error = f"{type(e).__name__}: {e}"[:200]
                retryable = not isinstance(
                    e, (httpx.ConnectError, httpx.ConnectTimeout,
                        httpx.UnsupportedProtocol)
                ) and "ConnectError" not in type(e).__name__
                if self.proxies.enabled and page.proxy and not retryable:
                    self.proxies.penalise(page.proxy, fatal=True)
                if attempt < cfg.max_retries:
                    delay = 0.6 * (2 ** attempt) + random.random() * 0.4
                    if page.blocker and hk not in self.walled:
                        delay = cfg.block_backoff * (2 ** attempt) + random.random()
                    if self.retry_after:
                        delay = max(delay, self.retry_after)
                        self.retry_after = 0.0
                    await asyncio.sleep(delay)
                    page.error = ""
                    continue
                page.elapsed_ms = int((time.perf_counter() - t0) * 1000)
                return page, b""
        return page, b""

    async def _fetch_httpx(self, client: httpx.AsyncClient, url: str, page: Page,
                           budget: int, is_binary: bool,
                           *, referer: str = "") -> bytes | None:
        headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
        if referer:
            headers["Referer"] = referer
            headers["Sec-Fetch-Site"] = "same-origin"
        async with client.stream("GET", url, headers=headers,
                                 follow_redirects=True) as r:
            return await self._consume(r.status_code, dict(r.headers), str(r.url),
                                       page, budget, is_binary,
                                       r.aiter_bytes(chunk_size=32768), r.aread,
                                       r.is_redirect)

    async def _fetch_curl(self, url: str, page: Page, budget: int,
                          is_binary: bool, *, referer: str = "") -> bytes | None:
        sess = await self._curl_session()
        if sess is None:
            return None
        kw: dict[str, Any] = {"allow_redirects": True}
        if referer:
            kw["referer"] = referer
        if page.proxy:
            kw["proxy"] = page.proxy
        async with sess.stream("GET", url, **kw) as r:
            async def aread() -> bytes:
                return await r.acontent()
            return await self._consume(r.status_code, dict(r.headers), str(r.url),
                                       page, budget, is_binary,
                                       r.aiter_content(chunk_size=32768), aread,
                                       False)

    async def _consume(self, status: int, headers: dict[str, str], final_url: str,
                       page: Page, budget: int, is_binary: bool,
                       chunks, aread, is_redirect: bool) -> bytes | None:
        """Shared response handling for both transports."""
        page.status = status
        page.final_url = final_url
        page.content_type = headers.get("content-type", headers.get("Content-Type", "")
                                        ).split(";")[0].strip().lower()
        ctype = page.content_type
        if is_redirect:
            loc = headers.get("location", headers.get("Location", ""))
            if loc:
                page.redirect_to = normalize_url(urljoin(final_url, loc)) or ""
        if status == 401 or status == 403 or status == 429 or 500 <= status < 600:
            body = await aread()
            page.bytes_read = len(body)
            probe = _decompress(body[:12288]).decode("utf-8", "ignore")
            page.blocker = detect_block(status, headers, probe)
            if page.blocker:
                self.blocked_hosts[norm_host(final_url)] += 1
                if self.proxies.enabled and page.proxy:
                    self.proxies.penalise(page.proxy)
                page.error = f"blocked by {page.blocker} (HTTP {status})"
                return None if page.blocker != "rate-limit" else b""
            page.error = f"HTTP {status}"
            if status == 429 or 500 <= status < 600:
                ra = headers.get("retry-after", headers.get("Retry-After"))
                if ra:
                    with contextlib.suppress(ValueError):
                        self.retry_after = min(float(ra), 30.0)
                return None
            return b""

        textual = (ctype.startswith(("text/html", "application/xhtml",
                                     "application/xml", "text/",
                                     "application/json")) or not ctype)
        if not textual or is_binary:
            self.host_ok[norm_host(final_url)] += 1
            page.kind = "asset"
            clen = headers.get("content-length", headers.get("Content-Length", ""))
            if clen.isdigit():
                page.content_type += f" ({clen}B)"
                page.bytes_read = int(clen)
            with contextlib.suppress(Exception):
                async for chunk in chunks:
                    page.bytes_read = max(page.bytes_read, len(chunk))
                    break
            return b""

        buf = bytearray()
        need_html = ctype.startswith(("text/html", "application/xhtml"))
        async for chunk in chunks:
            buf.extend(chunk)
            page.bytes_read += len(chunk)
            if len(buf) >= budget:
                page.truncated = True
                break
            if need_html and b"</body>" in chunk.lower():
                break
        if status in (200, 201) and buf:
            self.host_ok[norm_host(final_url)] += 1
            probe = _decompress(bytes(buf[:12288])).decode("utf-8", "ignore")
            title = ""
            m = re.search(r"<title[^>]*>(.*?)</title>", probe, re.S | re.I)
            if m:
                title = " ".join(html.unescape(m.group(1)).split())[:120]
            page.blocker = detect_block(status, headers, probe, title)
            if page.blocker:
                self.blocked_hosts[norm_host(final_url)] += 1
                page.error = f"blocked by {page.blocker} (challenge page)"
                return None
        return bytes(buf)

    def _attempt_for(self, page: Page) -> int:
        return self._attempts.get(id(page), 0)

    def decode(self, raw: bytes, ctype_header: str, fallback: str) -> str:
        raw = _decompress(raw)
        for enc in (fallback, "utf-8", "cp1252", "latin-1"):
            if not enc:
                continue
            with contextlib.suppress(Exception):
                return raw.decode(enc, errors="replace")
        return raw.decode("utf-8", errors="replace")

    # ── one page, end to end ─────────────────────────────────────────
    @staticmethod
    def flip_scheme(url: str) -> str:
        p = urlsplit(url)
        return urlunsplit(("http" if p.scheme == "https" else "https",
                           p.netloc, p.path, p.query, ""))

    async def process(self, client: httpx.AsyncClient, url: str, depth: int) -> Page:
        host = norm_host(url)
        page: Page | None = None
        raw: bytes = b""

        # A host that already answered with a wall goes straight to the browser:
        # retrying TLS-level requests against a JS challenge just burns time.
        if self.cfg.render and self._rendered < self.cfg.render_limit:
            page, raw = await self.fetch_rendered(url, depth, challenge=True)
            if page is not None and page.blocker:
                page = None                      # walled even in-browser
        if (page is None and self.cfg.auto_render and host in self.render_host
                and self._rendered < self.cfg.render_limit):
            rp, rraw = await self.fetch_rendered(url, depth, challenge=True)
            if rp is not None and not rp.blocker:
                page, raw = rp, rraw
            else:
                page = None                     # fall through to plain HTTP

        if page is None:
            page, raw = await self.fetch(client, url, referer=self.referer_for(url))
            if page.blocker and self.cfg.auto_render and host not in self.walled:
                recovered = await self.escalate(client, page, url, depth)
                if recovered is not None and not recovered.blocker:
                    # adopt the browser's payload — keeping the wall's empty
                    # body here is why recovered pages used to yield no links
                    page, raw = recovered, self._rendered_raw.get(url, b"")
                elif recovered is not None:
                    page = recovered
        # Normalisation (lowercasing, query cleanup) can 404 a URL that worked
        # verbatim. If we still have the original href, try it exactly once.
        orig = self.request_url.get(url)
        if (page.status in (404, 410) and orig and orig != url
                and self.in_scope(orig) == "internal"):
            retry, raw2 = await self.fetch(client, orig)
            if not retry.error and retry.status < 400:
                retry.depth = depth
                retry.url = url
                page, raw = retry, raw2
        if page.error and self.cfg.verify_tls and "CERTIFICATE_VERIFY_FAILED" in page.error:
            self.skipped["self-signed tls (use --insecure)"] = self.skipped.get(
                "self-signed tls (use --insecure)", 0) + 1
        if page.error and not page.status and (
            "SSL" in page.error or "Connect" in page.error or "ConnectTimeout" in page.error
        ):
            alt = self.flip_scheme(url)
            retry, raw2 = await self.fetch(client, alt)
            if not retry.error and retry.status < 400:
                retry.depth = depth
                retry.url = url
                page, raw = retry, raw2
        page.depth = depth

        ctype = page.content_type
        if raw and raw[:2] == b"\x1f\x8b" or raw[:4] == b"\x28\xb5\x2f\xfd":
            raw = _decompress(raw)
        is_html = ctype.startswith(("text/html", "application/xhtml")) or (
            not ctype and raw[:512].lstrip()[:15].lower().startswith((b"<!doctype", b"<html"))
        )

        # Rendered pages arrive as UTF-8 HTML with an error already cleared.  A
        # recovered challenge keeps the wall's error text on the record, so the
        # gate must look at the payload, not just at page.error.
        if is_html and raw and (not page.error or page.render_mode == "render"):
            page.error = ""
            charset = ""
            m = re.search(rb'charset=["\']?([\w\-]+)', raw[:4096], re.I)
            if m:
                charset = m.group(1).decode("ascii", "ignore")
            text = self.decode(raw, ctype, charset)
            del raw
            parser = None
            try:
                parser = HTMLParser(text) if HTMLParser else None
                if parser is None:
                    page.title = short(re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
                                       .group(1) if re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I) else "", 300)
                    page.links_out = 0
                else:
                    head = extract_head(parser)
                    page.title = head["title"] or head["og_title"]
                    page.description = head["description"]
                    page.canonical = head["canonical"]
                    page.lang = head["lang"]
                    page.framework = head["framework"]
                    if "noindex" in head["robots_meta"].lower():
                        page.sample = ""  # honour meta noindex for text sampling
                    page.sample = sample_text(parser, self.cfg.sample_chars)
                    self.harvest_links(url, page, parser, text)
            except Exception as e:
                page.error = f"parse: {type(e).__name__}: {e}"[:200]
            finally:
                text = ""

            # JS-only shell? try the optional renderer
            if (
                self.cfg.render
                and not page.error
                and parser is not None
                and page.links_out < 5
                and page.kind == "page"
            ):
                await self.maybe_render(client, page)

        section, kind = url_category(url)
        page.section = section
        if page.kind != "asset":
            page.kind = kind
        if not page.title and page.kind == "page" and not page.error:
            page.title = short(page.sample.split("\n")[0] if page.sample else "", 200)
        host = norm_host(page.final_url or url)
        if page.error and not page.status:
            n = self.host_fails.get(host, 0) + 1
            self.host_fails[host] = n
            if n >= self.cfg.circuit_break and host not in self.host_down:
                self.host_down.add(host)
                if self.cfg.verbose:
                    print(f"  ! {host} unreachable {n}x — skipping the rest of that host")
        else:
            self.host_fails.pop(host, None)
        self.pages[url] = page
        self.visited.add(url)
        self.adapt_pacing(norm_host(page.final_url or url))
        self.links.mark_visited(url, page)
        self.stats.record(page)
        return page

    # Impersonation profiles tried in order when a wall shows up. Each one is a
    # different JA3/JA4 + HTTP/2 SETTINGS fingerprint, so rotating genuinely
    # changes what the site's bot detector sees (unlike rotating the UA string).
    IMPERSONATE_ROTATION = ("chrome", "chrome136", "firefox133", "safari18",
                            "chrome131", "edge101")

    async def rotate_profile(self) -> str:
        """Switch the impersonated client to the next browser fingerprint."""
        if not _CURL_OK or not self.cfg.impersonate:
            return ""
        self._profile_i += 1
        profile = self.IMPERSONATE_ROTATION[self._profile_i % len(self.IMPERSONATE_ROTATION)]
        self.cfg.impersonate = profile
        if self._curl is not None:
            with contextlib.suppress(Exception):
                await self._curl.close()
            self._curl = None
        if self.cfg.verbose:
            print(f"  · switching TLS fingerprint → {profile}")
        return profile

    def referer_for(self, url: str) -> str:
        """Send a plausible Referer, as a browser following links would."""
        if not self.cfg.stealth:
            return ""
        src = self.links.rows.get(url, {}).get("sources") or []
        if src:
            return src[0]
        p = urlsplit(url)
        return f"{p.scheme}://{p.netloc}/" if p.path not in ("", "/") else ""

    async def escalate(self, client: httpx.AsyncClient, page: Page, url: str,
                       depth: int) -> Page | None:
        """Retry a blocked page through the stealth browser, waiting the wall out."""
        host = norm_host(url)
        # A host that beat the browser twice will beat it again: stop paying
        # the 15-20 s per page and record the wall instead.
        if self.escalations[host] >= self.cfg.escalate_limit:
            if host not in self.walled:
                self.walled.add(host)
                print(f"  ! {host} defeats both TLS impersonation and the browser "
                      f"— recording walls, not retrying. Try --proxy for fresh IPs.")
            self.skipped["host walled (browser exhausted)"] = self.skipped.get(
                "host walled (browser exhausted)", 0) + 1
            return None
        self.skipped["escalated to browser"] = self.skipped.get("escalated to browser", 0) + 1
        rendered, raw = await self.fetch_rendered(url, depth, challenge=True)
        if rendered is None or rendered.blocker:
            # still walled: remember the host so later pages skip the slow path
            self.blocked_pages += 1
            self.escalations[host] += 1
            if self.escalations[host] == 1 and self.cfg.rotate_profiles:
                await self.rotate_profile()
            if self.cfg.verbose:
                wall = (rendered.blocker if rendered else page.blocker)
                print(f"  ✗ {host} still blocked by {wall} even in-browser")
            return rendered
        self.render_host.add(host)
        self.escalated_hosts.add(host)
        self.host_fails.pop(host, None)
        self.recovered += 1
        if self.cfg.verbose:
            print(f"  ✓ {host} recovered via browser ({page.blocker} → {rendered.status})")
        return rendered

    async def fetch_rendered(self, url: str, depth: int, *, challenge: bool = False
                             ) -> tuple[Page | None, bytes]:
        """Load a URL in the stealth browser and return it as if it were HTML."""
        cfg = self.cfg
        if not _PW_OK:
            if not self._warned_pw:
                self._warned_pw = True
                print("  ! a browser is required for this host but playwright is "
                      "missing:\n    pip install playwright && playwright install "
                      "chromium", file=sys.stderr)
            return None, b""
        if self._rendered >= cfg.render_limit:
            return None, b""
        if self._renderer is None:
            self._renderer = _RenderPool(cfg)
            try:
                await self._renderer.start()
            except Exception as e:
                self._renderer = None
                print(f"  ! browser failed to start: {type(e).__name__}: {e}",
                      file=sys.stderr)
                return None, b""
        self._rendered += 1
        t0 = time.perf_counter()
        html_doc, final_url, status = await self._renderer.render(
            url, wait_for_challenge=challenge and cfg.solve_challenge
        )
        raw = html_doc.encode("utf-8", "replace")
        pg = Page(url=url, final_url=final_url or url, depth=depth, status=status or 200,
                  content_type="text/html", bytes_read=len(raw),
                  elapsed_ms=int((time.perf_counter() - t0) * 1000))
        pg.render_mode = "render"
        # A challenge that was passed still reports the wall's status code on the
        # original navigation, so judge the final DOM instead of the status.
        probe = raw[:12288].decode("utf-8", "ignore")
        m = re.search(r"<title[^>]*>(.*?)</title>", probe, re.S | re.I)
        title = " ".join(html.unescape(m.group(1)).split())[:120] if m else ""
        pg.blocker = detect_block(pg.status, {}, probe, title)
        if not pg.blocker and pg.status >= 400:
            # The wall was cleared: the navigation status belongs to the
            # interstitial, but the DOM we now hold is the real document.
            pg.status = 200
        if cfg.solve_challenge and pg.blocker in ("cloudflare", "challenge", "captcha"):
            # give the interstitial one more chance to clear itself
            html_doc, final_url, status = await self._renderer.render(
                url, wait_for_challenge=True)
            raw = html_doc.encode("utf-8", "replace")
            pg.final_url = final_url or pg.final_url
            pg.status = status or pg.status
            probe = raw[:12288].decode("utf-8", "ignore")
            pg.blocker = detect_block(pg.status, {}, probe, title)
        if pg.blocker:
            pg.error = f"blocked by {pg.blocker} (browser)"
        else:
            self._rendered_raw[url] = raw
        return pg, raw

    def harvest_links(self, base: str, page: Page, parser, raw_text: str) -> None:
        links = extract_links(parser, page.final_url or base, self.cfg.max_links_per_page)
        page.links_out = len(links)
        for absu, anchor, rel, nofollow in links:
            depth = page.depth + 1
            scope = self.in_scope(absu)
            if scope == "external":
                page.external_out += 1
            else:
                page.internal_out += 1
            if scope == "ignore":
                self.links.add(absu, scope="ignore", category="ignore", kind="page",
                               source=base, depth=depth)
                continue
            u = normalize_url(absu, drop_query=not self.cfg.query_crawl)
            if not u:
                continue
            self.enqueue(u, depth, source=base, anchor=anchor, rel=rel,
                         nofollow=nofollow, raw=absu)

        # meta refresh & canonical targets are real navigation signals
        for extra in (find_meta_refresh(parser), page.canonical):
            if not extra:
                continue
            u = normalize_url(urljoin(page.final_url or base, extra),
                              drop_query=not self.cfg.query_crawl)
            if u and u != base:
                self.enqueue(u, page.depth + 1, source=base, rel="canonical/refresh")

    # ── optional JS rendering fallback ───────────────────────────────
    async def maybe_render(self, client, page: Page) -> None:
        if self._rendered >= self.cfg.render_limit:
            return
        if not _PW_OK:
            if not self._warned_pw:
                self._warned_pw = True
                print("  ! --render needs playwright: pip install playwright && "
                      "playwright install chromium", file=sys.stderr)
            self.cfg.render = False
            return
        if self._renderer is None:
            self._renderer = _RenderPool(self.cfg)
            try:
                await self._renderer.start()
            except Exception as e:
                print(f"  ! browser failed to start ({type(e).__name__}): {e}",
                      file=sys.stderr)
                self._renderer = None
                self.cfg.render = False
                return
        self._rendered += 1
        try:
            html_doc = await self._renderer.render(page.final_url or page.url)
            if not html_doc:
                return
            parser = HTMLParser(html_doc)
            head = extract_head(parser)
            page.title = page.title or head["title"]
            page.description = page.description or head["description"]
            page.sample = sample_text(parser, self.cfg.sample_chars)
            page.render_mode = "render"
            self.harvest_links(page.url, page, parser, html_doc)
        except Exception as e:
            if self.cfg.verbose:
                print(f"  ! render failed {page.url}: {e}")

    # ── workers ──────────────────────────────────────────────────────
    async def worker(self, client: httpx.AsyncClient, wid: int) -> None:
        while not self._stop.is_set():
            job = await self.next_job()
            if job is None:
                return
            depth, url = job
            url = _fixup_host(url)
            host = norm_host(url) or url
            sem = self.gate_for(host)
            self._inflight += 1
            self._idle.clear()
            try:
                async with sem:
                    await self.polite_wait(host)
                    self._last_work = time.monotonic()
                    try:
                        page = await self.process(client, url, depth)
                    except Exception as e:  # never let one page kill a worker
                        page = Page(url=url, final_url=url, depth=depth, status=0,
                                    content_type="", elapsed_ms=0, bytes_read=0,
                                    error=f"{type(e).__name__}: {e}"[:200])
                        self.pages[url] = page
                        self.stats.record(page)
                        self.links.mark_visited(url, page)
            finally:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.set()
            self.log_page(page)
            if time.monotonic() - self._last_ckpt > self.cfg.checkpoint_every:
                self._last_ckpt = time.monotonic()
                self.checkpoint(final=False)

    def log_page(self, page: Page) -> None:
        n = len(self.pages)
        cap = self.cfg.max_pages or "∞"
        if self._use_tty:
            print(self.render_bar(n, cap, page))
        elif self.cfg.verbose or page.error:
            flag = "✗" if page.error else ("↪" if page.status in (301, 302, 307, 308) else "✓")
            print(f"[{n}/{cap}] {flag} d{page.depth} {page.status or '-'} "
                  f"{short(page.title or page.url, 60)}  {page.elapsed_ms}ms  "
                  f"→{page.internal_out}i/{page.external_out}e"
                  + (f"  !{page.error}" if page.error else ""))

    def render_bar(self, n: int, cap, page: Page) -> str:
        el = max(0.001, time.perf_counter() - self.t0)
        rate = n / el
        width = 28
        if isinstance(cap, int):
            filled = min(width, int(width * n / max(cap, 1)))
            bar = "█" * filled + "·" * (width - filled)
            frac = f"{n}/{cap}"
        else:
            bar = "█" * width
            frac = f"{n} pages"
        eta = ""
        if isinstance(cap, int) and 0 < n < cap:
            eta = f"  eta {int((cap - n) / rate) if rate else '?'}s"
        host = norm_host(page.url)
        return (f"\r\033[36m{bar}\033[0m {frac:>12}  {rate:5.1f}/s  "
                f"q{len(self.queue):<6} {host[:26]:<26} {short(page.section, 12):<12}"
                f"{eta}   ") + ("\033[K" if self._use_tty else "")

    # ── orchestration ────────────────────────────────────────────────
    def load_previous(self) -> int:
        """Re-hydrate pages/links/frontier from a previous run's checkpoint."""
        path = os.path.join(self.out_dir(), "crawl_index.json")
        if not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            return 0
        before = len(prev.get("pages", []))
        for pd in prev.get("pages", []):
            try:
                p = Page(**pd)
            except TypeError:
                continue
            self.pages[p.url] = p
            self.visited.add(p.url)
            self.links.mark_visited(p.url, p)
            self.stats.record(p)          # keep the counters truthful across runs
            self.stats.bytes += p.bytes_read
        for row in prev.get("links", []):
            if row.get("scope") == "ignore":
                continue
            cur = self.links.rows.get(row["url"])
            if cur is None:
                self.links.rows[row["url"]] = row
            else:  # keep the richer of the two records
                for k, v in row.items():
                    if not cur.get(k) and v:
                        cur[k] = v
        for depth, url in prev.get("frontier", []):
            url = normalize_url(url)
            if url and url not in self.visited and url not in self.seen:
                self.seen.add(url)
                self.seq += 1
                heapq.heappush(self.queue, (int(depth), self.seq, url))
        self.request_url.update(
            {r["url"]: r["raw"] for r in prev.get("links", [])
             if r.get("raw") and r.get("raw") != r.get("url")}
        )
        return before

    async def run(self) -> None:
        cfg = self.cfg
        self.compute_scope()
        restored = self.load_previous() if cfg.resume else 0
        if restored:
            print(f"  resume: {restored} pages and {len(self.queue)} queued urls "
                  f"restored from {self.out_dir()}/crawl_index.json")
        self._use_tty = sys.stdout.isatty()
        limits = httpx.Limits(
            max_connections=cfg.concurrency * 3,
            max_keepalive_connections=cfg.concurrency * 2,
            keepalive_expiry=60,
        )
        headers = {
            "User-Agent": cfg.user_agent,
            "Accept-Language": "en-US,en;q=0.9,*;q=0.5",
            "Cache-Control": "no-cache",
        }
        http2 = _H2_OK
        if cfg.impersonate:
            http2 = False        # curl_cffi carries its own ALPN/JA3 profile
        async with httpx.AsyncClient(
            http2=http2, limits=limits, headers=headers, timeout=cfg.timeout,
            follow_redirects=True, verify=cfg.verify_tls, trust_env=True,
            max_redirects=10,
        ) as client:
            self._client = client
            self._httpx_client = client
            # warm-up: robots + sitemaps for every seed host
            origins = {f"{urlsplit(s).scheme}://{urlsplit(s).netloc}" for s in self.seeds}
            await asyncio.gather(*(self.load_robots(client, o) for o in origins))
            for seed in self.seeds:
                self.enqueue(seed, 0, source=None, anchor="", rel="seed")
            if cfg.deep:
                await self.deep_seed(client)

            workers = [
                asyncio.create_task(self.worker(client, i))
                for i in range(max(1, cfg.concurrency))
            ]
            watchdog = asyncio.create_task(self.progress_loop())
            stallwatch = asyncio.create_task(self.stall_watch())
            try:
                await asyncio.gather(*workers)
            except asyncio.CancelledError:
                self._stop.set()
                raise
            finally:
                stallwatch.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stallwatch
                watchdog.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog
                if self._renderer is not None:
                    await self._renderer.stop()
                if self._curl is not None:
                    with contextlib.suppress(Exception):
                        await self._curl.close()
        if self._use_tty:
            print()
        await self.write_outputs()

    async def stall_watch(self) -> None:
        """Abort if nothing has completed for a long time and no work is left."""
        hard = max(60.0, self.cfg.timeout * 3)
        while not self._stop.is_set():
            await asyncio.sleep(min(5.0, hard / 2))
            if self._stop.is_set():
                return
            quiet = time.monotonic() - self._last_work
            if not self.queue and self._inflight == 0 and quiet > self.cfg.idle_grace:
                return
            if quiet > hard:
                print(f"  ! no progress for {quiet:.0f}s — stopping early",
                      file=sys.stderr)
                self._stop.set()
                self._idle.set()
                return

    async def progress_loop(self) -> None:
        """Periodic checkpoint + live counters, independent of page cadence."""
        while not self._stop.is_set():
            try:
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                return
            if self._stop.is_set():
                return
            self.checkpoint(final=False)

    # ── output ───────────────────────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        pages = list(self.pages.values())
        ok = [p for p in pages if not p.error]
        by_section = Counter(p.section for p in ok)
        by_status = Counter(p.status for p in pages)
        by_kind = Counter(p.kind for p in pages)
        link_rows = self.links.to_rows()
        by_scope = Counter(r["scope"] for r in link_rows)
        by_cat = Counter(r["category"] for r in link_rows if r["scope"] == "internal")
        visited = {u for u in self.pages}
        never_visited = [
            r for r in link_rows
            if r["scope"] == "internal" and r["url"] not in visited
        ]
        titles = Counter(p.title for p in ok if p.title)
        dup_titles = [(t, c) for t, c in titles.most_common(15) if c > 1]
        depth_hist = Counter(p.depth for p in ok)
        slowest = sorted(ok, key=lambda p: -p.elapsed_ms)[:10]
        broken = [p for p in pages if (p.error or (p.status and p.status >= 400))
                  and not p.blocker]
        blocked = [p for p in pages if p.blocker]
        thin = [p for p in ok if len(p.sample) < 200 and p.kind == "page"]
        no_title = [p for p in ok if not p.title and p.kind == "page"]
        elapsed = time.perf_counter() - self.stats.started
        return {
            "generated_at": now_iso(),
            "seeds": self.seeds,
            "root_domains": sorted(self.roots),
            "elapsed_s": round(elapsed, 2),
            "pages_per_sec": round(len(self.pages) / max(elapsed, 0.001), 2),
            "downloaded": human(self.stats.bytes),
            "counts": {
                "fetched": len(self.pages),
                "ok": self.stats.ok,
                "redirects": self.stats.redirects,
                "client_errors": self.stats.client_err,
                "server_errors": self.stats.server_err,
                "failed": self.stats.failed,
                "robots_blocked": self.stats.robots_blocked,
                "duplicates_dropped": self.stats.skipped_dupe,
                "urls_known": len(link_rows),
                "sitemap_urls": len(self.sitemap_urls),
            },
            "skipped": dict(self.skipped),
            "transport": self.transport_summary(),
            "blocked_count": len(blocked),
            "blocked_hosts": dict(self.blocked_hosts.most_common(40)),
            "host_blockers": self.host_blocker_map(),
            "by_status": {str(k): v for k, v in sorted(by_status.items())},
            "by_section": dict(by_section.most_common()),
            "by_kind": dict(by_kind),
            "link_scope": dict(by_scope),
            "internal_link_categories": dict(by_cat.most_common()),
            "depth_histogram": {str(k): v for k, v in sorted(depth_hist.items())},
            "external_hosts": dict(self.external_hosts.most_common(40)),
            "never_visited_internal": len(never_visited),
            "redirected": [{"from": p.url, "to": p.final_url}
                           for p in ok if p.final_url and p.final_url != p.url][:200],
            "duplicate_titles": dup_titles,
            "pages_without_title": len(no_title),
            "thin_pages": len(thin),
            "broken": [{"url": p.url, "status": p.status, "error": p.error}
                       for p in broken][:500],
            "slowest": [{"url": p.url, "ms": p.elapsed_ms, "bytes": p.bytes_read}
                        for p in slowest],
            "orphans_sample": [r["url"] for r in never_visited[:100]],
        }

    def transport_summary(self) -> dict[str, int]:
        modes = Counter(p.render_mode or "http" for p in self.pages.values())
        out = {
            "plain http/1.1+2 requests": modes.get("http", 0),
        }
        if modes.get("curl"):
            out["impersonated tls (curl_cffi)"] = modes["curl"]
        if modes.get("render"):
            out["browser renders"] = modes["render"]
        out["hosts that needed a browser"] = len(self.render_host)
        if self.escalated_hosts:
            out["hosts escalated (recovered or not)"] = len(self.escalated_hosts)
        out["proxy egress in use"] = len([p for p in self.proxies.proxies
                                          if p not in self.proxies.dead])
        return out

    def host_blocker_map(self) -> dict[str, list[str]]:
        m: dict[str, set[str]] = defaultdict(set)
        for p in self.pages.values():
            if p.blocker:
                m[norm_host(p.final_url or p.url)].add(p.blocker)
        return {h: sorted(v) for h, v in m.items()}

    def checkpoint(self, *, final: bool) -> None:
        """Write the growing index to disk so a crash/interrupt loses nothing."""
        path = os.path.join(self.out_dir(), "crawl_index.json")
        tmp = path + ".tmp"
        payload = {
            "state": "done" if final else "running",
            "summary": self.summary(),
            "pages": [p.as_dict() for p in self.pages.values()],
            "links": self.links.to_rows(),
            "queue_size": len(self.queue),
            "frontier": [[d, u] for d, _s, u in sorted(self.queue)[:100_000]],
        }
        with contextlib.suppress(Exception):
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(_dumps(payload, indent=False))
            os.replace(tmp, path)

    @staticmethod
    def out_name(url: str) -> str:
        host = norm_host(url)
        bare = host.rsplit(":", 1)[0] if ":" in host else host
        if bare == "localhost" or re.fullmatch(r"[\d.]+", bare):
            return re.sub(r"[^A-Za-z0-9._-]", "_", host)  # ip / loopback host:port
        return registrable(host) or bare or "site"

    def out_dir(self) -> str:
        root = self.out_name(self.seeds[0]) or "site"
        d = os.path.join(self.cfg.out_dir, re.sub(r"[^A-Za-z0-9._-]", "_", root))
        os.makedirs(d, exist_ok=True)
        return d

    async def write_outputs(self) -> None:
        d = self.out_dir()
        self.checkpoint(final=True)
        report = self.report_data()
        writer = ReportWriter(d, self.cfg, self.pages, self.links)
        await writer.write_all(report)
        elapsed = time.perf_counter() - self.stats.started
        c = report["summary"]["counts"]
        print(f"\n{'─'*68}")
        print(f" done in {elapsed:.1f}s · {len(self.pages)} pages · "
              f"{c['urls_known']} urls mapped · {human(self.stats.bytes)} moved")
        print(f" ok {c['ok']} · errors {c['failed'] + c['client_errors'] + c['server_errors']}"
              f" · robots-skipped {c['robots_blocked']} · dupes {c['duplicates_dropped']}")
        print(f" output → {d}/")
        for name in ("report.html", "report.md", "pages.csv", "links.csv", "links.txt"):
            if os.path.exists(os.path.join(d, name)):
                print(f"   · {name}")
        print(f"{'─'*68}")

    def report_data(self) -> dict[str, Any]:
        summary = self.summary()
        pages = [p.as_dict() for p in sorted(
            self.pages.values(), key=lambda p: (p.section, p.depth, p.url))]
        rows = self.links.to_rows()

        def sort_key(r):
            return ({"internal": 0, "external": 1, "ignore": 2}.get(r["scope"], 3),
                    r["category"], r["first_depth"], r["url"])

        links = sorted(rows, key=sort_key)
        grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in links:
            if r["scope"] == "ignore":
                continue
            grouped[r["scope"]][r["category"]].append(r)
        return {
            "summary": summary,
            "pages": pages,
            "links": links,
            "grouped": {s: dict(sorted(cats.items(), key=lambda kv: -len(kv[1])))
                        for s, cats in grouped.items()},
        }


class ReportWriter:
    """Writes the human and machine artifacts from crawled state."""

    def __init__(self, d: str, cfg: Config, pages: dict[str, Page],
                 links: LinkIndex) -> None:
        self.d = d
        self.cfg = cfg
        self.pages = pages
        self.links = links

    def path(self, name: str) -> str:
        return os.path.join(self.d, name)

    async def write_all(self, report: dict[str, Any]) -> None:
        tasks = [
            asyncio.to_thread(self.write_csv_pages),
            asyncio.to_thread(self.write_csv_links),
            asyncio.to_thread(self.write_txt_links, report),
        ]
        if self.cfg.report in ("md", "both"):
            tasks.append(asyncio.to_thread(self.write_markdown, report))
        if self.cfg.report in ("html", "both"):
            tasks.append(asyncio.to_thread(self.write_html, report))
        tasks.append(asyncio.to_thread(self.write_sitemap_xml, report))
        await asyncio.gather(*tasks)

    # ── machine readable ─────────────────────────────────────────────
    def write_csv_pages(self) -> None:
        cols = list(Page.__slots__)
        with open(self.path("pages.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for p in sorted(self.pages.values(), key=lambda x: x.url):
                w.writerow(p.as_dict())

    def write_csv_links(self) -> None:
        cols = ["scope", "category", "kind", "first_depth", "status", "visits",
                "url", "final_url", "anchor", "sources", "nofollow", "title", "raw_href"]
        with open(self.path("links.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in self.links.to_rows():
                w.writerow([
                    r["scope"], r["category"], r["kind"], r["first_depth"],
                    r["status"] or "", r["visits"], r["url"], r.get("final_url", ""),
                    r["anchor"], " ".join(r["sources"]), int(bool(r["nofollow"])),
                    r["title"], r.get("raw", ""),
                ])

    def write_sitemap_xml(self, report: dict[str, Any]) -> None:
        ok = [p for p in self.pages.values() if not p.error and 200 <= p.status < 300]
        lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
        for pg in sorted(ok, key=lambda x: x.url):
            lines.append(f"  <url><loc>{html.escape(pg.url)}</loc></url>")
        lines.append("</urlset>")
        with open(self.path("sitemap_crawled.xml"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def write_txt_links(self, report: dict[str, Any]) -> None:
        """Flat, greppable list — one URL per line, grouped and headered."""
        out: list[str] = []
        hostset = {norm_host(u) for u in self.cfg.start_urls}
        blocked_urls = {p.url for p in self.pages.values() if p.blocker}
        for scope, cats in report["grouped"].items():
            out.append(f"# ═══ {scope.upper()} ═══")
            for cat, rows in cats.items():
                out.append(f"\n# --- {cat} ({len(rows)}) ---")
                for r in rows:
                    flag = ""
                    if r["status"] and r["status"] >= 400:
                        flag = " [BROKEN]"
                    fin = r.get("final_url")
                    if fin and fin != r["url"]:
                        flag += f" [→ {fin}]"
                    if r["scope"] == "internal" and r.get("host") not in hostset:
                        flag += " [↗sub]"
                    if r["url"] in blocked_urls:
                        flag += " [BLOCKED]"
                    out.append(f"{r['url']}{flag}")
            out.append("")
        ign = [r for r in report["links"] if r["scope"] == "ignore"]
        if ign:
            out.append(f"\n# ═══ IGNORED ({len(ign)}) ═══")
            for r in ign:
                out.append(r["url"])
        with open(self.path("links.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(out))

    # ── markdown ─────────────────────────────────────────────────────
    def write_markdown(self, report: dict[str, Any]) -> None:
        s = report["summary"]
        c = s["counts"]
        L: list[str] = []
        A = L.append
        A(f"# Crawl report — {', '.join(s['seeds'])}\n")
        A(f"*{s['generated_at']} · {s['elapsed_s']}s · {s['pages_per_sec']} pages/s · "
          f"{s['downloaded']} transferred*\n")
        A("## Headline\n")
        A("| metric | value |\n|---|---|")
        for k, v in c.items():
            A(f"| {k.replace('_', ' ')} | {v} |")
        A("")
        A("## Transport & anti-bot\n")
        A("| metric | value |\n|---|---|")
        for k, v in s["transport"].items():
            A(f"| {k.replace('_', ' ')} | {v} |")
        A("")
        if s["blocked_hosts"]:
            A("Blocked hosts (and the wall that answered):\n")
            A("| host | hits | wall |\n|---|---|---|")
            for h, n in list(s["blocked_hosts"].items())[:40]:
                wall = ", ".join(s["host_blockers"].get(h, [])) or "unknown"
                A(f"| `{h}` | {n} | {wall} |")
            A("")
        A("## Pages by section\n")
        A("| section | pages |\n|---|---|")
        for k, v in list(s["by_section"].items())[:40]:
            A(f"| {k} | {v} |")
        A("")
        A("## Link map (internal, by category)\n")
        A("| category | urls |\n|---|---|")
        for k, v in list(s["internal_link_categories"].items())[:40]:
            A(f"| {k} | {v} |")
        A("")
        if s["external_hosts"]:
            A("## Outbound hosts\n")
            A("| host | links |\n|---|---|")
            for k, v in list(s["external_hosts"].items())[:30]:
                A(f"| {k} | {v} |")
            A("")
        A("## Health\n")
        A("| check | count |\n|---|---|")
        for k, v in [("broken", len(s["broken"])),
                     ("pages without title", s["pages_without_title"]),
                     ("thin pages (<200 chars)", s["thin_pages"]),
                     ("internal urls never visited", s["never_visited_internal"]),
                     ("blocked pages", s["blocked_count"])]:
            A(f"| {k} | {v} |")
        A("")
        if s["broken"]:
            A("### Broken / failing\n")
            for b in s["broken"][:60]:
                A(f"- `{b['status'] or '-'}` {b['url']} — {b['error']}")
            A("")
        if s.get("redirected"):
            A("## Redirects\n")
            A("| from | to |\n|---|---|")
            for r in s["redirected"][:60]:
                A(f"| <{r['from']}> | <{r['to']}> |")
            A("")
        for scope, cats in report["grouped"].items():
            A(f"## {scope.title()} links\n")
            for cat, rows in list(cats.items())[:40]:
                A(f"<details><summary><b>{cat}</b> — {len(rows)} urls</summary>\n")
                for r in rows[:300]:
                    title = f" — {r['title']}" if r["title"] else ""
                    A(f"- [{r['kind']}] <{r['url']}>{title}")
                if len(rows) > 300:
                    A(f"- …and {len(rows) - 300} more (see links.csv)")
                A("\n</details>\n")
        A("## All crawled pages\n")
        A("| d | status | section | transport | title | url |")
        A("|---|---|---|---|---|---|")
        for pg in report["pages"]:
            title = short(pg["title"], 60).replace("|", "\\|")
            A(f"| {pg['depth']} | {pg['status'] or '-'} | {pg['section']} | "
              f"{pg['render_mode'] or '-'} | {title} | <{pg['url']}> |")
        with open(self.path("report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(L))

    # ── html ─────────────────────────────────────────────────────────
    def write_html(self, report: dict[str, Any]) -> None:
        s = report["summary"]
        c = s["counts"]
        esc = html.escape

        def kpi(k: str, v: Any) -> str:
            return (f'<div class="kpi"><div class="v">{esc(str(v))}</div>'
                    f'<div class="k">{esc(k)}</div></div>')

        def bars(counter: dict[str, int], limit: int = 30) -> str:
            if not counter:
                return "<p class='muted'>none</p>"
            mx = max(counter.values()) or 1
            out = []
            for k, v in list(counter.items())[:limit]:
                pct = 100 * v / mx
                out.append(
                    f'<div class="bar"><span class="lbl" title="{esc(str(k))}">{esc(str(k))}</span>'
                    f'<span class="track"><i style="width:{pct:.1f}%"></i></span>'
                    f'<span class="n">{v}</span></div>')
            return "".join(out)

        def page_table(rows: list[dict], limit: int) -> str:
            if not rows:
                return "<p class='muted'>none</p>"
            body = []
            for r in rows[:limit]:
                bad = " class='bad'" if (r.get("error") or (r.get("status") or 0) >= 400) else ""
                transport = r.get("render_mode") or "-"
                if r.get("blocker"):
                    transport += f" ⚠{r['blocker']}"
                body.append(
                    f"<tr{bad}><td>{r.get('depth', '-')}</td>"
                    f"<td>{r.get('status') or '-'}</td>"
                    f"<td>{esc(str(r.get('section', '')))}</td>"
                    f"<td>{esc(transport)}</td>"
                    f"<td class='ttl'>{esc(short(str(r.get('title', '')), 70))}</td>"
                    f"<td class='u'><a href='{esc(r['url'])}' rel='noreferrer noopener'>{esc(r['url'])}</a></td>"
                    f"<td>{r.get('elapsed_ms', '')}</td></tr>")
            return ("<table class='tbl'><thead><tr><th>d</th><th>status</th><th>section</th>"
                    "<th>transport</th><th>title</th><th>url</th><th>ms</th></tr></thead>"
                    f"<tbody>{''.join(body)}</tbody></table>")

        sections = []
        for scope, cats in report["grouped"].items():
            blocks = []
            for cat, rows in list(cats.items()):
                table = page_table(rows, 400)
                blocks.append(
                    f"<details><summary><b>{esc(cat)}</b>"
                    f"<span class='pill'>{len(rows)}</span></summary>{table}</details>")
            sections.append(f"<h2 id='{esc(scope)}'>{esc(scope.title())} links "
                            f"<span class='pill'>{len(cats)} categories</span></h2>"
                            + "".join(blocks))

        blocked_rows = [
            {"first_depth": "-", "kind": "blocked", "anchor": pg["blocker"],
             "url": pg["url"], "sources": [], "status": pg["status"],
             "title": pg["error"], "render_mode": pg["render_mode"]}
            for pg in report["pages"] if pg["blocker"]
        ]

        doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crawl report — {esc(', '.join(s['seeds']))}</title>
<style>
:root{{--bg:#0d1117;--panel:#151b23;--line:#242c38;--fg:#e6edf3;--mut:#8b97a8;
--acc:#4dd4ac;--acc2:#5aa9ff;--bad:#ff7b72;--warn:#e3b341}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
header{{padding:28px 34px 18px;border-bottom:1px solid var(--line);
background:linear-gradient(160deg,#131a23,#0d1117)}}
h1{{margin:0 0 6px;font-size:21px;letter-spacing:-.2px}}
header small{{color:var(--mut);font-size:13px}}
h2{{margin:34px 0 12px;font-size:16px;display:flex;align-items:center;gap:10px}}
h3{{font-size:13px;color:var(--mut);margin:22px 0 8px;text-transform:uppercase;
letter-spacing:.08em}}
main{{padding:8px 34px 80px;max-width:1500px;margin:0 auto}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:10px;
margin:18px 0 6px}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.kpi .v{{font-size:20px;font-weight:650;color:var(--acc)}}
.kpi .k{{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:26px}}
.bar{{display:grid;grid-template-columns:150px 1fr 64px;align-items:center;gap:10px;
margin:5px 0;font-size:12.5px}}
.bar .lbl{{color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.bar .track{{background:#1c2530;border-radius:4px;height:8px;overflow:hidden}}
.bar .track i{{display:block;height:100%;background:linear-gradient(90deg,var(--acc),var(--acc2))}}
.bar .n{{text-align:right;font-variant-numeric:tabular-nums}}
.pill{{background:#1e2836;border:1px solid var(--line);border-radius:999px;
padding:1px 9px;font-size:11.5px;color:var(--acc)}}
details{{border:1px solid var(--line);border-radius:10px;margin:8px 0;background:var(--panel)}}
summary{{padding:10px 14px;cursor:pointer;font-size:13px;display:flex;gap:10px;
align-items:center}} summary::marker{{color:var(--mut)}}
details[open] summary{{border-bottom:1px solid var(--line)}}
.tbl{{width:100%;border-collapse:collapse;font-size:12.5px}}
.tbl th{{text-align:left;padding:6px 12px;color:var(--mut);font-weight:600;
font-size:11px;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--line)}}
.tbl td{{padding:5px 12px;border-bottom:1px solid #1b2230;vertical-align:top}}
.tbl tr:hover td{{background:#182029}}
.tbl .u a{{color:var(--acc2);text-decoration:none;word-break:break-all}}
.tbl .u a:hover{{text-decoration:underline}}
.tbl .ttl{{color:#c9d4e1;max-width:340px}}
tr.bad td{{background:#241417}} tr.bad .u a{{color:var(--bad)}}
.muted{{color:var(--mut)}}
nav.tabs{{display:flex;gap:8px;margin:16px 0 0;flex-wrap:wrap}}
nav.tabs a{{color:var(--mut);text-decoration:none;border:1px solid var(--line);
padding:5px 12px;border-radius:999px;font-size:12.5px;background:var(--panel)}}
nav.tabs a:hover{{color:var(--fg);border-color:var(--acc)}}
.warnbox{{border:1px solid var(--warn);background:#241f10;border-radius:10px;
padding:12px 16px;margin:14px 0;font-size:13px}}
</style></head><body>
<header>
<h1>Crawl report — {esc(', '.join(s['seeds']))}</h1>
<small>{esc(s['generated_at'])} · {s['elapsed_s']}s · {s['pages_per_sec']} pages/s ·
{s['downloaded']} transferred · roots: {esc(', '.join(s['root_domains']))}</small>
<div class="kpis">
{kpi('pages crawled', c['fetched'])}
{kpi('urls mapped', c['urls_known'])}
{kpi('internal', s['link_scope'].get('internal', 0))}
{kpi('external', s['link_scope'].get('external', 0))}
{kpi('browser renders', s['transport'].get('browser renders', 0))}
{kpi('blocked pages', s['blocked_count'])}
{kpi('errors', c['client_errors'] + c['server_errors'] + c['failed'])}
{kpi('pages/s', s['pages_per_sec'])}
</div>
<nav class="tabs">
<a href="#overview">Overview</a><a href="#antibot">Anti-bot</a>
<a href="#internal">Internal</a><a href="#external">External</a>
<a href="#health">Health</a><a href="#pages">All pages</a>
</nav>
</header><main>
<h2 id="overview">Overview</h2>
<div class="cols">
<div><h3>Pages by section</h3>{bars(s['by_section'])}</div>
<div><h3>Internal urls by category</h3>{bars(s['internal_link_categories'])}</div>
<div><h3>HTTP status</h3>{bars(s['by_status'])}</div>
<div><h3>Crawl depth</h3>{bars(s['depth_histogram'])}</div>
<div><h3>Outbound hosts</h3>{bars(s['external_hosts'])}</div>
</div>
<h2 id="antibot">Anti-bot &amp; transport</h2>
<div class="cols">
<div><h3>Transport mix</h3>{bars(s['transport'])}</div>
<div><h3>Blocked hosts</h3>{bars(s['blocked_hosts'])}</div>
<div><h3>Frontier / skips</h3>{bars(s['skipped'])}</div>
</div>
{f"<div class='warnbox'><b>{s['blocked_count']} pages were stopped by an anti-bot wall.</b> "
 f"If that is a large share of the crawl, re-run with <code>--render</code> "
 f"(browser escalation) or <code>--proxy proxies.txt</code> (fresh egress IPs).</div>"
 if s['blocked_count'] else ""}
{"<h3>Wall hits by page</h3>" + page_table(blocked_rows, 300) if blocked_rows else "<p class='muted'>No anti-bot interception detected 🎉</p>"}
{''.join(sections)}
<h2 id="health">Health</h2>
<div class="cols">
<div><h3>Issues</h3>{bars({'broken or failing': len(s['broken']),
'pages without title': s['pages_without_title'],
'thin pages (<200 chars)': s['thin_pages'],
'internal urls never visited': s['never_visited_internal'],
'blocked pages': s['blocked_count'],
'duplicate titles': len(s['duplicate_titles'])})}</div>
<div><h3>Slowest pages</h3>{bars({short(x['url'], 50): x['ms'] for x in s['slowest']}, 12)}</div>
</div>
<h3>Broken / failing urls</h3>
{page_table([{'depth': '-', 'section': 'error', 'title': b['error'], 'url': b['url'],
              'status': b['status'], 'render_mode': ''} for b in s['broken']], 400)
 if s['broken'] else "<p class='muted'>none 🎉</p>"}
<h2 id="pages">All crawled pages <span class="pill">{len(report['pages'])}</span></h2>
{page_table(report['pages'], 5000)}
</main></body></html>"""
        with open(self.path("report.html"), "w", encoding="utf-8") as f:
            f.write(doc)


class _RenderPool:
    """Stealth browser pool used to escalate JS walls and SPA shells.

    One Chromium, one persistent context: cookies and storage therefore survive
    across pages, which is exactly what a challenge-gated site expects from a
    real visitor. Pages are lazy-created per request and closed again so memory
    stays flat during long crawls.
    """

    STEALTH_ARGS = [
        "--disable-blink-features=AutomationControlled",
        "--disable-features=IsolateOrigins,site-per-process",
        "--no-first-run", "--no-default-browser-check", "--disable-infobars",
        "--disable-dev-shm-usage", "--disable-setuid-sandbox", "--no-sandbox",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--metrics-recording-only", "--mute-audio",
        "--window-size=1440,900",
    ]

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._pw = None
        self._browser = None
        self._ctx = None
        self._sem = asyncio.Semaphore(max(1, min(3, cfg.per_host)))
        self._started = False

    async def start(self) -> None:
        from playwright.async_api import async_playwright
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True, args=self.STEALTH_ARGS, chromium_sandbox=False,
        )
        self._ctx = await self._browser.new_context(
            user_agent=_browser_ua(self.cfg.impersonate),
            locale="en-US",
            timezone_id=self.cfg.timezone,
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
            is_mobile=False,
            has_touch=False,
            java_script_enabled=True,
            ignore_https_errors=not self.cfg.verify_tls,
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                           "image/avif,image/webp,*/*;q=0.8"),
                "sec-ch-ua": '"Chromium";v="140", "Not=A?Brand";v="24"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "Upgrade-Insecure-Requests": "1",
            },
        )
        self._ctx.set_default_timeout(int(self.cfg.timeout * 1000))
        if self.cfg.stealth:
            await self._ctx.add_init_script(STEALTH_INIT_JS)
            # hide the CDP/runtime surface that automation frameworks leave behind
            await self._ctx.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            )
        await self._ctx.route(re.compile(r"\.(png|jpe?g|gif|webp|avif|svg|ico|woff2?|"
                                         r"ttf|otf|eot|mp4|webm|mp3|ogg)($|\?)"),
                              lambda route: asyncio.ensure_future(route.abort()))
        self._started = True

    async def render(self, url: str, *, wait_for_challenge: bool = False,
                     proxy: str = "") -> tuple[str, str, int]:
        """Return (html, final_url, status). Waits out JS challenges if asked."""
        if not self._started:
            await self.start()
        assert self._ctx is not None
        async with self._sem:
            page = await self._ctx.new_page()
            try:
                resp = await page.goto(url, wait_until="domcontentloaded",
                                       referer=self._referer_for(url))
                status = resp.status if resp else 0
                # let hydration / lazy nav / challenge scripts finish
                await self._settle(page, status, wait_for_challenge)
                html_doc = await page.content()
                return html_doc, page.url, status
            finally:
                with contextlib.suppress(Exception):
                    await page.close()

    def _referer_for(self, url: str) -> str:
        p = urlsplit(url)
        return f"{p.scheme}://{p.netloc}/" if p.path not in ("", "/") else ""

    async def _settle(self, page, status: int, wait_for_challenge: bool) -> None:
        """Scroll, wait for the DOM to stabilise, and sit out a JS challenge."""
        cfg = self.cfg
        deadline = time.monotonic() + (cfg.challenge_wait if wait_for_challenge else 8.0)
        last_len = -1
        stable = 0
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                await page.mouse.wheel(0, 1200)   # trigger lazy content
            await page.wait_for_timeout(450)
            with contextlib.suppress(Exception):
                length = await page.evaluate("document.body ? document.body.innerText.length : 0")
                title = (await page.title()) or ""
                if CHALLENGE_TITLES.search(title.strip()) or (
                    "just a moment" in title.lower()
                ):
                    stable = 0            # still on the wall: keep waiting
                elif length == last_len:
                    stable += 1
                else:
                    stable = 0
                last_len = length
                if stable >= 2 and length > 200:
                    return
            if not wait_for_challenge and stable >= 1:
                return
        # final chance: wait for network to go quiet
        with contextlib.suppress(Exception):
            await page.wait_for_load_state("networkidle", timeout=3000)

    async def stop(self) -> None:
        with contextlib.suppress(Exception):
            if self._ctx:
                await self._ctx.close()
            if self._browser:
                await self._browser.close()
            if self._pw:
                await self._pw.stop()


def _browser_ua(profile: str) -> str:
    """A UA that matches the TLS fingerprint curl_cffi/Chromium actually sends."""
    major = {"firefox": "133", "safari": "18", "edge": "140"}.get(profile, "140")
    if profile.startswith("firefox"):
        return ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) "
                "Gecko/20100101 Firefox/133.0")
    if profile.startswith("safari"):
        return ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15")
    plat = "Edg/" if profile.startswith("edge") else "Chrome/"
    brand = f"Edg/{major}.0.0.0" if plat == "Edg/" else f"Chrome/{major}.0.0.0"
    return (f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            f"(KHTML, like Gecko) {brand} Safari/537.36")


class ProxyPool:
    """Rotating egress pool. Dead or challenged proxies are quarantined.

    Rate-limit and IP-reputation walls are beaten by spreading requests, so this
    is the single most effective lever when a site blocks a datacentre IP range.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.proxies: list[str] = []
        self.dead: set[str] = set()
        self.strikes: Counter[str] = Counter()
        self._host_affinity: dict[str, str] = {}
        self._i = 0
        if cfg.proxy_file:
            self._load(cfg.proxy_file)

    def _load(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "://" not in line:
                        line = "http://" + line
                    self.proxies.append(line)
        except OSError as e:
            print(f"  ! cannot read proxy file {path}: {e}", file=sys.stderr)
        if self.proxies:
            print(f"  proxies: {len(self.proxies)} loaded from {path}")

    @property
    def enabled(self) -> bool:
        return bool(self.proxies) and self.cfg.proxy_rotate != "off"

    def pick(self, host: str = "") -> str:
        if not self.enabled:
            return ""
        live = [p for p in self.proxies if p not in self.dead]
        if not live:
            return ""
        if self.cfg.proxy_rotate == "per-host":
            if host not in self._host_affinity:
                self._host_affinity[host] = live[self._i % len(live)]
                self._i += 1
            return self._host_affinity[host]
        self._i += 1
        return live[self._i % len(live)]

    def penalise(self, proxy: str, *, fatal: bool = False) -> None:
        if not proxy:
            return
        self.strikes[proxy] += 1
        if fatal or self.strikes[proxy] >= 3:
            self.dead.add(proxy)
            if self.cfg.verbose:
                print(f"  · proxy quarantined ({self.strikes[proxy]} strikes): "
                      f"{proxy.rsplit('@', 1)[-1]}")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sitecrawl",
        description="Deep-crawl a site and map/classify every link.",
        epilog="examples:\n"
               "  sitecrawl https://example.com\n"
               "  sitecrawl https://shop.example.com --deep --render -c 24\n"
               "  sitecrawl https://example.com --proxy proxies.txt --render-limit 500\n",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("urls", nargs="+", help="one or more start URLs")

    core = p.add_argument_group("crawl")
    core.add_argument("-c", "--concurrency", type=int, default=16)
    core.add_argument("--per-host", type=int, default=4)
    core.add_argument("-n", "--max-pages", type=int, default=0, help="0 = unlimited")
    core.add_argument("--max-depth", type=int, default=100)
    core.add_argument("-d", "--delay", type=float, default=0.0, help="per-host delay (s)")
    core.add_argument("--timeout", type=float, default=20.0)
    core.add_argument("--max-retries", type=int, default=2)
    core.add_argument("--max-body-mb", type=float, default=3.0, metavar="MB",
                      help="cap on how much of one document is buffered")
    core.add_argument("--sample-chars", type=int, default=1200)
    core.add_argument("-o", "--out", default="crawl_output")
    core.add_argument("--only-host", action="store_true",
                      help="stay on the exact hostname (no subdomains)")
    core.add_argument("--no-subdomains", action="store_true",
                      help="treat subdomains as external")
    core.add_argument("--external", action="store_true",
                      help="also crawl external domains")
    core.add_argument("--assets", action="store_true",
                      help="visit same-origin non-HTML files too (metadata only)")
    core.add_argument("--query", action="store_true",
                      help="treat ?query variants as distinct pages")
    core.add_argument("--no-robots", action="store_true", help="ignore robots.txt")
    core.add_argument("--no-sitemap", action="store_true", help="skip sitemap seeding")
    core.add_argument("--respect-nofollow", action="store_true",
                      help="skip rel=nofollow links instead of just flagging them")
    core.add_argument("--insecure", action="store_true",
                      help="do not verify TLS certificates")
    core.add_argument("--ua", default=None, help="custom User-Agent")
    core.add_argument("--circuit-break", type=int, default=4, metavar="N",
                      help="skip a host after N consecutive connection failures")

    deep = p.add_argument_group("depth")
    deep.add_argument("--deep", action="store_true",
                      help="harvest extra URLs from search engines, archives and "
                           "path dictionaries before crawling")
    deep.add_argument("--search-query", action="append", default=[], metavar="Q",
                      help="extra search query for --deep (repeatable)")
    deep.add_argument("--archives", action="store_true",
                      help="with --deep: also mine the Wayback CDX index")
    deep.add_argument("--sitemap-only", action="store_true",
                      help="crawl only URLs that came from sitemaps")

    ab = p.add_argument_group("anti-bot")
    ab.add_argument("--impersonate", default="chrome", metavar="PROFILE",
                    help="curl_cffi TLS/JA3 profile: chrome, chrome136, firefox133, "
                         "safari18, edge101 — '' disables impersonation")
    ab.add_argument("--no-stealth", action="store_true",
                    help="do not inject browser stealth shims")
    ab.add_argument("--no-render", action="store_true",
                    help="never escalate a blocked page to the browser")
    ab.add_argument("--render", action="store_true",
                    help="render every page in the browser instead of on demand")
    ab.add_argument("--render-limit", type=int, default=200,
                    help="cap on browser loads per crawl")
    ab.add_argument("--block-retries", type=int, default=2, metavar="N")
    ab.add_argument("--escalate-limit", type=int, default=2, metavar="N",
                    help="failed browser attempts per host before giving up on it")
    ab.add_argument("--no-rotate-profiles", action="store_true",
                    help="keep one TLS fingerprint instead of cycling on walls")
    ab.add_argument("--challenge-wait", type=float, default=18.0, metavar="S",
                    help="max seconds to wait out a JS challenge in-browser")
    ab.add_argument("--proxy", metavar="FILE",
                    help="file of proxy URLs (one per line) for egress rotation")
    ab.add_argument("--proxy-rotate", choices=["per-request", "per-host", "off"],
                    default="per-request")

    out = p.add_argument_group("output")
    out.add_argument("--report", choices=["html", "md", "both", "none"], default="both")
    out.add_argument("--resume", action="store_true",
                     help="continue an interrupted crawl in the same -o dir")
    out.add_argument("-v", "--verbose", action="store_true")
    return p


def cfg_from_args(argv: list[str]) -> Config:
    a = build_parser().parse_args(argv)
    cfg = Config(
        start_urls=a.urls,
        concurrency=max(1, a.concurrency),
        per_host=max(1, a.per_host),
        max_pages=max(0, a.max_pages),
        max_depth=max(0, a.max_depth),
        delay=max(0.0, a.delay),
        timeout=a.timeout,
        max_retries=max(0, a.max_retries),
        circuit_break=max(1, a.circuit_break),
        sample_chars=max(0, a.sample_chars),
        max_body_bytes=int(max(0.1, a.max_body_mb) * 1024 * 1024),
        robots=not a.no_robots,
        sitemap=not a.no_sitemap,
        include_subdomains=not (a.no_subdomains or a.only_host),
        only_host=a.only_host or a.no_subdomains,
        respect_nofollow=a.respect_nofollow,
        verify_tls=not a.insecure,
        follow_external=a.external,
        assets=a.assets,
        query_crawl=a.query,
        render_limit=a.render_limit,
        out_dir=a.out,
        report=a.report,
        resume=a.resume,
        verbose=a.verbose,
        # ── anti-bot ──
        impersonate="" if a.impersonate in ("", "none", "off") else a.impersonate,
        stealth=not a.no_stealth,
        auto_render=not a.no_render,
        render=a.render,
        block_retries=max(0, a.block_retries),
        escalate_limit=max(0, a.escalate_limit),
        rotate_profiles=not a.no_rotate_profiles,
        challenge_wait=max(1.0, a.challenge_wait),
        proxy_file=a.proxy or "",
        proxy_rotate=a.proxy_rotate,
        # ── depth ──
        deep=a.deep,
        archive_seed=a.archives,
        search_queries=a.search_query or [],
        sitemap_only=a.sitemap_only,
    )
    if a.ua:
        cfg.user_agent = a.ua
    return cfg


def main(argv: list[str] | None = None) -> int:
    cfg = cfg_from_args(sys.argv[1:] if argv is None else argv)
    crawler = Crawler(cfg)

    async def runner() -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()

        def bye() -> None:
            print("\n  interrupted — writing partial results…", file=sys.stderr)
            crawler._stop.set()
            if task:
                task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, bye)
        try:
            await crawler.run()
        except asyncio.CancelledError:
            crawler._stop.set()
            for w in asyncio.all_tasks(loop):
                if w is task:
                    continue
                w.cancel()
            await asyncio.sleep(0)
            await crawler.write_outputs()

    print(f"sitecrawl · samples: {'–' if not cfg.sample_chars else cfg.sample_chars}"
          f" · renderer: {'playwright' if cfg.render else 'off'}"
          f" · parser: {_PARSER} · robots: {'on' if cfg.robots else 'off'}")
    started = time.perf_counter()
    asyncio.run(runner())
    print(f"total wall clock: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
