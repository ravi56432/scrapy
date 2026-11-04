"""
crawl_site.py — Deep-crawl an entire website, save every page + all links.
Usage:  python crawl_site.py https://example.com
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
)
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilterLXML

# ──────────────────────────────────────────────
# CONFIG — tweak these
# ──────────────────────────────────────────────
MAX_DEPTH       = 100   # how many link-levels deep (100 ≈ "unlimited")
MAX_PAGES       = 0     # 0 = no limit; set e.g. 500 to cap
DELAY           = 1.0   # seconds between pages (be polite)
OUTPUT_DIR      = "crawl_output"
HEADLESS        = True
CHECK_ROBOTS    = False  # respect robots.txt
# ──────────────────────────────────────────────


async def crawl(start_url: str):
    domain = urlparse(start_url).netloc
    out_dir = os.path.join(OUTPUT_DIR, domain.replace(":", "_"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"🕸  Crawling: {start_url}")
    print(f"   Depth: {MAX_DEPTH}  |  Page cap: {MAX_PAGES or '∞'}  |  Delay: {DELAY}s")
    print(f"   Output: {out_dir}/\n")

    strategy = BFSDeepCrawlStrategy(
        max_depth=MAX_DEPTH,
        max_pages=MAX_PAGES if MAX_PAGES > 0 else 999_999,
        include_external=False,          # stay on the same domain
    )

    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        deep_crawl_strategy=strategy,
        stream=True,                     # process pages as they finish
        check_robots_txt=CHECK_ROBOTS,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilterLXML(threshold=0.5),
        ),
        page_timeout=60_000,             # 60 s per page
        delay_before_return_html=30,
        excluded_tags=["header", "footer", "nav", "script", "style"],
        verbose=False,
    )

    browser_conf = BrowserConfig(headless=HEADLESS)

    all_pages   = []
    all_links   = set()
    errors      = []
    t0          = time.perf_counter()
    count       = 0

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        async for result in await crawler.arun(start_url, config=config):
            count += 1
            depth = result.metadata.get("depth", 0)

            if result.success:
                # collect links
                internal  = result.links.get("internal", [])
                external  = result.links.get("external", [])
                for link in internal:
                    all_links.add(link["href"])
                for link in external:
                    all_links.add(link["href"])

                # save markdown
                safe_name = (
                    result.url
                    .replace("https://", "")
                    .replace("http://", "")
                    .replace("/", "_")
                    .replace("?", "_")
                    .replace("&", "_")
                    .replace("=", "_")
                    .replace("#", "")
                    or "index"
                )
                md_path = os.path.join(out_dir, f"{count:04d}_d{depth}_{safe_name}.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(f"<!-- URL: {result.url} | Depth: {depth} -->\n\n")
                    f.write(result.markdown or "")

                page_record = {
                    "url":         result.url,
                    "depth":       depth,
                    "title":       (result.markdown or "").split("\n")[0][:200],
                    "chars":       len(result.markdown or ""),
                    "internal":    len(internal),
                    "external":    len(external),
                    "file":        os.path.basename(md_path),
                }
                all_pages.append(page_record)

                print(
                    f"  [{count:04d}] d{depth}  {result.url}  "
                    f"({len(result.markdown or '')} chars, "
                    f"{len(internal)} int / {len(external)} ext links)"
                )
            else:
                errors.append({"url": result.url, "error": result.error_message})
                print(f"  [{count:04d}] ❌ {result.url} — {result.error_message}")

            # polite delay
            await asyncio.sleep(DELAY)

    elapsed = time.perf_counter() - t0

    # ── save index ──
    index = {
        "site":        start_url,
        "domain":      domain,
        "crawled_at":  datetime.now().isoformat(),
        "elapsed_s":   round(elapsed, 2),
        "total_pages": len(all_pages),
        "total_links": len(all_links),
        "errors":      len(errors),
        "pages":       all_pages,
        "all_links":   sorted(all_links),
        "failed":      errors,
    }
    with open(os.path.join(out_dir, "_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ Done in {elapsed:.1f}s")
    print(f"   Pages crawled : {len(all_pages)}")
    print(f"   Total links   : {len(all_links)}")
    print(f"   Errors        : {len(errors)}")
    print(f"   Output dir    : {out_dir}/")
    print(f"   Index file    : {out_dir}/_index.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    asyncio.run(crawl(url))   