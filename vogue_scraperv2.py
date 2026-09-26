"""
vogue_scraper_v2.py  —  Selenium-based Vogue Runway scraper
Handles Cloudflare JS challenges by driving a real Chrome browser.

Install:  pip install selenium webdriver-manager beautifulsoup4 requests
Run:      python vogue_scraper_v2.py --dry-run
          python vogue_scraper_v2.py --seasons spring-2024-ready-to-wear --designers gucci prada --max-looks 10
          python vogue_scraper_v2.py   (full scrape, all default seasons)
"""

import argparse
import csv
import json
import re
import time
import random
import requests
from pathlib import Path
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE            = "https://www.vogue.com"
SHOWS_BASE      = f"{BASE}/fashion-shows"
OUTPUT_DIR      = Path("./runway_images")
METADATA_CSV    = Path("./runway_metadata.csv")

DEFAULT_SEASONS = [
    "spring-2022-ready-to-wear",
    "fall-2022-ready-to-wear",
    "spring-2023-ready-to-wear",
    "fall-2023-ready-to-wear",
    "spring-2024-ready-to-wear",
]

# ── BROWSER ───────────────────────────────────────────────────────────────────

def make_driver(headless: bool = True) -> webdriver.Chrome:
    """
    Create a Chrome WebDriver that looks like a real browser to Cloudflare.
    Set headless=False if you want to watch the browser and debug.
    """
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")          # new headless mode (Chrome 112+)
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts,
    )
    # Patch navigator.webdriver to False so Cloudflare can't detect Selenium
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def get_soup(driver: webdriver.Chrome, url: str, wait_seconds: float = 4.0) -> BeautifulSoup:
    """
    Navigate to url, wait for JS to render, return BeautifulSoup of the page.
    Raises on hard failures.
    """
    driver.get(url)
    time.sleep(wait_seconds + random.uniform(0, 1.5))   # let Cloudflare JS run
    return BeautifulSoup(driver.page_source, "html.parser")


# ── NEXT_DATA EXTRACTION ──────────────────────────────────────────────────────

def extract_next_data(soup: BeautifulSoup) -> dict | None:
    """
    Vogue is a Next.js app. After JS renders, __NEXT_DATA__ is in the DOM.
    This is the most reliable source — structured JSON, not fragile HTML.
    """
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return None
    try:
        return json.loads(tag.string)
    except json.JSONDecodeError:
        return None


# ── SEASON INDEX ──────────────────────────────────────────────────────────────

def get_shows_for_season(driver: webdriver.Chrome, season: str) -> list[dict]:
    """
    Load the season index page and extract all designer show URLs.
    Returns: [ {designer, season, url}, ... ]
    """
    url = f"{SHOWS_BASE}/{season}"
    print(f"\n📋  Season: {season}")
    soup = get_soup(driver, url, wait_seconds=5)

    # ── Try 1: __NEXT_DATA__ ──────────────────────────────────────────────
    data = extract_next_data(soup)
    shows = []
    if data:
        shows = _shows_from_next_data(data, season)

    # ── Try 2: HTML <a> tags ──────────────────────────────────────────────
    if not shows:
        print("   ℹ  __NEXT_DATA__ empty, trying HTML link scan")
        shows = _shows_from_html(soup, season)

    # ── Try 3: scroll and retry (lazy-loaded grids) ───────────────────────
    if not shows:
        print("   ℹ  No links yet — scrolling page to trigger lazy load")
        _scroll_page(driver)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        shows = _shows_from_html(soup, season)

    print(f"   ✓  {len(shows)} shows found")
    return shows


def _shows_from_next_data(data: dict, season: str) -> list[dict]:
    shows = []
    pattern = re.compile(
        rf"/fashion-shows/{re.escape(season)}/([^/?#\"]+)"
    )

    def walk(node, depth=0):
        if depth > 12:
            return
        if isinstance(node, dict):
            # Look for any string value that looks like a show URL
            for v in node.values():
                if isinstance(v, str):
                    m = pattern.search(v)
                    if m:
                        designer = m.group(1)
                        full_url = f"{SHOWS_BASE}/{season}/{designer}"
                        if not any(s["designer"] == designer for s in shows):
                            shows.append({"designer": designer, "season": season, "url": full_url})
                else:
                    walk(v, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, depth + 1)

    walk(data)
    return shows


def _shows_from_html(soup: BeautifulSoup, season: str) -> list[dict]:
    pattern = re.compile(
        rf"/fashion-shows/{re.escape(season)}/([^/?#\"]+)$"
    )
    shows = []
    seen = set()
    for a in soup.find_all("a", href=True):
        m = pattern.search(a["href"])
        if m:
            designer = m.group(1)
            if designer not in seen:
                seen.add(designer)
                shows.append({
                    "designer": designer,
                    "season": season,
                    "url": f"{BASE}{a['href']}" if a["href"].startswith("/") else a["href"],
                })
    return shows


def _scroll_page(driver: webdriver.Chrome, scrolls: int = 5):
    """Scroll down to trigger lazy-loaded content."""
    for _ in range(scrolls):
        driver.execute_script("window.scrollBy(0, window.innerHeight);")
        time.sleep(1.2)


# ── SHOW PAGE ─────────────────────────────────────────────────────────────────

def scrape_show(driver: webdriver.Chrome, show: dict, max_looks: int | None) -> list[dict]:
    """
    Scrape all look image URLs from a single designer show page.
    Returns list of image metadata dicts.
    """
    soup = get_soup(driver, show["url"], wait_seconds=4)
    data = extract_next_data(soup)
    images = []

    if data:
        images = _images_from_next_data(data, show["designer"], show["season"])

    if not images:
        # Scroll to load all slideshow images then re-parse
        _scroll_page(driver, scrolls=8)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        images = _images_from_html(soup, show["designer"], show["season"])

    if max_looks:
        images = images[:max_looks]

    return images


def _images_from_next_data(data: dict, designer: str, season: str) -> list[dict]:
    """
    Recursively find all CDN image URLs in the Next.js JSON blob.
    Vogue CDN pattern: assets.vogue.com  or  media.vogue.com
    """
    cdn = re.compile(
        r'https://[^"\'>\s]*(?:assets|media)\.vogue\.com[^"\'>\s]+\.(?:jpg|jpeg|webp)',
        re.IGNORECASE,
    )
    found = []
    seen = set()

    def walk(node, depth=0):
        if depth > 14:
            return
        if isinstance(node, str):
            for url in cdn.findall(node):
                clean = re.sub(r'\?.*', '', url)
                if clean not in seen and _looks_like_look(clean):
                    seen.add(clean)
                    found.append(clean)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, depth + 1)

    walk(data)
    return _build_records(found, designer, season)


def _images_from_html(soup: BeautifulSoup, designer: str, season: str) -> list[dict]:
    cdn = re.compile(
        r'https://[^"\'>\s]*(?:assets|media)\.vogue\.com[^"\'>\s]+\.(?:jpg|jpeg|webp)',
        re.IGNORECASE,
    )
    found = []
    seen = set()

    # Check <img src>, <img data-src>, <img srcset>, and raw text in <script> tags
    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src"):
            val = img.get(attr, "")
            clean = re.sub(r'\?.*', '', val)
            if cdn.match(clean) and clean not in seen and _looks_like_look(clean):
                seen.add(clean)
                found.append(clean)
        srcset = img.get("srcset", "")
        if srcset:
            for part in srcset.split(","):
                url = part.strip().split(" ")[0]
                clean = re.sub(r'\?.*', '', url)
                if cdn.match(clean) and clean not in seen and _looks_like_look(clean):
                    seen.add(clean)
                    found.append(clean)

    # Also scan raw page source text for CDN URLs (catches JSON-in-script)
    for url in cdn.findall(soup.get_text()):
        clean = re.sub(r'\?.*', '', url)
        if clean not in seen and _looks_like_look(clean):
            seen.add(clean)
            found.append(clean)

    return _build_records(found, designer, season)


def _looks_like_look(url: str) -> bool:
    """
    Filter heuristic: skip thumbnails, logos, ads, and other non-look images.
    Runway look images on Vogue CDN have paths like /photos/XXXXXXX/ and are
    never tiny (thumbnails usually have 't_' or 'thumb' in the path).
    """
    low = url.lower()
    skip_patterns = ["thumb", "/t_", "logo", "icon", "avatar", "ad_", "banner", "sprite"]
    return not any(p in low for p in skip_patterns)


def _build_records(urls: list[str], designer: str, season: str) -> list[dict]:
    return [
        {
            "designer":    designer,
            "season":      season,
            "look_number": i + 1,
            "image_url":   f"{url}?w=1080&auto=format",   # request 1080px width from CDN
            "local_path":  None,
        }
        for i, url in enumerate(urls)
    ]


# ── DOWNLOADER ────────────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
})

def download_image(meta: dict, output_dir: Path) -> dict:
    designer = meta["designer"]
    season   = meta["season"]
    look_num = meta["look_number"]

    save_dir  = output_dir / season / designer
    save_dir.mkdir(parents=True, exist_ok=True)
    filename  = f"{designer}_{season}_look{look_num:03d}.jpg"
    save_path = save_dir / filename

    if save_path.exists():                     # resume-safe
        meta["local_path"] = str(save_path)
        return meta

    try:
        r = _session.get(meta["image_url"], timeout=20, stream=True)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            meta["local_path"] = str(save_path)
        else:
            print(f"      ⚠  HTTP {r.status_code} for look {look_num}")
    except Exception as e:
        print(f"      ✗  Download error: {e}")

    return meta


# ── CSV ───────────────────────────────────────────────────────────────────────

def append_csv(records: list[dict], path: Path):
    fields = ["designer", "season", "look_number", "image_url", "local_path"]
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in records])


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons",    nargs="+", default=DEFAULT_SEASONS)
    ap.add_argument("--designers",  nargs="+", default=None)
    ap.add_argument("--max-looks",  type=int,  default=None)
    ap.add_argument("--delay",      type=float,default=3.0)
    ap.add_argument("--output-dir", default="./runway_images")
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--no-headless",action="store_true",
                    help="Show the browser window (good for debugging)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 56)
    print("  VOGUE RUNWAY SCRAPER  (Selenium edition)")
    print(f"  Seasons  : {len(args.seasons)}")
    print(f"  Delay    : {args.delay}s + jitter")
    print(f"  Dry run  : {args.dry_run}")
    print(f"  Headless : {not args.no_headless}")
    print("═" * 56)

    headless = not args.no_headless
    driver   = make_driver(headless=headless)

    total_shows  = 0
    total_images = 0

    try:
        for season in args.seasons:
            shows = get_shows_for_season(driver, season)
            time.sleep(args.delay + random.uniform(0, 2))

            if args.designers:
                shows = [s for s in shows if s["designer"] in args.designers]
                print(f"   Filtered to {len(shows)} designers")

            for show in shows:
                total_shows += 1
                print(f"  [{total_shows:>4}]  {show['designer']:<35} {show['season']}")

                images = scrape_show(driver, show, args.max_looks)
                time.sleep(args.delay + random.uniform(0, 1.5))

                if images and not args.dry_run:
                    downloaded = [download_image(img, output_dir) for img in images]
                    append_csv(downloaded, METADATA_CSV)
                    total_images += len(downloaded)
                else:
                    total_images += len(images)

                print(f"         {len(images)} looks  |  running total: {total_images}")

    finally:
        driver.quit()

    print("\n" + "═" * 56)
    print(f"  ✓  COMPLETE")
    print(f"  Shows  : {total_shows}")
    print(f"  Images : {total_images}  ({'discovered' if args.dry_run else 'downloaded'})")
    if not args.dry_run:
        print(f"  CSV    : {METADATA_CSV}")
        print(f"  Images : {output_dir}/")
    print("═" * 56)


if __name__ == "__main__":
    main()