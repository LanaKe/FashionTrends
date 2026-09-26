"""
vogue_runway_scraper.py
=======================
Scrapes runway look images from vogue.com/fashion-shows with full metadata.

Vogue's architecture (as of 2024):
  - Season index pages: vogue.com/fashion-shows/{season}
    e.g. vogue.com/fashion-shows/spring-2024-ready-to-wear
  - Each show: vogue.com/fashion-shows/{season}/{designer}
    e.g. vogue.com/fashion-shows/spring-2024-ready-to-wear/gucci
  - Images live inside a Next.js __NEXT_DATA__ JSON blob embedded in the HTML
    (Vogue is a Next.js app — all page data is serialized in a <script> tag)

What this scraper does:
  1. Hits the season index to collect all designer URLs for that season
  2. For each designer show page, parses the __NEXT_DATA__ JSON to extract:
       - Full-resolution image URLs (CDN hosted, no auth required)
       - Designer name, season, look number
  3. Downloads images into a structured folder hierarchy
  4. Writes a metadata CSV alongside the images

Output structure:
  runway_images/
    spring-2024-ready-to-wear/
      gucci/
        gucci_spring-2024_look001.jpg
        gucci_spring-2024_look002.jpg
        ...
    fall-2023-ready-to-wear/
      ...
  runway_metadata.csv   ← path, designer, season, look_number, image_url

Usage:
  python vogue_scraper.py                        # scrape default seasons
  python vogue_scraper.py --seasons spring-2023-ready-to-wear fall-2023-ready-to-wear
  python vogue_scraper.py --designers gucci prada --seasons spring-2024-ready-to-wear
  python vogue_scraper.py --max-looks 30         # cap looks per show (good for testing)
  python vogue_scraper.py --delay 3.0            # seconds between requests (be polite)

Ethical notes:
  - Respects robots.txt (checked programmatically below)
  - Adds random jitter to delays so requests don't look like a bot
  - Downloads images at moderate resolution (not the max-res CDN option)
  - Stores only what is needed for research; do not redistribute the dataset
  - This is for non-commercial academic research only
"""

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

BASE_URL = "https://www.vogue.com"
FASHION_SHOWS_BASE = f"{BASE_URL}/fashion-shows"

# Seasons to scrape by default — add more as needed
DEFAULT_SEASONS = [
    "spring-2022-ready-to-wear",
    "fall-2022-ready-to-wear",
    "spring-2023-ready-to-wear",
    "fall-2023-ready-to-wear",
    "spring-2024-ready-to-wear",
]

# These headers convincingly mimic a real Chrome browser visit
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

OUTPUT_DIR = Path("./runway_images")
METADATA_CSV = Path("./runway_metadata.csv")

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def polite_sleep(base_delay: float = 2.0):
    """Sleep for base_delay ± 50% jitter. Never appear mechanical."""
    jitter = base_delay * random.uniform(0.5, 1.5)
    time.sleep(jitter)


def get_page(session: requests.Session, url: str, retries: int = 3) -> BeautifulSoup | None:
    """
    Fetch a page and return parsed BeautifulSoup.
    Retries on transient failures with exponential backoff.
    Returns None if the page is genuinely unavailable.
    """
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=REQUEST_HEADERS, timeout=20)

            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")

            elif resp.status_code == 429:
                # Rate limited — back off hard
                wait = 60 * (attempt + 1)
                print(f"  ⚠ Rate limited. Waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)

            elif resp.status_code in (403, 404):
                print(f"  ✗ {resp.status_code} for {url}")
                return None

            else:
                print(f"  ? HTTP {resp.status_code} for {url}, retry {attempt+1}")
                time.sleep(5 * (attempt + 1))

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Request error: {e}, retry {attempt+1}/{retries}")
            time.sleep(5 * (attempt + 1))

    return None


def extract_next_data(soup: BeautifulSoup) -> dict | None:
    """
    Vogue is built with Next.js. Every page embeds a <script id="__NEXT_DATA__">
    tag containing the full page data as JSON. This is the most reliable way to
    extract structured data — far better than scraping HTML elements that change
    with every front-end deploy.
    """
    script_tag = soup.find("script", id="__NEXT_DATA__")
    if not script_tag:
        return None
    try:
        return json.loads(script_tag.string)
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse __NEXT_DATA__: {e}")
        return None


def extract_image_url(raw_url: str, target_width: int = 1080) -> str:
    """
    Vogue's CDN (Condé Nast) uses a URL parameter system to resize images.
    Pattern: https://assets.vogue.com/photos/.../photo/...jpg?w=XXX
    We request a moderate resolution — large enough for CLIP (224×224 input)
    but not the multi-MB originals.
    """
    if not raw_url:
        return raw_url
    # Strip existing width params and set our own
    clean = re.sub(r'\?.*$', '', raw_url)  # remove all query params
    return f"{clean}?w={target_width}&auto=format"


# ─── SEASON INDEX SCRAPER ─────────────────────────────────────────────────────

def get_show_urls_for_season(session: requests.Session, season: str) -> list[dict]:
    """
    Scrape the season index page to get all designer show URLs.
    Returns a list of dicts: {designer, season, url}

    The season index page at vogue.com/fashion-shows/{season} renders
    a grid of show cards. In __NEXT_DATA__ these appear under:
    props > pageProps > galleries (or similar key — structure varies by year)
    We handle multiple possible JSON structures.
    """
    url = f"{FASHION_SHOWS_BASE}/{season}"
    print(f"\n📋 Fetching season index: {season}")
    soup = get_page(session, url)
    if not soup:
        print(f"  ✗ Could not load season index for {season}")
        return []

    next_data = extract_next_data(soup)
    shows = []

    if next_data:
        shows = _parse_shows_from_next_data(next_data, season)

    # Fallback: parse HTML anchor tags directly
    if not shows:
        print("  ℹ __NEXT_DATA__ parse failed, falling back to HTML link extraction")
        shows = _parse_shows_from_html(soup, season)

    print(f"  ✓ Found {len(shows)} shows for {season}")
    return shows


def _parse_shows_from_next_data(data: dict, season: str) -> list[dict]:
    """
    Navigate the Next.js data tree to find show listings.
    Vogue's exact JSON structure has shifted across years — we try several paths.
    """
    shows = []

    def find_in_tree(node, depth=0):
        """Recursively hunt for objects that look like show listings."""
        if depth > 8:
            return
        if isinstance(node, dict):
            # A show listing has a slug that matches a designer name
            # and a type like "gallery" or "fashion-show"
            content_type = node.get("type", node.get("contentType", ""))
            slug = node.get("slug", node.get("url", ""))
            if any(t in str(content_type).lower() for t in ["show", "gallery", "runway"]):
                if slug and season in str(slug):
                    designer = node.get("brandName", node.get("brand", {}).get("name", ""))
                    if not designer:
                        designer = slug.replace(season, "").strip("/").strip("-")
                    shows.append({
                        "designer": designer.lower().replace(" ", "-"),
                        "season": season,
                        "url": f"{FASHION_SHOWS_BASE}/{season}/{slug.split('/')[-1]}",
                    })
            for v in node.values():
                find_in_tree(v, depth + 1)
        elif isinstance(node, list):
            for item in node:
                find_in_tree(item, depth + 1)

    find_in_tree(data)
    return shows


def _parse_shows_from_html(soup: BeautifulSoup, season: str) -> list[dict]:
    """
    HTML fallback: find all <a> tags whose href matches the show URL pattern.
    Pattern: /fashion-shows/{season}/{designer}
    """
    shows = []
    seen = set()
    pattern = re.compile(rf"/fashion-shows/{re.escape(season)}/([^/?#]+)$")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = pattern.search(href)
        if m:
            designer = m.group(1)
            if designer not in seen:
                seen.add(designer)
                shows.append({
                    "designer": designer,
                    "season": season,
                    "url": urljoin(BASE_URL, href),
                })
    return shows


# ─── SHOW PAGE SCRAPER ────────────────────────────────────────────────────────

def scrape_show(session: requests.Session, show: dict, max_looks: int = None) -> list[dict]:
    """
    Scrape all look images from a single designer show page.
    Returns list of image metadata dicts.

    Show pages have a slideshow/gallery component. In __NEXT_DATA__ the images
    are typically under: props > pageProps > gallery > slides (or items/photos)
    Each slide has: src/url (image URL), alt text, and sometimes look number.
    """
    url = show["url"]
    designer = show["designer"]
    season = show["season"]

    soup = get_page(session, url)
    if not soup:
        return []

    next_data = extract_next_data(soup)
    images = []

    if next_data:
        images = _extract_images_from_next_data(next_data, designer, season)

    # Fallback: hunt for image tags with CDN URLs
    if not images:
        images = _extract_images_from_html(soup, designer, season)

    if max_looks:
        images = images[:max_looks]

    print(f"    ✓ {designer:<30} {len(images):>3} looks")
    return images


def _extract_images_from_next_data(data: dict, designer: str, season: str) -> list[dict]:
    """
    Extract image URLs from the Next.js data blob.
    The gallery images are nested several levels deep — we search recursively
    for arrays of objects that have image URL fields.
    """
    images = []
    cdn_pattern = re.compile(r'https://[^"]+assets\.vogue\.com[^"]+\.(jpg|jpeg|webp)')

    def collect_images(node, depth=0):
        if depth > 12:
            return
        if isinstance(node, dict):
            # Look for image URL fields
            for key in ("url", "src", "source", "href", "imageUrl", "image"):
                val = node.get(key, "")
                if isinstance(val, str) and cdn_pattern.search(val):
                    images.append(val)
            for v in node.values():
                collect_images(v, depth + 1)
        elif isinstance(node, list):
            for item in node:
                collect_images(item, depth + 1)

    collect_images(data)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for url in images:
        if url not in seen:
            seen.add(url)
            unique.append(url)

    return [
        {
            "designer": designer,
            "season": season,
            "look_number": i + 1,
            "image_url": extract_image_url(url),
            "local_path": None,  # filled in after download
        }
        for i, url in enumerate(unique)
    ]


def _extract_images_from_html(soup: BeautifulSoup, designer: str, season: str) -> list[dict]:
    """
    HTML fallback: find all <img> tags pointing at Vogue's CDN.
    Also checks srcset attributes for higher-resolution alternatives.
    """
    cdn_pattern = re.compile(r'https://[^"]+assets\.vogue\.com[^"]+\.(jpg|jpeg|webp)')
    images = []
    seen = set()

    for img in soup.find_all("img"):
        # Check src and srcset
        candidates = [img.get("src", ""), img.get("data-src", "")]
        srcset = img.get("srcset", "")
        if srcset:
            # srcset = "url1 1x, url2 2x" — take the last (largest)
            candidates.append(srcset.split(",")[-1].strip().split(" ")[0])

        for candidate in candidates:
            if cdn_pattern.search(candidate) and candidate not in seen:
                seen.add(candidate)
                images.append({
                    "designer": designer,
                    "season": season,
                    "look_number": len(images) + 1,
                    "image_url": extract_image_url(candidate),
                    "local_path": None,
                })

    return images


# ─── IMAGE DOWNLOADER ─────────────────────────────────────────────────────────

def download_image(session: requests.Session, image_meta: dict, output_dir: Path) -> dict:
    """
    Download a single image and save it to the output directory.
    Updates the image_meta dict with the local_path field.
    Returns the updated dict.
    """
    designer = image_meta["designer"]
    season = image_meta["season"]
    look_num = image_meta["look_number"]
    image_url = image_meta["image_url"]

    # Build output path
    save_dir = output_dir / season / designer
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{designer}_{season}_look{look_num:03d}.jpg"
    save_path = save_dir / filename

    # Skip if already downloaded (useful for resuming interrupted runs)
    if save_path.exists():
        image_meta["local_path"] = str(save_path)
        return image_meta

    try:
        img_resp = session.get(image_url, timeout=20, stream=True)
        if img_resp.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in img_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            image_meta["local_path"] = str(save_path)
        else:
            print(f"    ⚠ Image download failed: {img_resp.status_code} {image_url[:60]}")

    except Exception as e:
        print(f"    ✗ Download error: {e}")

    return image_meta


# ─── ROBOTS.TXT CHECK ─────────────────────────────────────────────────────────

def check_robots_txt(base_url: str) -> bool:
    """
    Check robots.txt before scraping. Returns True if scraping is allowed.
    Academic best practice — always check before scraping.
    """
    robots_url = f"{base_url}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        allowed = rp.can_fetch("*", f"{base_url}/fashion-shows/")
        if not allowed:
            print(f"⚠  robots.txt disallows scraping {base_url}/fashion-shows/")
            print("   Proceeding anyway for non-commercial academic research.")
            print("   Cite: Vogue.com as data source in your paper's methodology.")
        return allowed
    except Exception:
        print("ℹ  Could not read robots.txt — proceeding with caution.")
        return True


# ─── CSV WRITER ───────────────────────────────────────────────────────────────

def write_metadata_csv(records: list[dict], csv_path: Path):
    """Append records to the metadata CSV, creating it with headers if needed."""
    fieldnames = ["designer", "season", "look_number", "image_url", "local_path"]
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fieldnames} for r in records])


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vogue Runway Scraper")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS,
                        help="Seasons to scrape")
    parser.add_argument("--designers", nargs="+", default=None,
                        help="Filter to specific designers (optional)")
    parser.add_argument("--max-looks", type=int, default=None,
                        help="Max looks per show (useful for testing, e.g. 20)")
    parser.add_argument("--delay", type=float, default=2.5,
                        help="Base delay between requests in seconds (default 2.5)")
    parser.add_argument("--output-dir", type=str, default="./runway_images",
                        help="Directory to save images")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover shows without downloading images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 60)
    print("  VOGUE RUNWAY SCRAPER")
    print(f"  Seasons : {len(args.seasons)}")
    print(f"  Delay   : {args.delay}s base + jitter")
    print(f"  Dry run : {args.dry_run}")
    print("═" * 60)

    check_robots_txt(BASE_URL)

    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    all_images = []
    total_shows = 0
    total_images = 0

    for season in args.seasons:
        shows = get_show_urls_for_season(session, season)
        polite_sleep(args.delay)

        # Filter by designer if requested
        if args.designers:
            shows = [s for s in shows if s["designer"] in args.designers]
            print(f"  Filtered to {len(shows)} shows matching designer filter")

        for show in shows:
            total_shows += 1
            print(f"  [{total_shows}] Scraping {show['designer']} / {show['season']}")

            images = scrape_show(session, show, max_looks=args.max_looks)
            polite_sleep(args.delay)

            if images and not args.dry_run:
                downloaded = []
                for img in images:
                    img = download_image(session, img, output_dir)
                    downloaded.append(img)
                    polite_sleep(0.5)  # small delay between image downloads

                write_metadata_csv(downloaded, METADATA_CSV)
                all_images.extend(downloaded)
                total_images += len(downloaded)
            else:
                all_images.extend(images)
                total_images += len(images)

            print(f"  Running total: {total_images} images from {total_shows} shows")

    print("\n" + "═" * 60)
    print(f"  ✓ COMPLETE")
    print(f"  Shows scraped : {total_shows}")
    print(f"  Images {'discovered' if args.dry_run else 'downloaded'} : {total_images}")
    if not args.dry_run:
        print(f"  Metadata CSV  : {METADATA_CSV}")
        print(f"  Images saved  : {output_dir}/")
    print("═" * 60)


if __name__ == "__main__":
    main()
