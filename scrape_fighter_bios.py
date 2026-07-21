

import argparse
import json
import os
import re
import time

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page

BASE_URL = "http://ufcstats.com"
BLOCK_SIGNALS = ("requires JavaScript", "Just a moment", "Checking your browser", "Attention Required")


def fetch(page: Page, url: str, delay: float = 2.0, retries: int = 3) -> str:
    last_err = None
    for attempt in range(retries):
        time.sleep(delay)
        try:
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)
            text = page.content()
            if any(s in text for s in BLOCK_SIGNALS):
                page.wait_for_timeout(5000)
                text = page.content()
            if any(s in text for s in BLOCK_SIGNALS):
                raise RuntimeError(f"Blocked page for {url}")
            return text
        except Exception as e:
            last_err = e
            time.sleep(delay * (attempt + 2))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def clean_text(node) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def parse_fighter_bio(html: str, fighter_id: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    name_tag = soup.select_one("span.b-content__title-highlight")
    name = clean_text(name_tag)

    record_tag = soup.select_one("span.b-content__title-record")
    record = clean_text(record_tag).replace("Record:", "").strip()

    bio: dict = {"fighter_id": fighter_id, "name": name, "record": record}

    for item in soup.select("li.b-list__box-list-item"):
        label_tag = item.find("i")
        if not label_tag:
            continue
        label = clean_text(label_tag).rstrip(":")
        full_text = clean_text(item)
        label_text = clean_text(label_tag)
        value = full_text[len(label_text):].strip() if full_text.startswith(label_text) else full_text
        if label and label not in bio:
            bio[label] = value

    return bio


def load_existing(output_path: str) -> dict:
    if not os.path.exists(output_path):
        return {}
    with open(output_path, encoding="utf-8") as fh:
        rows = json.load(fh)
    return {r["fighter_id"]: r for r in rows if r.get("fighter_id")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape UFC fighter bio pages for unique fighters in your fight data")
    parser.add_argument("--fights-json", default="data/raw/ufc_fights.json")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None, help="cap number of NEW fighters scraped this run")
    args = parser.parse_args()

    with open(args.fights_json, encoding="utf-8") as fh:
        fights = json.load(fh)

    fighter_ids = set()
    for f in fights:
        for fid in f.get("fighter_ids", []):
            if fid:
                fighter_ids.add(fid)
    print(f"{len(fighter_ids)} unique fighter_ids found in {args.fights_json}")

    out_path = os.path.join(args.output_dir, "ufc_fighters.json")
    existing = load_existing(out_path)
    new_ids = sorted(fid for fid in fighter_ids if fid not in existing)
    if args.limit:
        new_ids = new_ids[: args.limit]
    print(f"{len(existing)} already scraped, {len(new_ids)} new to fetch this run")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )

        for i, fid in enumerate(new_ids, 1):
            url = f"{BASE_URL}/fighter-details/{fid}"
            print(f"[{i}/{len(new_ids)}] {url}")
            try:
                html = fetch(page, url, delay=args.delay)
            except RuntimeError as e:
                print(f"  skipping {fid}: {e}")
                continue
            bio = parse_fighter_bio(html, fid)
            if not bio.get("name"):
                print(f"  warning: no name parsed for {fid} -- check selectors against live page")
            existing[fid] = bio

        browser.close()

    os.makedirs(args.output_dir, exist_ok=True)
    records = list(existing.values())

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    print(f"Wrote {len(records)} fighter bio records to {out_path}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)")