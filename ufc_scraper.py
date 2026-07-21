import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page

BASE_URL = "http://ufcstats.com"

# Signals that we got a bot-check / interstitial page instead of real content.
BLOCK_SIGNALS = ("requires JavaScript", "Just a moment", "Checking your browser", "Attention Required")


def fetch(page: Page, url: str, delay: float = 1.0, retries: int = 3) -> str:
    last_err = None
    for attempt in range(retries):
        time.sleep(delay)
        try:
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)
            text = page.content()
            if any(signal in text for signal in BLOCK_SIGNALS):
                page.wait_for_timeout(5000)
                text = page.content()
            if any(signal in text for signal in BLOCK_SIGNALS):
                raise RuntimeError(
                    f"Got a bot-check/interstitial page for {url} instead of real content "
                    f"(matched: {[s for s in BLOCK_SIGNALS if s in text]})."
                )
            return text
        except Exception as e:
            last_err = e
            time.sleep(delay * (attempt + 2))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err}")


def clean_text(node: Any) -> str:
    if node is None:
        return ""
    text = node.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def extract_parts(cell: Any) -> list[str]:
    parts = [clean_text(p) for p in cell.find_all("p")]
    parts = [p for p in parts if p]
    if not parts:
        text = clean_text(cell)
        if text:
            parts.append(text)
    return parts


def parse_event_list(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    events: list[dict[str, str]] = []
    table = soup.find("table")
    if not table:
        return events

    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        link = cells[0].find("a", href=True)
        if not link:
            continue
        href = link.get("href", "")
        if not href.startswith("/event-details/") and "/event-details/" not in href:
            continue
        date_match = re.search(r"([A-Z][a-z]+\s+\d{1,2},\s*\d{4})", clean_text(cells[0]))
        events.append(
            {
                "name": clean_text(link),
                "url": urljoin(BASE_URL, href),
                "date": date_match.group(1) if date_match else "",
                "location": clean_text(cells[1]),
            }
        )
    return events


def filter_completed_events(events: list[dict[str, str]]) -> list[dict[str, str]]:
    today = datetime.now()
    completed = []
    for e in events:
        if not e["date"]:
            continue
        try:
            event_date = datetime.strptime(e["date"], "%B %d, %Y")
        except ValueError:
            continue
        if event_date <= today:
            completed.append(e)
    return completed


def parse_event_fights(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    fights: list[dict[str, Any]] = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header_text = " ".join(clean_text(cell) for cell in rows[0].find_all(["th", "td"]))
        if "W/L" not in header_text or "Fighter" not in header_text or "Weight class" not in header_text:
            continue

        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 10:
                continue

            fight_url = row.get("data-link", "").strip()

            fighter_links = cells[1].find_all("a", href=True)
            fighters = [clean_text(link) for link in fighter_links]
            if not fighters:
                fighters = [clean_text(cells[1])]

            fights.append(
                {
                    "fight_url": fight_url,
                    "fighter_names": fighters,
                    "result_raw": clean_text(cells[0]),
                    "method": clean_text(cells[7]),
                    "round": clean_text(cells[8]),
                    "time": clean_text(cells[9]),
                    "weight_class": clean_text(cells[6]),
                }
            )
        if fights:
            break
    return fights


def parse_fight_details(
    html: str,
    event_name: str,
    event_url: str,
    event_date: str,  
    fight_url: str,
    fight_summary: dict[str, Any],
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    fighter_names = []
    fighter_ids = []
    winner = ""
    for person in soup.select("div.b-fight-details__person"):
        name_tag = person.select_one("a")
        status_tag = person.select_one("i.b-fight-details__person-status")
        name = clean_text(name_tag) if name_tag else ""
        fighter_id = name_tag["href"].rstrip("/").split("/")[-1] if name_tag and name_tag.get("href") else ""
        if name:
            fighter_names.append(name)
            fighter_ids.append(fighter_id)
        status = clean_text(status_tag) if status_tag else ""
        if status == "W":
            winner = name

    if not fighter_names:
        fighter_names = fight_summary.get("fighter_names", [])

    full_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    detail_match = re.search(r"Method:\s*(.*?)\s*Round:\s*(\d+)\s*Time:\s*([0-9:]+)", full_text)
    method = detail_match.group(1).strip() if detail_match else fight_summary.get("method", "")
    round_no = detail_match.group(2).strip() if detail_match else fight_summary.get("round", "")
    time_value = detail_match.group(3).strip() if detail_match else fight_summary.get("time", "")

    totals_table = None
    per_round_breakdown = ""
    for table in soup.find_all("table"):
        header_cells = [clean_text(cell) for cell in table.find_all(["th", "td"])]
        header_text = " ".join(header_cells[:10])
        if "Fighter" in header_text and "Total str." in header_text and "Ctrl" in header_text:
            totals_table = table
            break

    totals_payload: dict[str, Any] = {}
    if totals_table:
        rows = totals_table.find_all("tr")
        if rows:
            headers = [clean_text(cell) for cell in rows[0].find_all(["th", "td"])]
            for row in rows[1:]:
                cells = row.find_all("td")
                if not cells:
                    continue
                fighter_parts = extract_parts(cells[0])
                if len(fighter_parts) < 2:
                    fighter_parts = fighter_parts + fighter_parts[:1] * (2 - len(fighter_parts))
                row_payload: dict[str, Any] = {}
                for idx, header in enumerate(headers):
                    if idx >= len(cells):
                        continue
                    values = extract_parts(cells[idx])
                    if len(values) < 2 and idx == 0:
                        values = fighter_parts
                    if len(values) < 2:
                        values = values + values[:1] * (2 - len(values))
                    row_payload[header] = values[:2]
                totals_payload["rows"] = totals_payload.get("rows", []) + [row_payload]
                if not totals_payload.get("fighters"):
                    totals_payload["fighters"] = fighter_parts[:2]

    for heading in soup.find_all(["h4", "h3"]):
        if clean_text(heading).lower() == "per round":
            section = heading.parent
            per_round_breakdown = clean_text(section)
            break

    return {
        "event_name": event_name,
        "event_url": event_url,
        "event_date": event_date,  
        "fight_url": fight_url,
        "fighter_names": fighter_names,
        "fighter_ids": fighter_ids,
        "winner": winner,
        "result_raw": fight_summary.get("result_raw", ""),
        "method": method,
        "round": round_no,
        "time": time_value,
        "weight_class": fight_summary.get("weight_class", ""),
        "totals": totals_payload,
        "per_round_breakdown": per_round_breakdown,
        "raw_text_excerpt": full_text[:600],
    }


def load_existing_records(output_dir: str) -> list[dict[str, Any]]:
    json_path = os.path.join(output_dir, "ufc_fights.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


def write_outputs(records: list[dict[str, Any]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "ufc_fights.json")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape UFC event and fight statistics into JSON/CSV")
    parser.add_argument("--events-url", default=f"{BASE_URL}/statistics/events/completed?page=all")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of NEW events scraped this run (omit for no cap)")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds to wait between requests")
    args = parser.parse_args()

    existing_records = load_existing_records(args.output_dir)
    existing_event_urls = {r["event_url"] for r in existing_records}
    print(f"Loaded {len(existing_records)} existing fight records ({len(existing_event_urls)} events already scraped)")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )

        html = fetch(page, args.events_url, delay=args.delay)
        all_events = filter_completed_events(parse_event_list(html))
        new_events = [e for e in all_events if e["url"] not in existing_event_urls]
        new_events = new_events[: args.limit]
        print(f"Found {len(all_events)} completed events total, {len(new_events)} new to scrape")

        records: list[dict[str, Any]] = []
        for i, event in enumerate(new_events, 1):
            print(f"[{i}/{len(new_events)}] {event['name']}")
            try:
                event_html = fetch(page, event["url"], delay=args.delay)
            except RuntimeError as e:
                print(f"  skipping event, failed to load: {e}")
                continue

            fights = parse_event_fights(event_html)
            if not fights:
                print(f"  warning: no fights parsed for {event['name']} -- check page structure")
            for fight_summary in fights:
                if not fight_summary.get("fight_url"):
                    print("  warning: fight row had no fight_url, skipping one fight")
                    continue
                try:
                    fight_html = fetch(page, fight_summary["fight_url"], delay=args.delay)
                except RuntimeError as e:
                    print(f"  skipping fight {fight_summary['fight_url']}: {e}")
                    continue
                record = parse_fight_details(
                    fight_html,
                    event_name=event["name"],
                    event_url=event["url"],
                    event_date=event["date"],  
                    fight_url=fight_summary["fight_url"],
                    fight_summary=fight_summary,
                )
                records.append(record)

        browser.close()

    all_records = existing_records + records
    write_outputs(all_records, args.output_dir)
    print(f"Scraped {len(records)} new fights ({len(all_records)} total) into {args.output_dir}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")