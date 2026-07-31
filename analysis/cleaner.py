import json
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_FIGHTERS_PATH = RAW_DIR / "ufc_fighters.json"
RAW_FIGHTS_PATH = RAW_DIR / "ufc_fights.json"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True)

with open(RAW_FIGHTERS_PATH, encoding="utf-8") as f:
    raw_fighters = json.load(f)

with open(RAW_FIGHTS_PATH, encoding="utf-8") as f:
    raw_fights = json.load(f)

print(f"Loaded {len(raw_fighters):,} fighter records")
print(f"Loaded {len(raw_fights):,} fight records")

def parse_height_to_inches(val):
    if not val or val == "--":
        return np.nan
    m = re.match(r"(\d+)'\s*(\d+)\"", val)
    if not m:
        return np.nan
    feet, inches = int(m.group(1)), int(m.group(2))
    return float(feet * 12 + inches)


def parse_weight_lbs(val):
    if not val or val == "--":
        return np.nan
    m = re.match(r"(\d+)\s*lbs\.?", val)
    return float(m.group(1)) if m else np.nan


def parse_reach_in(val):
    if not val or val == "--":
        return np.nan
    m = re.match(r"(\d+)\"", val)
    return float(m.group(1)) if m else np.nan


def parse_pct(val):
    if not val or val in ("--", "---"):
        return np.nan
    m = re.match(r"(\d+(?:\.\d+)?)%", val)
    return float(m.group(1)) if m else np.nan


def parse_float(val):
    if not val or val in ("--", "---"):
        return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan


def parse_dob(val):
    if not val or val == "--":
        return pd.NaT
    try:
        return datetime.strptime(val, "%b %d, %Y")
    except ValueError:
        return pd.NaT


RECORD_RE = re.compile(r"^(\d+)-(\d+)-(\d+)(?:\s*\((\d+)\s*NC\))?$")


def parse_record(val):
    if not val:
        return (np.nan, np.nan, np.nan, np.nan)
    m = RECORD_RE.match(val.strip())
    if not m:
        return (np.nan, np.nan, np.nan, np.nan)
    wins, losses, draws, nc = m.groups()
    return int(wins), int(losses), int(draws), int(nc) if nc else 0

fighter_rows = []
for r in raw_fighters:
    wins, losses, draws, nc = parse_record(r.get("record"))
    fighter_rows.append(
        {
            "fighter_id": r["fighter_id"],
            "name": r["name"],
            "height_in": parse_height_to_inches(r.get("Height")),
            "weight_lbs": parse_weight_lbs(r.get("Weight")),
            "reach_in": parse_reach_in(r.get("Reach")),
            "stance": r.get("STANCE") or np.nan,
            "dob": parse_dob(r.get("DOB")),
            # ---- QUARANTINED (career-to-date, scrape-time snapshot):----
            "LEAK_record_raw": r.get("record") or np.nan,
            "LEAK_record_wins": wins,
            "LEAK_record_losses": losses,
            "LEAK_record_draws": draws,
            "LEAK_record_nc": nc,
            "LEAK_slpm": parse_float(r.get("SLpM")),
            "LEAK_str_acc_pct": parse_pct(r.get("Str. Acc.")),
            "LEAK_sapm": parse_float(r.get("SApM")),
            "LEAK_str_def_pct": parse_pct(r.get("Str. Def")),
            "LEAK_td_avg": parse_float(r.get("TD Avg.")),
            "LEAK_td_acc_pct": parse_pct(r.get("TD Acc.")),
            "LEAK_td_def_pct": parse_pct(r.get("TD Def.")),
            "LEAK_sub_avg": parse_float(r.get("Sub. Avg.")),
        }
    )

fighters_clean = pd.DataFrame(fighter_rows)
LEAKAGE_COLUMNS = [c for c in fighters_clean.columns if c.startswith("LEAK_")]

print(f"fighters_clean: {fighters_clean.shape[0]:,} rows x {fighters_clean.shape[1]} cols")
print(f"Quarantined leakage columns ({len(LEAKAGE_COLUMNS)}): {LEAKAGE_COLUMNS}")
fighters_clean.head()

# --- Structural integrity checks: fighters_clean ---
assert fighters_clean["fighter_id"].is_unique, "duplicate fighter_id found"
assert fighters_clean["fighter_id"].notna().all(), "null fighter_id found"
assert fighters_clean["name"].notna().all(), "null name found"

# plausibility bounds are intentionally generous (early UFC had sumo-sized
# heavyweights, e.g. Emmanuel Yarborough at 770 lbs) -- this flags for review,
# it does not drop rows
implausible = {
    "height_in": fighters_clean[fighters_clean["height_in"].notna() & ~fighters_clean["height_in"].between(48, 90)],
    "weight_lbs": fighters_clean[fighters_clean["weight_lbs"].notna() & ~fighters_clean["weight_lbs"].between(95, 800)],
    "reach_in": fighters_clean[fighters_clean["reach_in"].notna() & ~fighters_clean["reach_in"].between(48, 90)],
}
for field, df in implausible.items():
    print(f"{field}: {len(df)} implausible value(s)")

print("\nMissing-value counts:")
print(fighters_clean.drop(columns=LEAKAGE_COLUMNS).isna().sum())

print(f"\nDuplicate names (distinct fighter_ids sharing a name): "
      f"{(fighters_clean['name'].value_counts() > 1).sum()}")


# Reach imputation by linear regression (reach ~ height)

fit_mask = fighters_clean["height_in"].notna() & fighters_clean["reach_in"].notna()
slope, intercept = np.polyfit(fighters_clean.loc[fit_mask, "height_in"], fighters_clean.loc[fit_mask, "reach_in"], 1)

predicted = slope * fighters_clean.loc[fit_mask, "height_in"] + intercept
residuals = fighters_clean.loc[fit_mask, "reach_in"] - predicted
r_squared = 1 - (residuals**2).sum() / ((fighters_clean.loc[fit_mask, "reach_in"] - fighters_clean.loc[fit_mask, "reach_in"].mean())**2).sum()

print(f"reach_in = {slope:.3f} * height_in + {intercept:.3f}")
print(f"R^2 = {r_squared:.3f} (fit on {fit_mask.sum():,} fighters with both height and reach)")
print(f"residual std = {residuals.std():.2f} inches")

needs_impute = fighters_clean["reach_in"].isna() & fighters_clean["height_in"].notna()
fighters_clean["reach_in_imputed"] = fighters_clean["reach_in"]
fighters_clean.loc[needs_impute, "reach_in_imputed"] = (
    slope * fighters_clean.loc[needs_impute, "height_in"] + intercept
)
fighters_clean["reach_imputed"] = needs_impute

print(f"\nImputed: {needs_impute.sum():,} fighters")
print(f"Still missing (no height to impute from either): {fighters_clean['reach_in_imputed'].isna().sum():,}")


def parse_x_of_y(val):
    if not val:
        return (np.nan, np.nan)
    m = re.match(r"(\d+) of (\d+)", val)
    if not m:
        return (np.nan, np.nan)
    return int(m.group(1)), int(m.group(2))


def parse_clock_to_seconds(val):
    if not val or val == "--":
        return np.nan
    m = re.match(r"(\d+):(\d\d)", val)
    if not m:
        return np.nan
    return int(m.group(1)) * 60 + int(m.group(2))


def parse_event_date(val):
    if not val:
        return pd.NaT
    try:
        return datetime.strptime(val, "%B %d, %Y")
    except ValueError:
        return pd.NaT


def parse_int(val):
    if val is None or val == "":
        return np.nan
    try:
        return int(val)
    except ValueError:
        return np.nan


def stats_for_fighter(fight, idx):
    """Pull one fighter's stat row (idx 0 or 1) out of `totals`; NaNs if absent."""
    rows = fight["totals"].get("rows") or []
    if not rows:
        return {
            "kd": np.nan,
            "sig_str_landed": np.nan, "sig_str_attempted": np.nan, "sig_str_pct": np.nan,
            "total_str_landed": np.nan, "total_str_attempted": np.nan,
            "td_landed": np.nan, "td_attempted": np.nan, "td_pct": np.nan,
            "sub_att": np.nan, "rev": np.nan, "ctrl_sec": np.nan,
        }
    row = rows[0]
    sig_l, sig_a = parse_x_of_y(row["Sig. str."][idx])
    tot_l, tot_a = parse_x_of_y(row["Total str."][idx])
    td_l, td_a = parse_x_of_y(row["Td"][idx])
    return {
        "kd": parse_int(row["KD"][idx]),
        "sig_str_landed": sig_l, "sig_str_attempted": sig_a,
        "sig_str_pct": parse_pct(row["Sig. str. %"][idx]),
        "total_str_landed": tot_l, "total_str_attempted": tot_a,
        "td_landed": td_l, "td_attempted": td_a,
        "td_pct": parse_pct(row["Td %"][idx]),
        "sub_att": parse_int(row["Sub. att"][idx]),
        "rev": parse_int(row["Rev."][idx]),
        "ctrl_sec": parse_clock_to_seconds(row["Ctrl"][idx]),
    }


TIME_FORMAT_RE = re.compile(r"Time format:\s*(.+?)\s*Referee:")
SCHEDULED_ROUNDS_RE = re.compile(r"^(\d+)\s*Rnd")


def parse_time_format(raw_text_excerpt):
    m = TIME_FORMAT_RE.search(raw_text_excerpt or "")
    if not m:
        return None, np.nan
    fmt = m.group(1).strip()
    rm = SCHEDULED_ROUNDS_RE.match(fmt)
    scheduled_rounds = int(rm.group(1)) if rm else np.nan
    return fmt, scheduled_rounds


fight_rows = []
for fight in raw_fights:
    names = fight["fighter_names"]
    ids = fight["fighter_ids"]
    winner = fight.get("winner") or None
    result_raw = fight["result_raw"]

    if winner:
        outcome = "win"
        winner_id = ids[names.index(winner)]
    elif result_raw.startswith("draw"):
        outcome, winner_id = "draw", None
    elif result_raw.startswith("nc"):
        outcome, winner_id = "no_contest", None
    else:
        outcome, winner_id = "unknown", None

    time_format_raw, scheduled_rounds = parse_time_format(fight.get("raw_text_excerpt"))

    fight_rows.append(
        {
            "fight_id": fight["fight_url"].rstrip("/").rsplit("/", 1)[-1],
            "fight_url": fight["fight_url"],
            "event_name": fight["event_name"],
            "event_url": fight["event_url"],
            "event_date": parse_event_date(fight["event_date"]),
            "weight_class": fight["weight_class"],
            "fighter_1_id": ids[0], "fighter_1_name": names[0],
            "fighter_2_id": ids[1], "fighter_2_name": names[1],
            "winner_id": winner_id, "winner_name": winner,
            "outcome": outcome,
            "method": fight["method"],
            "round": parse_int(fight["round"]),
            "time_sec": parse_clock_to_seconds(fight["time"]),
            "time_format_raw": time_format_raw,
            "scheduled_rounds": scheduled_rounds,
            "has_stats": bool(fight["totals"].get("rows")),
        }
    )

fights_clean = pd.DataFrame(fight_rows)
print(f"fights_clean: {fights_clean.shape[0]:,} rows x {fights_clean.shape[1]} cols")
fights_clean.head()

fights_clean = fights_clean.sort_values("event_date", kind="stable").reset_index(drop=True)

# --- Structural integrity checks: fights_clean ---
assert fights_clean["fight_id"].is_unique, "duplicate fight_id found"
assert fights_clean["event_date"].notna().all(), "unparsed event_date found"
assert fights_clean["round"].between(1, 5).all(), "round outside 1-5 found"
assert fights_clean["outcome"].isin(["win", "draw", "no_contest"]).all(), "unrecognized outcome found"

# every fighter_id referenced in fights must exist in the fighters table
known_ids = set(fighters_clean["fighter_id"])
referenced_ids = set(fights_clean["fighter_1_id"]) | set(fights_clean["fighter_2_id"])
orphaned = referenced_ids - known_ids
print(f"Fighter IDs referenced in fights but missing from fighters table: {len(orphaned)}")

# winner_id, when present, must be one of the two fighters in that fight
bad_winner = fights_clean[
    fights_clean["winner_id"].notna()
    & ~fights_clean.apply(lambda r: r["winner_id"] in (r["fighter_1_id"], r["fighter_2_id"]), axis=1)
]
print(f"Fights with winner_id not among the two fighters: {len(bad_winner)}")

print(f"\nFights without compiled stats (has_stats=False): {(~fights_clean['has_stats']).sum()}")
print("\nOutcome breakdown:")
print(fights_clean["outcome"].value_counts())

# scheduled_rounds: NaN only for non round-based formats (e.g. old 'No Time
# Limit' rule) or the rare unparseable excerpt -- not an error, just informational
print(f"\nscheduled_rounds missing: {fights_clean['scheduled_rounds'].isna().sum()} "
      f"(time_format_raw for those: {fights_clean.loc[fights_clean['scheduled_rounds'].isna(), 'time_format_raw'].value_counts().to_dict()})")
print("\nscheduled_rounds distribution:")
print(fights_clean["scheduled_rounds"].value_counts(dropna=False).sort_index())

# a fight can't end in a round beyond its scheduled distance, except for
# formats with an overtime round tacked on ("+ OT" / "+ 2OT" in time_format_raw)
over_scheduled = fights_clean[
    fights_clean["scheduled_rounds"].notna()
    & (fights_clean["round"] > fights_clean["scheduled_rounds"])
    & ~fights_clean["time_format_raw"].str.contains("OT", na=False)
]
print(f"\nFights ending beyond scheduled_rounds with no OT format (should be 0): {len(over_scheduled)}")

long_rows = []
for fight, meta in zip(raw_fights, fight_rows):
    ids = fight["fighter_ids"]
    names = fight["fighter_names"]
    for idx in (0, 1):
        other = 1 - idx
        fighter_id, opp_id = ids[idx], ids[other]
        fighter_name, opp_name = names[idx], names[other]

        if meta["outcome"] == "win":
            result = "win" if fighter_id == meta["winner_id"] else "loss"
        else:
            result = meta["outcome"]  # draw / no_contest / unknown

        own_stats = stats_for_fighter(fight, idx)
        opp_stats = {f"opp_{k}": v for k, v in stats_for_fighter(fight, other).items()}

        long_rows.append(
            {
                "fight_id": meta["fight_id"],
                "event_name": meta["event_name"],
                "event_date": meta["event_date"],
                "weight_class": meta["weight_class"],
                "method": meta["method"],
                "round": meta["round"],
                "time_sec": meta["time_sec"],
                "scheduled_rounds": meta["scheduled_rounds"],
                "time_format_raw": meta["time_format_raw"],
                "has_stats": meta["has_stats"],
                "fighter_id": fighter_id,
                "fighter_name": fighter_name,
                "opponent_id": opp_id,
                "opponent_name": opp_name,
                "result": result,
                "is_win": result == "win",
                **own_stats,
                **opp_stats,
            }
        )

fights_long_clean = pd.DataFrame(long_rows)
print(f"fights_long_clean: {fights_long_clean.shape[0]:,} rows x {fights_long_clean.shape[1]} cols")
fights_long_clean.head()

fights_long_clean = fights_long_clean.sort_values(
    ["fighter_id", "event_date"], kind="stable"
).reset_index(drop=True)



fighters_clean.to_parquet(OUT_DIR / "fighters_clean.parquet", index=False)
fights_long_clean.to_parquet(OUT_DIR / "fights_long_clean.parquet", index=False)
print("Saved cleaned tables to data/processed/")