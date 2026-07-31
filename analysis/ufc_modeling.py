import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.stats import randint, uniform, loguniform
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import joblib

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_FIGHTERS_PATH = RAW_DIR / "ufc_fighters.json"
RAW_FIGHTS_PATH = RAW_DIR / "ufc_fights.json"
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True)
fighters_clean = pd.read_parquet(OUT_DIR / "fighters_clean.parquet")
fights_long_clean = pd.read_parquet(OUT_DIR / "fights_long_clean.parquet")

RECENT_WINDOW = 5
ELO_K = 40
ELO_BASE = 1500.0

def add_momentum(df):
    g = df.groupby("fighter_id", group_keys=False)
    prior_win = g["is_win"].shift(1)

    df["pf_total_fights"] = g.cumcount()                       # prior fights, exclusive
    df["pf_wins"] = prior_win.groupby(df["fighter_id"]).cumsum()
    df["pf_win_pct"] = df["pf_wins"] / df["pf_total_fights"].replace(0, np.nan)
    df["pf_win_pct_last5"] = (prior_win.groupby(df["fighter_id"])
        .rolling(RECENT_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True))
    df["pf_win_streak"] = _signed_streak(df, df["is_win"])

    prev_date = g["event_date"].shift(1)
    df["pf_days_since_last"] = (df["event_date"] - prev_date).dt.days
    df["pf_fights_last_365d"] = _fights_in_window(df, 365)     # activity / chronic inactivity
    df["pf_fights_last_730d"] = _fights_in_window(df, 730)
    return df

def _signed_streak(df, wins):
    """Signed current streak coming INTO each fight (+k win streak, -k loss streak)."""
    out = np.zeros(len(df))
    for _, idx in df.groupby("fighter_id", sort=False).groups.items():
        streak = 0
        for i in list(idx):
            out[i] = streak                       # value BEFORE this fight
            w = wins.iloc[i]                  # this fight's result
            if pd.isna(w):
                continue
            streak = (streak + 1 if streak > 0 else 1) if w == 1 else \
                     (streak - 1 if streak < 0 else -1)
    return out


def _fights_in_window(df, days):
    """Count of a fighter's PRIOR fights within `days` before the current one."""
    out = np.zeros(len(df))
    for _, idx in df.groupby("fighter_id", sort=False).groups.items():
        idx = list(idx)
        dates = df.loc[idx, "event_date"].values
        for pos in range(len(idx)):
            cutoff = dates[pos] - np.timedelta64(days, "D")
            out[idx[pos]] = np.sum(dates[:pos] >= cutoff)   # prior fights only
    return out

PERF_COLS = ["sig_str_landed", "sig_str_attempted", "total_str_landed",
             "total_str_attempted", "td_landed", "td_attempted",
             "kd", "sub_att", "rev", "ctrl_sec"]
OPP_PERF_COLS = ["opp_" + c for c in PERF_COLS]


def add_rolling_perf(df):
    gid = df["fighter_id"]
    for col in PERF_COLS + OPP_PERF_COLS:
        shifted = df.groupby("fighter_id", sort=False)[col].shift(1)
        df[f"car_{col}"] = shifted.groupby(gid).expanding().mean().reset_index(level=0, drop=True)
        df[f"r5_{col}"] = (shifted.groupby(gid).rolling(RECENT_WINDOW, min_periods=1)
            .mean().reset_index(level=0, drop=True))

    # Career accuracy from SUMMED volume (never the mean of per-fight %) —
    # keeps a 1-for-1 night from distorting the rate the way a per-fight-% mean would.
    df = _rate_career(df, "sig_str", "sig_str_landed", "sig_str_attempted")
    df = _rate_career(df, "td", "td_landed", "td_attempted")
    df = _rate_career(df, "opp_sig_str", "opp_sig_str_landed", "opp_sig_str_attempted")
    df["car_str_def"] = 1 - df["car_opp_sig_str_acc"]
    df = _rate_career(df, "opp_td", "opp_td_landed", "opp_td_attempted")
    df["car_td_def"] = 1 - df["car_opp_td_acc"]

    # Recent-window accuracy, same Σlanded/Σattempted logic — needed for honest trends.
    df = _rate_windowed(df, "sig_str", "sig_str_landed", "sig_str_attempted")
    df = _rate_windowed(df, "td", "td_landed", "td_attempted")
    return df


def _rate_career(df, name, landed_col, att_col):
    gid = df["fighter_id"]
    landed = df.groupby("fighter_id", sort=False)[landed_col].shift(1)
    att = df.groupby("fighter_id", sort=False)[att_col].shift(1)
    cum_landed = landed.groupby(gid).cumsum()
    cum_att = att.groupby(gid).cumsum()
    df[f"car_{name}_acc"] = cum_landed / cum_att.replace(0, np.nan)
    # volume as its own signal (Gaethje/Merab: low accuracy at high volume is a gameplan,
    # not a weakness — the tree can separate the two only if it sees volume separately)
    df[f"car_{name}_att_pervol"] = cum_att / df["pf_total_fights"].replace(0, np.nan)
    return df


def _rate_windowed(df, name, landed_col, att_col):
    gid = df["fighter_id"]
    landed = df.groupby("fighter_id", sort=False)[landed_col].shift(1)
    att = df.groupby("fighter_id", sort=False)[att_col].shift(1)
    roll_landed = landed.groupby(gid).rolling(RECENT_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)
    roll_att = att.groupby(gid).rolling(RECENT_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)
    df[f"r5_{name}_acc"] = roll_landed / roll_att.replace(0, np.nan)
    return df

def add_trends(df):
    df["trend_sig_str_acc"] = df["car_sig_str_acc"] - df["r5_sig_str_acc"]
    df["trend_td_acc"] = df["car_td_acc"] - df["r5_td_acc"]
    df["trend_td_attempts"] = df["car_td_attempted"] - df["r5_td_attempted"]   # fading volume wrestling
    df["trend_sig_output"] = df["car_sig_str_landed"] - df["r5_sig_str_landed"]
    df["trend_damage_absorbed"] = df["r5_opp_sig_str_landed"] - df["car_opp_sig_str_landed"]
    df["trend_winpct"] = df["pf_win_pct"] - df["pf_win_pct_last5"]
    return df

def add_loss_profile(df):
    df = df.sort_values(["fighter_id", "event_date", "fight_id"]).reset_index(drop=True)
    method_u = df["method"].str.upper().fillna("")
    is_loss = (df["result"] == "loss")
    df["_ko_loss"] = (is_loss & method_u.str.contains("KO|TKO", na=False)).astype(int)
    df["_sub_loss"] = (is_loss & method_u.str.contains("SUB", na=False)).astype(int)
    df["_dec_loss"] = (is_loss & method_u.str.contains("DEC", na=False)).astype(int)

    gid = df["fighter_id"]
    for kind in ["ko", "sub", "dec"]:
        shifted = df.groupby("fighter_id", sort=False)[f"_{kind}_loss"].shift(1)
        df[f"pf_{kind}_losses"] = shifted.groupby(gid).cumsum()

    # Recency of most recent KO / finish loss: NaN if it has never happened
    # (XGBoost routes NaN natively), plus explicit "ever" booleans.
    df["pf_fights_since_ko_loss"] = _fights_since(df, df["_ko_loss"])
    df["pf_fights_since_finish_loss"] = _fights_since(df, (df["_ko_loss"] | df["_sub_loss"]).astype(int))
    df["pf_ever_ko_loss"] = (df["pf_ko_losses"].fillna(0) > 0).astype(int)
    df["pf_ever_finish_loss"] = (
        (df["pf_ko_losses"].fillna(0) + df["pf_sub_losses"].fillna(0)) > 0).astype(int)

    # Standalone cumulative damage absorbed (Ferguson mileage — decoupled from age)
    absorbed = df.groupby("fighter_id", sort=False)["opp_sig_str_landed"].shift(1)
    df["pf_career_damage_absorbed"] = absorbed.groupby(gid).cumsum()
    df["pf_damage_absorbed_per_fight"] = (
        df["pf_career_damage_absorbed"] / df["pf_total_fights"].replace(0, np.nan))

    return df.drop(columns=["_ko_loss", "_sub_loss", "_dec_loss"])



def _fights_since(df, event_series):
    """Fights since the most recent occurrence (shifted). NaN if it never happened."""
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("fighter_id", sort=False).groups.items():
        idx = list(idx)
        last_seen = None
        for pos, i in enumerate(idx):
            if last_seen is not None:
                out[i] = pos - last_seen          # written BEFORE updating → shifted
            if event_series.iloc[i] == 1:
                last_seen = pos
    return out


def add_cardio(df):
    df = df.sort_values(["fighter_id", "event_date", "fight_id"]).reset_index(drop=True)
    gid = df["fighter_id"]
    reached_late = (df["round"] >= 3).astype(int)
    won_late = ((df["round"] >= 3) & (df["is_win"] == 1)).astype(int)

    sh_reached = reached_late.groupby(gid).shift(1)
    sh_won = won_late.groupby(gid).shift(1)
    df["pf_pct_reach_r3"] = sh_reached.groupby(gid).expanding().mean().reset_index(level=0, drop=True)
    df["pf_late_win_rate"] = sh_won.groupby(gid).expanding().mean().reset_index(level=0, drop=True)

    # Avg fight duration to date. ambiguous alone (fast finisher vs. gets finished)
    
    dur = df.groupby("fighter_id", sort=False)["time_sec"].shift(1)
    df["pf_avg_fight_duration"] = dur.groupby(gid).expanding().mean().reset_index(level=0, drop=True)
    return df

def add_elo(df):
    df = df.sort_values(["event_date", "fight_id"]).reset_index(drop=True)
    ratings = {}
    df["pf_elo"] = ELO_BASE
    for _, rows in df.groupby("fight_id", sort=False):
        if len(rows) == 1:
            i = rows.index[0]
            df.at[i, "pf_elo"] = ratings.get(df.at[i, "fighter_id"], ELO_BASE)
            continue
        if len(rows) != 2:
            continue                              # malformed pairing already validated out
        a, b = rows.index[0], rows.index[1]
        fa, fb = df.at[a, "fighter_id"], df.at[b, "fighter_id"]
        Ra, Rb = ratings.get(fa, ELO_BASE), ratings.get(fb, ELO_BASE)
        df.at[a, "pf_elo"], df.at[b, "pf_elo"] = Ra, Rb   # write pre-fight rating
        Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
        Sa = df.at[a, "is_win"]
        if pd.isna(Sa) or df.at[a, "result"] in ("draw", "no_contest"):
            Sa = 0.5                              # non-decisive → nudge toward mean
        ratings[fa] = Ra + ELO_K * (Sa - Ea)
        ratings[fb] = Rb + ELO_K * ((1 - Sa) - (1 - Ea))
    return df


def add_opponent_quality(df):
    opp_elo = (df[["fight_id", "fighter_id", "pf_elo"]]
        .rename(columns={"fighter_id": "opponent_id", "pf_elo": "this_opp_elo"}))
    df = df.merge(opp_elo, on=["fight_id", "opponent_id"], how="left")
    df = df.sort_values(["fighter_id", "event_date", "fight_id"]).reset_index(drop=True)
    gid = df["fighter_id"]
    shifted = df.groupby("fighter_id", sort=False)["this_opp_elo"].shift(1)
    df["pf_sos_avg_opp_elo"] = shifted.groupby(gid).expanding().mean().reset_index(level=0, drop=True)
    df["pf_sos_opp_elo_last5"] = (shifted.groupby(gid).rolling(RECENT_WINDOW, min_periods=1)
        .mean().reset_index(level=0, drop=True))
    # quality-adjusted defense
    baseline = df["pf_sos_avg_opp_elo"].fillna(ELO_BASE)
    df["car_td_def_qadj"] = df["car_td_def"] * (baseline / ELO_BASE)
    df["car_str_def_qadj"] = df["car_str_def"] * (baseline / ELO_BASE)
    return df

def add_physical(df, fighters):
    f = fighters[["fighter_id", "height_in", "weight_lbs", "reach_in",
                  "reach_in_imputed", "reach_imputed", "stance", "dob"]].copy()
    df = df.merge(f, on="fighter_id", how="left")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age_at_fight"] = (df["event_date"] - df["dob"]).dt.days / 365.25   # point-in-time, not leakage
    df["reach_use"] = df["reach_in"].fillna(df["reach_in_imputed"])
    return df

DIFF_FEATURES = [
    # form / momentum
    "pf_win_pct", "pf_win_pct_last5", "pf_win_streak", "pf_total_fights",
    "pf_days_since_last", "pf_fights_last_365d",
    # performance rates
    "car_sig_str_acc", "car_sig_str_att_pervol", "car_td_acc", "car_td_att_pervol",
    "car_str_def", "car_td_def", "car_str_def_qadj", "car_td_def_qadj",
    "car_ctrl_sec", "car_kd", "car_sub_att",
    # trends
    "trend_sig_str_acc", "trend_td_acc", "trend_td_attempts", "trend_sig_output",
    "trend_damage_absorbed", "trend_winpct",
    # loss profile / damage
    "pf_ko_losses", "pf_sub_losses", "pf_dec_losses",
    "pf_fights_since_ko_loss", "pf_fights_since_finish_loss",
    "pf_ever_ko_loss", "pf_ever_finish_loss",
    "pf_career_damage_absorbed", "pf_damage_absorbed_per_fight",
    # cardio
    "pf_pct_reach_r3", "pf_late_win_rate", "pf_avg_fight_duration",
    # opponent quality
    "pf_elo", "pf_sos_avg_opp_elo", "pf_sos_opp_elo_last5",
    # physical
    "age_at_fight", "reach_use", "height_in",
]


def build_training_matrix(df):
    decided = df[df["result"].isin(["win", "loss"])].copy()
    rows, rng = [], np.random.default_rng(42)
    for fid, pair in decided.groupby("fight_id", sort=False):
        if len(pair) != 2:
            continue
        r0, r1 = pair.iloc[0], pair.iloc[1]
        A, B = (r0, r1) if rng.random() < 0.5 else (r1, r0)
        row = {"fight_id": fid, "event_date": A["event_date"], "weight_class": A["weight_class"],
               "A_id": A["fighter_id"], "B_id": B["fighter_id"],
               "label": int(A["is_win"])}
        for feat in DIFF_FEATURES:
            row[f"d_{feat}"] = A.get(feat, np.nan) - B.get(feat, np.nan)
        row["stance_A"], row["stance_B"] = A.get("stance"), B.get("stance")
        row["stance_mismatch"] = int(
            pd.notna(A.get("stance")) and pd.notna(B.get("stance"))
            and A.get("stance") != B.get("stance"))
        row["A_total_fights"] = A.get("pf_total_fights", np.nan)
        row["B_total_fights"] = B.get("pf_total_fights", np.nan)
        rows.append(row)
    mat = pd.DataFrame(rows)
    mat["A_is_debut"] = mat["A_total_fights"].fillna(0) == 0
    mat["B_is_debut"] = mat["B_total_fights"].fillna(0) == 0
   
    return mat   


def build_features(long, fighters):
    long = long.copy()
    long["is_win"] = long["is_win"].astype(float)
    long["event_date"] = pd.to_datetime(long["event_date"])
    long = long.sort_values(["fighter_id", "event_date", "fight_id"]).reset_index(drop=True)

    long = add_momentum(long)
    long = add_rolling_perf(long)
    long = add_trends(long)             
    long = add_loss_profile(long)
    long = add_cardio(long)
    long = add_elo(long)
    long = add_opponent_quality(long)   
    long = add_physical(long, fighters)

    long = long.sort_values(["fighter_id", "event_date", "fight_id"]).reset_index(drop=True)
    matrix = build_training_matrix(long)
    _leakage_smoke_test(long)
    return long, matrix


def _leakage_smoke_test(df):
    firsts = df.sort_values(["fighter_id", "event_date", "fight_id"]).groupby("fighter_id").head(1)
    for c in ["pf_win_pct", "car_sig_str_acc", "pf_days_since_last",
              "pf_career_damage_absorbed", "trend_td_acc"]:
        bad = firsts[c].notna().sum()
        assert bad == 0, f"LEAKAGE: {bad} debut fights have non-null {c}"
    print("test passed:  debut fights have empty history.")

fights_long_features, training_matrix = build_features(fights_long_clean, fighters_clean)

fights_long_features.to_csv(OUT_DIR / "fights_long_features.csv", index=False)
training_matrix.to_csv(OUT_DIR / "training_matrix.csv", index=False)

# ML STEP


tm = training_matrix.copy()
tm = tm.sort_values("event_date").reset_index(drop=True)

tm["weight_class"] = tm["weight_class"].astype("category")

tm["d_is_southpaw"] = (tm["stance_A"] == "Southpaw").astype(int) - (tm["stance_B"] == "Southpaw").astype(int)
tm["d_is_debut"] = tm["A_is_debut"].astype(int) - tm["B_is_debut"].astype(int)

tm = tm[tm["event_date"] >= pd.Timestamp("2012-01-01")].reset_index(drop=True)
n = len(tm)

# Train-Validation-Test Split
train_end = int(n * 0.70)
valid_end = int(n * 0.85)

train = tm.iloc[:train_end]
valid = tm.iloc[train_end:valid_end]
test = tm.iloc[valid_end:]

diff_cols = [c for c in tm.columns if c.startswith("d_")]



feature_cols = diff_cols + ["weight_class"]

def prep_X_y(df):
    return df[feature_cols], df["label"]

X_train, y_train = prep_X_y(train)
X_val, y_val = prep_X_y(valid)
X_test, y_test = prep_X_y(test)

def evaluate(model, X, y, name):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"{name:5s}  acc={accuracy_score(y, pred):.4f}  "
          f"logloss={log_loss(y, proba):.4f}  "
          f"auc={roc_auc_score(y, proba):.4f}  "
          f"brier={brier_score_loss(y, proba):.4f}")


# Baseline XGBoost
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
    enable_categorical=True,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

print(f"\nBest iteration: {xgb_model.best_iteration}")

# ELO accuracy
for name, d in [("val", valid), ("test",test)]:
    elo_pred = (d["d_pf_elo"] > 0).astype(int)
    print(f"{name} Elo accuracy: {accuracy_score(d['label'], elo_pred):.4f}")


print("\nXGBoost evaluation:")
for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
    evaluate(xgb_model, X, y, name)

# feature importance
tg = xgb_model.get_booster().get_score(importance_type="total_gain")
xgb_importances = pd.DataFrame({
    "feature": X_train.columns,
    "total_gain": [tg.get(f, 0.0) for f in X_train.columns],
}).sort_values("total_gain", ascending=False)
print(xgb_importances.to_string())

# Hyperparameter Tuning

# TimeSeriesSplit (4 folds)

trainval = tm.iloc[:valid_end]
X_trainval, y_trainval = prep_X_y(trainval)
tscv = TimeSeriesSplit(n_splits=4)

for i, (tr_idx, va_idx) in enumerate(tscv.split(X_trainval), 1):
    tr_dates = trainval["event_date"].iloc[tr_idx]
    va_dates = trainval["event_date"].iloc[va_idx]
    print(f"fold {i}: train {len(tr_idx):>5} rows ({tr_dates.min().date()} → {tr_dates.max().date()})  "
          f"val {len(va_idx):>5} rows ({va_dates.min().date()} → {va_dates.max().date()})")
    
""""


          
# Tuning


param_dist = {
    "max_depth": randint(2, 6),
    "min_child_weight": randint(1, 20),
    "learning_rate": loguniform(0.01, 0.15),
    "n_estimators": randint(150, 800),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_alpha": uniform(0, 3),
    "reg_lambda": uniform(0, 10),
}

xgb_base = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    enable_categorical=True,
    random_state=42,
)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=60,
    scoring="neg_log_loss",
    cv=tscv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    refit=True,
)

search.fit(X_trainval, y_trainval)

print(f"Best CV log loss: {-search.best_score_:.4f}")
for k, v in sorted(search.best_params_.items()):
    print(f"  {k}: {v}")


# Best Model

best_model = search.best_estimator_
evaluate(best_model, X_trainval, y_trainval, "trainval")
evaluate(best_model, X_test, y_test, "test")
"""

# Final Tuned Model 
TUNED_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    enable_categorical=True,
    random_state=42,
    max_depth=4,
    min_child_weight=17,
    learning_rate=0.0104,
    n_estimators=541,
    subsample=0.537,
    colsample_bytree=0.647,
    reg_alpha=2.187,
    reg_lambda=7.713,
)
tuned_model = xgb.XGBClassifier(**TUNED_PARAMS)

tuned_model.fit(X_trainval, y_trainval)

evaluate(tuned_model, X_trainval, y_trainval, "trainval")
evaluate(tuned_model, X_test, y_test, "test")

# Final Production Model
X_all, y_all = prep_X_y(tm)
final_model = xgb.XGBClassifier(**TUNED_PARAMS)
final_model.fit(X_all,y_all)


today = pd.Timestamp.today().normalize()
name_map = fights_long_clean.groupby("fighter_id")["fighter_name"].last()

pending = pd.DataFrame({
    "fighter_id": name_map.index,
    "fighter_name": name_map.values,
    "fight_id": "PENDING_" + name_map.index,
    "event_date": today,
    "result": "pending",
    "is_win": np.nan,
})
pending = pending.reindex(columns=fights_long_clean.columns)

long_plus = pd.concat([fights_long_clean, pending], ignore_index=True)
feats_plus, _ = build_features(long_plus, fighters_clean)

snapshots = feats_plus[feats_plus["fight_id"].str.startswith("PENDING")].set_index("fighter_id")


# trim to active fighters
last_fight = fights_long_clean.groupby("fighter_id")["event_date"].max()
active = last_fight[last_fight >= today - pd.Timedelta(days=730)].index
snapshots = snapshots.loc[snapshots.index.intersection(active)]

 

Path("app").mkdir(exist_ok=True)
joblib.dump(final_model, "app/model.pkl")
joblib.dump(DIFF_FEATURES, "app/diff_features.pkl")
joblib.dump(feature_cols, "app/feature_cols.pkl")
joblib.dump(list(tm["weight_class"].cat.categories), "app/weight_classes.pkl")
snapshots.to_parquet("app/fighter_snapshots.parquet")