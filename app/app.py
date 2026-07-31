import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from pathlib import Path
from streamlit_searchbox import st_searchbox

st.set_page_config(page_title="UFC Fight Predictor", page_icon="\U0001F94A", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data" / "processed"

# Okabe-Ito colorblind-safe pair. Color is assigned to fighter identity
# (A always blue, B always vermillion) and never to who is favored.
COLOR_A = "#0072B2"
COLOR_B = "#D55E00"

DIVISIONS = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
    "Middleweight", "Light Heavyweight", "Heavyweight",
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
]

FEATURE_LABELS = {
    "d_pf_win_pct": "Career win %",
    "d_pf_win_pct_last5": "Win % (last 5)",
    "d_pf_win_streak": "Current streak",
    "d_pf_total_fights": "Total pro fights",
    "d_pf_days_since_last": "Days since last fight",
    "d_pf_fights_last_365d": "Fights in last 365 days",
    "d_car_sig_str_acc": "Career striking accuracy",
    "d_car_sig_str_att_pervol": "Career strike volume",
    "d_car_td_acc": "Career takedown accuracy",
    "d_car_td_att_pervol": "Career takedown volume",
    "d_car_str_def": "Career striking defense",
    "d_car_td_def": "Career takedown defense",
    "d_car_str_def_qadj": "Quality-adj. striking defense",
    "d_car_td_def_qadj": "Quality-adj. takedown defense",
    "d_car_ctrl_sec": "Career control time/fight",
    "d_car_kd": "Career knockdowns/fight",
    "d_car_sub_att": "Career sub attempts/fight",
    "d_trend_sig_str_acc": "Striking accuracy trend",
    "d_trend_td_acc": "Takedown accuracy trend",
    "d_trend_td_attempts": "Takedown volume trend",
    "d_trend_sig_output": "Striking output trend",
    "d_trend_damage_absorbed": "Damage absorbed trend",
    "d_trend_winpct": "Win % trend",
    "d_pf_ko_losses": "KO/TKO losses",
    "d_pf_sub_losses": "Submission losses",
    "d_pf_dec_losses": "Decision losses",
    "d_pf_fights_since_ko_loss": "Fights since KO loss",
    "d_pf_fights_since_finish_loss": "Fights since finish loss",
    "d_pf_ever_ko_loss": "Ever KO'd",
    "d_pf_ever_finish_loss": "Ever finished",
    "d_pf_career_damage_absorbed": "Career damage absorbed",
    "d_pf_damage_absorbed_per_fight": "Damage absorbed/fight",
    "d_pf_pct_reach_r3": "% fights reaching round 3",
    "d_pf_late_win_rate": "Late-round win rate",
    "d_pf_avg_fight_duration": "Avg fight duration",
    "d_pf_elo": "Elo rating",
    "d_pf_sos_avg_opp_elo": "Career strength of schedule",
    "d_pf_sos_opp_elo_last5": "Strength of schedule (last 5)",
    "d_age_at_fight": "Age",
    "d_reach_use": "Reach",
    "d_height_in": "Height",
    "d_is_southpaw": "Southpaw stance",
    "d_is_debut": "UFC debut",
    "weight_class": "Weight class",
}

# (label, column, formatter, higher_is_better)
STAT_SPECS = [
    ("Elo rating", "pf_elo", lambda v: f"{v:.0f}", True),
    ("Career win %", "pf_win_pct", lambda v: f"{v * 100:.1f}%", True),
    ("Win % (last 5)", "pf_win_pct_last5", lambda v: f"{v * 100:.1f}%", True),
    ("Current streak", "pf_win_streak", None, True),
    ("Age", "age_at_fight", lambda v: f"{v:.1f} yrs", None),
    ("Height", "height_in", lambda v: f"{v:.0f} in", None),
    ("Reach", "reach_use", lambda v: f"{v:.0f} in", None),
    ("Stance", "stance", lambda v: str(v), None),
    ("Striking accuracy", "car_sig_str_acc", lambda v: f"{v * 100:.1f}%", True),
    ("Striking defense", "car_str_def", lambda v: f"{v * 100:.1f}%", True),
    ("Takedown accuracy", "car_td_acc", lambda v: f"{v * 100:.1f}%", True),
    ("Takedown defense", "car_td_def", lambda v: f"{v * 100:.1f}%", True),
    ("Control time/fight", "car_ctrl_sec", lambda v: f"{v:.0f} s", True),
    ("Knockdowns/fight", "car_kd", lambda v: f"{v:.2f}", True),
    ("Sub attempts/fight", "car_sub_att", lambda v: f"{v:.2f}", None),
    ("Damage absorbed/fight", "pf_damage_absorbed_per_fight", lambda v: f"{v:.1f}", False),
    ("Days since last fight", "pf_days_since_last", lambda v: f"{v:.0f}", None),
]


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.pkl")


@st.cache_resource
def load_artifacts():
    diff_features = joblib.load(APP_DIR / "diff_features.pkl")
    feature_cols = joblib.load(APP_DIR / "feature_cols.pkl")
    weight_classes = joblib.load(APP_DIR / "weight_classes.pkl")
    return diff_features, feature_cols, weight_classes


@st.cache_data
def load_division_and_record():
    cols = ["fighter_id", "weight_class", "event_date", "result"]
    df = pd.read_csv(DATA_DIR / "fights_long_features.csv", usecols=cols, parse_dates=["event_date"])

    last_division = (
        df.dropna(subset=["weight_class"])
        .sort_values("event_date")
        .groupby("fighter_id")["weight_class"]
        .last()
    )

    fought_divisions = (
        df.dropna(subset=["weight_class"])
        .groupby("fighter_id")["weight_class"]
        .agg(lambda s: set(s.unique()))
    )

    counts = df.groupby(["fighter_id", "result"]).size().unstack(fill_value=0)
    for c in ["win", "loss", "draw", "no_contest"]:
        if c not in counts.columns:
            counts[c] = 0
    record = counts[["win", "loss", "draw", "no_contest"]]
    return last_division,fought_divisions ,record


@st.cache_data
def build_roster():
    snap = pd.read_parquet(APP_DIR / "fighter_snapshots.parquet")
    last_division, fought_divisions, record = load_division_and_record()

    roster = snap.copy()
    roster["division"] = last_division.reindex(roster.index)
    roster["fought_divisions"] = fought_divisions.reindex(roster.index)
    roster["fought_divisions"] = roster["fought_divisions"].apply(
        lambda s: s if isinstance(s, set) else set()
    )
    roster = roster.join(record.reindex(roster.index).fillna(0).astype(int))
    roster["gender"] = np.where(
        roster["division"].str.startswith("Women", na=False), "Women", "Men"
    )
    return roster


def record_str(row):
    nc = f" ({row['no_contest']} NC)" if row["no_contest"] else ""
    return f"{row['win']}-{row['loss']}-{row['draw']}{nc}"


def filter_by_name_prefix(df, query):
    """Keep rows where the search text matches the start of a first or last name token."""
    if not query:
        return df
    q = query.strip().lower()
    if not q:
        return df
    mask = df["fighter_name"].apply(
        lambda name: any(tok.lower().startswith(q) for tok in name.split())
    )
    return df[mask]


def build_matchup_row(roster, diff_features, feature_cols, weight_classes, id_a, id_b, weight_class):
    A, B = roster.loc[id_a], roster.loc[id_b]
    row = {f"d_{feat}": A[feat] - B[feat] for feat in diff_features}
    row["d_is_southpaw"] = int(A["stance"] == "Southpaw") - int(B["stance"] == "Southpaw")
    row["d_is_debut"] = int(A["pf_total_fights"] == 0) - int(B["pf_total_fights"] == 0)
    row["weight_class"] = weight_class
    X = pd.DataFrame([row])[feature_cols]
    X["weight_class"] = pd.Categorical(X["weight_class"], categories=weight_classes)
    return X, A, B


def render_probability_bar(name_a, proba_a, name_b, proba_b):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Win probability"], x=[proba_a], name=name_a, orientation="h",
        marker_color=COLOR_A, text=[f"{name_a}  {proba_a * 100:.1f}%"],
        textposition="inside", insidetextanchor="middle",
        hovertemplate=f"{name_a}: %{{x:.1%}}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=["Win probability"], x=[proba_b], name=name_b, orientation="h",
        marker_color=COLOR_B, text=[f"{name_b}  {proba_b * 100:.1f}%"],
        textposition="inside", insidetextanchor="middle",
        hovertemplate=f"{name_b}: %{{x:.1%}}<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        height=140,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=15),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_stat_comparison(name_a, name_b, A, B, record_a, record_b):
    rows = []
    for label, col, fmt, higher_better in STAT_SPECS:
        va, vb = A.get(col), B.get(col)
        disp_a = "—" if pd.isna(va) else (fmt(va) if fmt else str(va))
        disp_b = "—" if pd.isna(vb) else (fmt(vb) if fmt else str(vb))
        edge = ""
        if higher_better is not None and pd.notna(va) and pd.notna(vb) and va != vb:
            a_wins = (va > vb) if higher_better else (va < vb)
            edge = name_a if a_wins else name_b
        rows.append({"Stat": label, name_a: disp_a, name_b: disp_b, "Edge": edge})

    df = pd.DataFrame(
        [{"Stat": "Record (W-L-D)", name_a: record_a, name_b: record_b, "Edge": ""}] + rows
    ).set_index("Stat")

    def highlight_edge(display_row, edge):
        styles = [""] * len(display_row)
        if edge == name_a:
            styles[display_row.index.get_loc(name_a)] = "background-color: rgba(0,114,178,0.18); font-weight: 600"
        elif edge == name_b:
            styles[display_row.index.get_loc(name_b)] = "background-color: rgba(213,94,0,0.18); font-weight: 600"
        return styles

    edge_by_stat = df["Edge"]
    styled = df.drop(columns="Edge").style.apply(
        lambda r: highlight_edge(r, edge_by_stat.loc[r.name]), axis=1
    )
    st.dataframe(styled, use_container_width=True, height=38 + 35 * len(df))


def render_shap_waterfall(model, X, name_a, name_b):
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    sv.feature_names = [FEATURE_LABELS.get(c, c) for c in X.columns]

    fig = plt.figure()
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)
    st.caption(
        f"Values are in log-odds space. Bars pushing right (red) move the prediction "
        f"toward **{name_a}** winning; bars pushing left (blue) move it toward **{name_b}**."
    )


def fighter_search_function(pool, exclude_id=None):
    dup = pool["fighter_name"].duplicated(keep=False)

    def _label(fid, name):
        return f"{name} ({fid[:6]})" if dup.get(fid, False) else name

    def _search(searchterm):
        matches = filter_by_name_prefix(pool, searchterm) if searchterm else pool
        if exclude_id is not None:
            matches = matches[matches.index != exclude_id]
        matches = matches.head(25)
        return [(_label(fid, row["fighter_name"]), fid) for fid, row in matches.iterrows()]

    return _search


def predictor_page(roster, model, diff_features, feature_cols, weight_classes):
    st.title("UFC Fight Outcome Predictor")
    st.caption("Pick a weight class, then search for the two fighters to match up.")

    weight_class = st.selectbox("1. Weight class", DIVISIONS, index=0)
    division_pool = roster[roster["fought_divisions"].apply(lambda s: weight_class in s)].sort_values("fighter_name")

    st.markdown("**2. Choose the two fighters**")
    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"Fighter A · {len(division_pool)} fighters have fought at {weight_class}")
        id_a = st_searchbox(
            fighter_search_function(division_pool),
            label="Fighter A",
            key=f"searchbox_a_{weight_class}",
            clear_on_submit=False,
        )
    with c2:
        st.caption(f"Fighter B · {len(division_pool)} fighters have fought at {weight_class}")
        id_b = st_searchbox(
            fighter_search_function(division_pool, exclude_id=id_a),
            label="Fighter B",
            key=f"searchbox_b_{weight_class}",
            clear_on_submit=False,
        )

    if not st.button("Predict", use_container_width=True):
        return

    if not id_a or not id_b:
        st.error("Search for and select both fighters before predicting.")
        return
    if id_a == id_b:
        st.error("Choose two different fighters.")
        return

    X, A, B = build_matchup_row(roster, diff_features, feature_cols, weight_classes, id_a, id_b, weight_class)
    proba_a = float(model.predict_proba(X)[0, 1])
    proba_b = 1 - proba_a
    name_a, name_b = A["fighter_name"], B["fighter_name"]

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric(name_a, f"{proba_a * 100:.1f}%")
    m2.metric(name_b, f"{proba_b * 100:.1f}%")
    render_probability_bar(name_a, proba_a, name_b, proba_b)

    st.subheader("Tale of the tape")
    render_stat_comparison(name_a, name_b, A, B, record_str(A), record_str(B))

    st.subheader("What drove this prediction")
    render_shap_waterfall(model, X, name_a, name_b)


def render_leaderboard_table(df):
    if df.empty:
        st.info("No ranked fighters found for this group.")
        return
    out = df.reset_index(drop=True).copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    out["Elo"] = out["pf_elo"].round(0).astype(int)
    out["Record (W-L-D)"] = out.apply(record_str, axis=1)
    out = out.rename(columns={"fighter_name": "Fighter", "division": "Division"})
    st.dataframe(
        out[["Rank", "Fighter", "Division", "Elo", "Record (W-L-D)"]],
        use_container_width=True,
        hide_index=True,
        height=38 + 35 * len(out),
    )


def leaderboard_page(roster):
    st.title("Elo Leaderboard")
    st.caption("Current Elo ratings for active fighters (fought within the last ~3 years).")

    tab_labels = DIVISIONS + ["Pound-for-Pound"]
    tabs = st.tabs(tab_labels)

    for division, tab in zip(DIVISIONS, tabs[:-1]):
        with tab:
            sub = roster[roster["division"] == division].sort_values("pf_elo", ascending=False).head(15)
            render_leaderboard_table(sub)

    with tabs[-1]:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Men")
            men = roster[roster["gender"] == "Men"].sort_values("pf_elo", ascending=False).head(15)
            render_leaderboard_table(men)
        with c2:
            st.subheader("Women")
            women = roster[roster["gender"] == "Women"].sort_values("pf_elo", ascending=False).head(15)
            render_leaderboard_table(women)


def main():
    roster = build_roster()
    model = load_model()
    diff_features, feature_cols, weight_classes = load_artifacts()

    st.sidebar.title("UFC Prediction Engine")
    page = st.sidebar.radio("Page", ["Fight Predictor", "Elo Leaderboard"])

    if page == "Fight Predictor":
        predictor_page(roster, model, diff_features, feature_cols, weight_classes)
    else:
        leaderboard_page(roster)


if __name__ == "__main__":
    main()
