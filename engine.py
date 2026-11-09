"""
Shared decision engine used by the modelling script (04) and the Streamlit app.

It turns three analytics layers into one PRESCRIPTIVE decision:
  * Classification (KNN / Decision Tree / Random Forest / Gradient Boosting)
        -> probability that each department's check will pass
  * Clustering (K-Means)
        -> campaign archetype, used to explain risk and pick the approval lane
  * ARIMA demand forecast
        -> is there enough stock for the demand the campaign will create?
and then applies business rules ("guard-rails") + a what-if optimiser that
proposes the smallest change that turns a risky campaign into a safe one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NUM_FEATURES = [
    "duration_days", "lead_time_to_launch_days", "festive_window", "log_budget", "discount_pct",
    "channel_hist_conv_rate_pct", "brand_fit_score", "expected_uplift_pct", "inventory_coverage_ratio",
    "supplier_lead_time_days", "supplier_lead_gap_days", "warehouse_utilization_pct",
    "logistics_capacity_available_pct", "dept_budget_headroom_pct", "projected_roi",
    "gross_margin_after_disc_pct", "cost_per_reach_inr",
]
CAT_FEATURES = ["region", "product_category", "channel", "campaign_type", "competitor_intensity"]
FEATURES = NUM_FEATURES + CAT_FEATURES

TARGETS = {
    "campaign_success": "Overall GO (delivered in full & profitable)",
    "supply_ok": "Supply Chain check (in-full & on-time delivery)",
    "finance_ok": "Finance check (positive incremental profit)",
    "demand_target_achieved": "Marketing check (>=85% of projected demand)",
}
DEPT_OF = {"supply_ok": "Supply Chain", "finance_ok": "Finance", "demand_target_achieved": "Marketing"}

CLUSTER_FEATURES = ["log_budget", "discount_pct", "projected_roi", "inventory_coverage_ratio",
                    "gross_margin_after_disc_pct", "lead_time_to_launch_days", "duration_days", "brand_fit_score"]

DEFAULT_POLICY = dict(
    green_prob=0.65,        # P(GO) at or above which a campaign is auto-approved
    red_prob=0.35,          # P(GO) below which a campaign is sent back for redesign
    dept_prob=0.50,         # a department is pulled in only if its own P(pass) is below this
    min_coverage=0.80,      # inventory must cover >= 80% of projected units ...
    min_roi=0.0,            # projected incremental ROI must be >= 0
    min_brand_fit=4,        # brand-fit score floor (marketing guard-rail)
    max_wh_util=95.0,       # warehouse utilisation ceiling (%)
    min_headroom=5.0,       # department budget headroom floor (%)
)
UPLIFT_PER_DISCOUNT_PT = 2.7   # marketing's planning rule of thumb (used in what-if)
CANNIBAL = 0.25

LANE_SLA_HOURS = {"GREEN": 1, "AMBER": 24, "RED": 1}
LANE_LABEL = {"GREEN": "Auto-approve", "AMBER": "Fast-track review", "RED": "Redesign required"}


# ----------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute every derived field from the base plan (used for new / what-if requests)."""
    d = df.copy()
    net = d["unit_price_inr"] * (1 - d["discount_pct"] / 100)
    base = d["baseline_weekly_units"] * d["duration_days"] / 7
    d["projected_units"] = np.round(base * (1 + d["expected_uplift_pct"] / 100)).astype(int)
    d["projected_revenue_inr"] = (d["projected_units"] * net).round()
    d["projected_incremental_profit_inr"] = (base * d["expected_uplift_pct"] / 100 * (net - d["unit_cost_inr"])
                                             - base * CANNIBAL * d["unit_price_inr"] * d["discount_pct"] / 100
                                             - d["budget_inr"]).round()
    d["inventory_coverage_ratio"] = (d["inventory_on_hand_units"] / d["projected_units"]).round(3)
    d["projected_roi"] = (d["projected_incremental_profit_inr"] / d["budget_inr"]).round(3)
    d["gross_margin_after_disc_pct"] = ((net - d["unit_cost_inr"]) / net * 100).round(1)
    d["supplier_lead_gap_days"] = d["supplier_lead_time_days"] - d["lead_time_to_launch_days"]
    d["cost_per_reach_inr"] = (d["budget_inr"] / d["target_reach"]).round(3)
    d["log_budget"] = np.log10(d["budget_inr"])
    return d


def predict_all(df: pd.DataFrame, bundle: dict, model_choice: dict | None = None) -> pd.DataFrame:
    """P(pass) for every target using the chosen (default: best) algorithm."""
    out = pd.DataFrame(index=df.index)
    for t, info in bundle["targets"].items():
        name = (model_choice or {}).get(t, info["best"])
        out[f"p_{t}"] = info["models"][name].predict_proba(df[FEATURES])[:, 1]
    return out


def assign_cluster(df: pd.DataFrame, bundle: dict) -> pd.Series:
    X = df[CLUSTER_FEATURES].copy()
    X["projected_roi"] = X["projected_roi"].clip(-3, 5)
    lab = bundle["kmeans"].predict(bundle["cluster_scaler"].transform(X))
    return pd.Series([bundle["cluster_names"][i] for i in lab], index=df.index)


def arima_coverage(row: pd.Series, forecasts: pd.DataFrame | None, region_share: dict) -> float | None:
    """Inventory cover vs. ARIMA-forecast demand for the campaign window (None if outside horizon)."""
    if forecasts is None:
        return None
    start = pd.Timestamp(row["planned_launch_date"])
    end = start + pd.Timedelta(days=int(row["duration_days"]))
    f = forecasts[(forecasts["product_category"] == row["product_category"])
                  & (forecasts["week_start"] >= start - pd.Timedelta(days=6)) & (forecasts["week_start"] <= end)]
    if f.empty:
        return None
    weekly = f["forecast_units"].mean() * region_share.get(row["region"], 0.2)
    demand = weekly * row["duration_days"] / 7 * (1 + row["expected_uplift_pct"] / 100)
    return float(row["inventory_on_hand_units"] / demand)


@dataclass
class Decision:
    lane: str
    p_go: float
    dept_probs: dict
    flags: list = field(default_factory=list)       # (department, message)
    route_to: list = field(default_factory=list)
    cluster: str = ""
    arima_cov: float | None = None


def decide(row: pd.Series, probs: pd.Series, policy: dict, cluster: str = "", arima_cov=None) -> Decision:
    flags = []
    cov = arima_cov if arima_cov is not None else row["inventory_coverage_ratio"]
    if cov < policy["min_coverage"]:
        msg = f"Stock covers only {cov:.0%} of expected demand"
        if row["supplier_lead_time_days"] > row["lead_time_to_launch_days"]:
            msg += f" and replenishment ({row['supplier_lead_time_days']:.0f} d) arrives after launch"
        flags.append(("Supply Chain", msg))
    if row["warehouse_utilization_pct"] > policy["max_wh_util"]:
        flags.append(("Supply Chain", f"Warehouse {row['warehouse_utilization_pct']:.0f}% full - delivery risk"))
    if row["projected_roi"] < policy["min_roi"]:
        flags.append(("Finance", f"Projected ROI {row['projected_roi']:.2f}x is below the {policy['min_roi']:.2f}x hurdle"))
    if row["dept_budget_headroom_pct"] < policy["min_headroom"]:
        flags.append(("Finance", f"Only {row['dept_budget_headroom_pct']:.0f}% budget headroom left this quarter"))
    if row["brand_fit_score"] < policy["min_brand_fit"]:
        flags.append(("Marketing", f"Brand-fit score {row['brand_fit_score']:.0f}/10 below floor"))

    p_go = float(probs["p_campaign_success"])
    dept = {DEPT_OF[t]: float(probs[f"p_{t}"]) for t in DEPT_OF}
    route = sorted({d for d, _ in flags} | {d for d, p in dept.items() if p < policy["dept_prob"]})

    if p_go < policy["red_prob"] or len(flags) >= 3:
        lane = "RED"
    elif p_go >= policy["green_prob"] and not flags:
        lane = "GREEN"
        route = []
    else:
        lane = "AMBER"
        if not route:
            route = [min(dept, key=dept.get)]
    return Decision(lane, p_go, dept, flags, route, cluster, arima_cov)


def prescribe(row: pd.Series, bundle: dict, policy: dict, model_choice=None, top=4) -> pd.DataFrame:
    """What-if optimiser: try realistic single levers + the best pair; rank by P(GO) gain."""
    base = row.to_frame().T.copy()
    for c in base.columns:
        try:
            base[c] = pd.to_numeric(base[c])
        except (ValueError, TypeError):
            pass
    p0 = predict_all(add_features(base), bundle, model_choice)["p_campaign_success"].iloc[0]
    cands = []

    def try_(label, dept, mutate):
        x = base.copy()
        mutate(x)
        x = add_features(x)
        p = predict_all(x, bundle, model_choice)["p_campaign_success"].iloc[0]
        cands.append(dict(action=label, owner=dept, new_p_go=p, gain=p - p0,
                          new_roi=float(x["projected_roi"].iloc[0]), new_coverage=float(x["inventory_coverage_ratio"].iloc[0]),
                          _m=mutate))

    d0 = float(row["discount_pct"])
    for d in [d0 - 5, d0 - 10, d0 - 15]:
        if d >= 0:
            def m(x, d=d):
                x["expected_uplift_pct"] = max(2, float(x["expected_uplift_pct"].iloc[0]) - UPLIFT_PER_DISCOUNT_PT * (d0 - d))
                x["discount_pct"] = d
            try_(f"Reduce discount {d0:.0f}% -> {d:.0f}%", "Marketing", m)
    for f in [0.75, 0.6]:
        def m(x, f=f):
            x["budget_inr"] = round(float(x["budget_inr"].iloc[0]) * f, -3)
            x["target_reach"] = int(float(x["target_reach"].iloc[0]) * (0.5 + 0.5 * f))
        try_(f"Trim media budget by {1 - f:.0%} (to INR {row['budget_inr'] * f:,.0f})", "Finance", m)
    need = float(row["projected_units"]) * 1.10 - float(row["inventory_on_hand_units"])
    if need > 0:
        def m(x, need=need):
            x["inventory_on_hand_units"] = float(x["inventory_on_hand_units"].iloc[0]) + need
        try_(f"Pre-position {need:,.0f} extra units before launch", "Supply Chain", m)
    for wk in [7, 14]:
        def m(x, wk=wk):
            x["lead_time_to_launch_days"] = float(x["lead_time_to_launch_days"].iloc[0]) + wk
        try_(f"Push launch back {wk} days (lets replenishment arrive)", "Supply Chain", m)
    for ch in ["Email", "Search Ads", "In-Store"]:
        if ch != row["channel"]:
            def m(x, ch=ch):
                x["channel"] = ch
            try_(f"Switch channel {row['channel']} -> {ch}", "Marketing", m)

    res = pd.DataFrame(cands).sort_values("gain", ascending=False)
    good = res[res["gain"] > 0.01]
    if len(good) >= 2:      # best combination of two levers from different owners
        a = good.iloc[0]
        b = good[good["owner"] != a["owner"]].head(1)
        if len(b):
            b = b.iloc[0]
            x = base.copy()
            a["_m"](x)
            b["_m"](x)
            x = add_features(x)
            p = predict_all(x, bundle, model_choice)["p_campaign_success"].iloc[0]
            res = pd.concat([pd.DataFrame([dict(action=f"{a['action']}  +  {b['action']}", owner=f"{a['owner']} + {b['owner']}",
                                                new_p_go=p, gain=p - p0, new_roi=float(x["projected_roi"].iloc[0]),
                                                new_coverage=float(x["inventory_coverage_ratio"].iloc[0]), _m=None)]), res])
    res = res.drop(columns="_m").sort_values("gain", ascending=False).head(top)
    res["current_p_go"] = p0
    res["reaches_green"] = res["new_p_go"] >= policy["green_prob"]
    return res.reset_index(drop=True)
