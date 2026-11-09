"""
Aurora Campaign Command Center
==============================
One decision system shared by Marketing, Supply Chain, Finance, the C-suite and Admins.
Run:  streamlit run app/streamlit_app.py
"""
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from engine import (DEFAULT_POLICY, DEPT_OF, TARGETS, add_features, arima_coverage,  # noqa: E402
                    assign_cluster, decide, predict_all, prescribe)
from ui import (AQUA, BLUE, CLUSTER_COLORS, CLUSTER_SYMBOLS, DIVERGING, INK_2, LANE_ICON, LANE_TEXT,  # noqa: E402
                ORANGE, OUTCOME, SEQ_BLUE, SERIES, STATUS, YELLOW, inject_css, inr, insight, kpi, lane_badge,
                lane_label, show, title)

st.set_page_config(page_title="Aurora Campaign Command Center", page_icon="🎯", layout="wide")
inject_css()

REGION_SHARE = {"North": 0.24, "South": 0.22, "West": 0.22, "East": 0.16, "Central": 0.16}
TODAY = date(2026, 11, 9)
ALGOS = ["K-Nearest Neighbours", "Decision Tree", "Random Forest", "Gradient Boosted Trees"]
NEW_DAYS = {"GREEN": 1 / 24, "AMBER": 1.0, "RED": 1 / 24}   # decision time under the new system (days)


# ============================================================================ data
def F(rel):
    """Path to a data file: project layout (data/, outputs/, models/) or the flat Streamlit-Cloud bundle."""
    p = ROOT / rel
    return p if p.exists() else Path(__file__).parent / Path(rel).name


@st.cache_data
def load():
    hist = pd.read_csv(F("data/clean/campaign_requests_clean.csv"), parse_dates=["submit_date", "planned_launch_date"])
    hist = hist.merge(pd.read_csv(F("outputs/cluster_assignments.csv")), on="campaign_id", how="left")
    hist["outcome"] = hist["campaign_result"].replace({"No outcome (rejected)": "Rejected"})
    return dict(
        hist=hist,
        pend=pd.read_csv(F("data/clean/pending_campaign_requests_clean.csv"), parse_dates=["submit_date", "planned_launch_date"]),
        sales=pd.read_csv(F("data/clean/weekly_category_sales_clean.csv"), parse_dates=["week_start"]),
        inv=pd.read_csv(F("data/clean/inventory_snapshot_clean.csv")),
        log=pd.read_csv(F("data/clean/cleaning_log.csv")),
        fc=pd.read_csv(F("outputs/arima_forecasts.csv"), parse_dates=["week_start"]),
        hold=pd.read_csv(F("outputs/arima_holdout.csv"), parse_dates=["week_start"]),
        arsel=pd.read_csv(F("outputs/arima_model_selection.csv")),
        vol=pd.read_csv(F("outputs/weekly_request_volume.csv"), parse_dates=["week_start"]),
        metrics=pd.read_csv(F("outputs/model_comparison.csv")),
        roc=pd.read_csv(F("outputs/roc_curves.csv")),
        cm=pd.read_csv(F("outputs/confusion_matrices.csv")),
        imp=pd.read_csv(F("outputs/feature_importance.csv")),
        ksel=pd.read_csv(F("outputs/kmeans_k_selection.csv")),
        cprof=pd.read_csv(F("outputs/cluster_profiles.csv")),
        replay=pd.read_csv(F("outputs/decision_replay_test.csv")),
        rules=(F("outputs/decision_tree_rules.txt")).read_text(),
    )


@st.cache_resource
def load_bundle():
    return joblib.load(F("models/model_bundle.joblib"))


D = load()
B = load_bundle()
if "policy" not in st.session_state:
    st.session_state.policy = dict(DEFAULT_POLICY)
if "choice" not in st.session_state:
    st.session_state.choice = {t: B["targets"][t]["best"] for t in TARGETS}


@st.cache_data(show_spinner=False)
def run_engine(df: pd.DataFrame, policy_items: tuple, choice_items: tuple, use_arima: bool) -> pd.DataFrame:
    policy, choice = dict(policy_items), dict(choice_items)
    probs = predict_all(df, B, choice)
    clusters = assign_cluster(df, B)
    rows = []
    for i, r in df.iterrows():
        ac = arima_coverage(r, D["fc"], REGION_SHARE) if use_arima else None
        d = decide(r, probs.loc[i], policy, clusters[i], ac)
        rows.append(dict(lane=d.lane, p_go=d.p_go, p_marketing=d.dept_probs["Marketing"],
                         p_supply=d.dept_probs["Supply Chain"], p_finance=d.dept_probs["Finance"],
                         route_to=", ".join(d.route_to) or "—", flags=" | ".join(f"{a}: {b}" for a, b in d.flags) or "—",
                         n_flags=len(d.flags), cluster_live=d.cluster, arima_cov=ac))
    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)


def engine(df, use_arima=False):
    return run_engine(df, tuple(sorted(st.session_state.policy.items())),
                      tuple(sorted(st.session_state.choice.items())), use_arima)


# ============================================================================ sidebar
st.sidebar.markdown("## 🎯 Aurora\n**Campaign Command Center**")
PAGES = ["🏠 Overview & Analytics Journey", "🔍 EDA: Descriptive & Diagnostic", "🛠️ Admin Portal", "📣 Marketing",
         "🚚 Supply Chain & Inventory", "💰 Finance", "🏛️ C-Suite"]
SLUGS = ["overview", "eda", "admin", "marketing", "supply", "finance", "csuite"]   # deep links: ?page=finance
_qp = st.query_params.get("page", "overview")
page = st.sidebar.radio("Choose your workspace", PAGES, index=SLUGS.index(_qp) if _qp in SLUGS else 0,
                        label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**Filters for historical views**")
hist_all = D["hist"]
years = st.sidebar.multiselect("Year", sorted(hist_all.submit_year.unique()), default=sorted(hist_all.submit_year.unique()))
regions = st.sidebar.multiselect("Region", sorted(hist_all.region.unique()), default=sorted(hist_all.region.unique()))
cats = st.sidebar.multiselect("Product category", sorted(hist_all.product_category.unique()),
                              default=sorted(hist_all.product_category.unique()))
H = hist_all[hist_all.submit_year.isin(years) & hist_all.region.isin(regions) & hist_all.product_category.isin(cats)]
HL = H[H.has_outcome == 1]          # campaigns with an observed outcome
st.sidebar.caption(f"{len(H):,} of {len(hist_all):,} historical campaigns selected · data as of {TODAY:%d %b %Y}")
if HL.empty:
    st.warning("No campaigns match the filters - widen the selection in the sidebar.")
    st.stop()


# ============================================================================ shared widgets
def decision_card(row: pd.Series, res: pd.Series, show_prescription=True, key="x"):
    pol = st.session_state.policy
    lane = res["lane"]
    st.markdown(f"<div style='margin:6px 0 14px 0'>{lane_badge(lane)} &nbsp; "
                f"<span style='color:{INK_2}'>Routed to: <b>{res['route_to']}</b> · decision SLA "
                f"{'1 hour' if lane != 'AMBER' else '24 hours (parallel review)'}</span></div>", unsafe_allow_html=True)
    g, c1, c2, c3 = st.columns([1.25, 1, 1, 1])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=res["p_go"] * 100, number=dict(suffix="%", font=dict(size=34)),
        title=dict(text="P(GO) - delivered & profitable", font=dict(size=13, color=INK_2)),
        gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color="#1B2333", thickness=0.28),
                   steps=[dict(range=[0, pol["red_prob"] * 100], color="#F6CFCF"),
                          dict(range=[pol["red_prob"] * 100, pol["green_prob"] * 100], color="#FDE9B8"),
                          dict(range=[pol["green_prob"] * 100, 100], color="#CDEFCD")])))
    fig.update_layout(height=230, margin=dict(l=40, r=40, t=40, b=10))
    with g:
        show(fig)
    cov = res["arima_cov"] if pd.notna(res.get("arima_cov", np.nan)) else row["inventory_coverage_ratio"]
    for col, dept, p, lines in [
        (c1, "📣 Marketing", res["p_marketing"], [f"Brand fit: <b>{row['brand_fit_score']:.0f}/10</b>",
                                                f"Archetype: <b>{res['cluster_live']}</b>"]),
        (c2, "🚚 Supply Chain", res["p_supply"], [f"Stock cover: <b>{row['inventory_coverage_ratio']:.0%}</b> of plan",
                                                 f"ARIMA-adjusted cover: <b>{cov:.0%}</b>",
                                                 f"Supplier lead {row['supplier_lead_time_days']:.0f} d vs launch in {row['lead_time_to_launch_days']:.0f} d"]),
        (c3, "💰 Finance", res["p_finance"], [f"Projected ROI: <b>{row['projected_roi']:.2f}x</b>",
                                             f"Projected profit: <b>{inr(row['projected_incremental_profit_inr'])}</b>",
                                             f"Margin after discount: <b>{row['gross_margin_after_disc_pct']:.0f}%</b>"])]:
        ok = p >= pol["dept_prob"]
        icon = "✅" if ok else "⚠️"
        col.markdown(f"<div class='card'><h4>{dept}</h4><div class='big'>{icon} {p:.0%}</div>"
                     f"<div class='flag'>probability this check passes</div>"
                     + "".join(f"<div class='flag'>{x}</div>" for x in lines) + "</div>", unsafe_allow_html=True)
    if res["flags"] != "—":
        st.markdown("**Guard-rail flags:** " + "  ".join(f"`{f}`" for f in res["flags"].split(" | ")))
    if show_prescription:
        st.markdown("##### 🧭 Prescriptive levers - smallest changes that raise P(GO)")
        with st.spinner("Running what-if optimiser..."):
            rx = prescribe(row, B, st.session_state.policy, st.session_state.choice)
        if rx.empty or rx["gain"].max() <= 0.005:
            st.success("No lever materially improves this campaign - it is already close to its best design.")
        else:
            rx = rx[rx["gain"] > 0.005]
            if rx["new_p_go"].max() < st.session_state.policy["red_prob"]:
                st.warning(f"Even the best lever only lifts P(GO) to {rx['new_p_go'].max():.0%}. This design is structurally unprofitable or "
                           "unsupplyable - recommend a full redesign (e.g. move away from deep discounts / high-cost channels).")
            st.dataframe(
                rx.assign(reaches=np.where(rx.reaches_green, "✅ GREEN", "—"))[
                    ["action", "owner", "current_p_go", "new_p_go", "new_roi", "new_coverage", "reaches"]],
                hide_index=True, key=f"rx_{key}",
                column_config={"action": st.column_config.TextColumn("Recommended action", width="large"),
                               "owner": "Owner", "current_p_go": st.column_config.NumberColumn("P(GO) now", format="percent"),
                               "new_p_go": st.column_config.ProgressColumn("P(GO) after", format="percent", min_value=0, max_value=1),
                               "new_roi": st.column_config.NumberColumn("New ROI", format="%.2fx"),
                               "new_coverage": st.column_config.NumberColumn("New stock cover", format="percent"),
                               "reaches": "Lane"})


def lane_bar(df, title_txt):
    c = df["lane"].value_counts().reindex(["GREEN", "AMBER", "RED"]).fillna(0).reset_index()
    c.columns = ["lane", "n"]
    c["label"] = c["lane"].map(lambda x: f"{LANE_ICON[x]} {x}")
    fig = px.bar(c, x="n", y="label", orientation="h", color="lane", color_discrete_map=STATUS, text="n",
                 title=title_txt, labels={"n": "Campaigns", "label": ""})
    fig.update_traces(textposition="outside", hovertemplate="%{y}: %{x} campaigns<extra></extra>")
    fig.update_layout(showlegend=False, yaxis=dict(categoryorder="array", categoryarray=c["label"][::-1]))
    return fig


QUEUE_COLS = {"campaign_id": "Campaign", "lane_lbl": "Lane", "p_go": st.column_config.ProgressColumn("P(GO)", format="percent", min_value=0, max_value=1),
              "route_to": "Routed to", "flags": st.column_config.TextColumn("Flags", width="large"),
              "channel": "Channel", "campaign_type": "Type", "product_category": "Category", "region": "Region",
              "budget_inr": st.column_config.NumberColumn("Budget (₹)", format="%d"), "discount_pct": "Disc %",
              "planned_launch_date": st.column_config.DateColumn("Launch", format="DD MMM YYYY"),
              "cluster_live": "Archetype"}


def queue_table(q, key):
    q = q.assign(lane_lbl=q["lane"].map(lane_label)).sort_values(["lane", "p_go"], key=lambda s: s.map({"RED": 0, "AMBER": 1, "GREEN": 2}) if s.name == "lane" else s)
    st.dataframe(q[list(QUEUE_COLS)], hide_index=True, column_config=QUEUE_COLS, key=key, height=380)


# ============================================================================ pages
def page_overview():
    title("Aurora Campaign Command Center",
          "One shared, data-driven approval system for Marketing, Supply Chain and Finance - so good campaigns launch on time and bad ones are fixed before they lose money.")
    late = HL["launched_late"].mean()
    c = st.columns(5)
    kpi(c[0], "Historical campaigns", f"{len(H):,}", "Jan 2024 - Sep 2026")
    kpi(c[1], "Manual approval time", f"{H['total_approval_days'].mean():.1f} days", "3 sequential sign-offs + rework", "bad")
    kpi(c[2], "Launched late", f"{late:.0%}", "approval took longer than lead time", "bad")
    kpi(c[3], "Delivered & profitable", f"{HL['campaign_success'].mean():.0%}", "historical success rate")
    kpi(c[4], "Requests awaiting decision", f"{len(D['pend'])}", "live queue (last 2 weeks)")
    st.markdown("")

    a, b = st.columns(2)
    with a:
        st.markdown("#### ❌ As-is: sequential approvals")
        m, s_, f, r = (H[x].mean() for x in ["marketing_review_days", "supply_chain_review_days", "finance_review_days", "rework_loops"])
        steps = pd.DataFrame({"stage": ["Marketing review", "Supply-chain review", "Finance review", "Rework loops", "Hand-off waiting"],
                              "days": [m, s_, f, r * 2.5, H["total_approval_days"].mean() - m - s_ - f - r * 2.5]})
        steps["start"] = steps["days"].cumsum() - steps["days"]
        fig = go.Figure(go.Bar(x=steps["days"], y=steps["stage"], base=steps["start"], orientation="h",
                               marker_color=[BLUE, AQUA, YELLOW, ORANGE, "#B9BEC7"],
                               text=[f"{d:.1f} d" for d in steps["days"]], textposition="inside",
                               hovertemplate="%{y}: %{x:.1f} days<extra></extra>"))
        fig.update_layout(title="Average approval timeline (days)", yaxis=dict(autorange="reversed"), xaxis_title="Days since submission")
        show(fig, 300)
    with b:
        st.markdown("#### ✅ To-be: parallel, model-driven decision")
        st.markdown("""
<div class='card'>
<b>1. Submit once</b> - Marketing enters the campaign plan; stock, lead-time, warehouse and budget data are auto-pulled from SCM/ERP.<br><br>
<b>2. Three digital checks run in parallel (seconds)</b><br>
&nbsp;&nbsp;📣 Marketing - P(demand target met) · campaign archetype (K-Means)<br>
&nbsp;&nbsp;🚚 Supply Chain - P(in-full, on-time delivery) · ARIMA-adjusted stock cover<br>
&nbsp;&nbsp;💰 Finance - P(profitable) · ROI hurdle · budget headroom<br><br>
<b>3. One lane decision</b><br>
&nbsp;&nbsp;✅ <b>GREEN</b> auto-approve (≤1 h) &nbsp; ⚠️ <b>AMBER</b> only the flagged team reviews (≤24 h) &nbsp; ⛔ <b>RED</b> sent back with fixes (≤1 h)<br><br>
<b>4. Prescriptive fixes</b> - the what-if optimiser proposes the smallest change (discount, budget, stock, launch date, channel) that turns AMBER/RED into GREEN.
</div>""", unsafe_allow_html=True)

    st.markdown("#### 🧭 The analytics journey - from hindsight to action")
    stages = [("Descriptive", "What happened?", ["Volumes, budgets, channels", "Approval timelines", "Success & ROI rates"], "EDA page · every dashboard's KPI row"),
              ("Diagnostic", "Why did it happen?", ["Correlation heat-map", "Discount → margin erosion", "Delay → lost profit", "K-Means archetypes"], "EDA page · Marketing/Finance/Supply diagnostics"),
              ("Predictive", "What will happen?", ["KNN, Decision Tree, Random Forest, Gradient Boosting", "ARIMA demand & request-volume forecasts"], "Admin model tab · Supply forecast tab"),
              ("Prescriptive", "What should we do?", ["GREEN / AMBER / RED lanes", "Parallel routing", "What-if levers", "Reorder quantities"], "Marketing request form · Admin queue · C-Suite value")]
    cols = st.columns(4)
    for col, (name, q, items, where), colr in zip(cols, stages, [SEQ_BLUE[1], SEQ_BLUE[2], SEQ_BLUE[3], SEQ_BLUE[5]]):
        col.markdown(f"<div class='stage' style='border-top:6px solid {colr}'><div class='q'>{q}</div><div class='t'>{name}</div><ul>"
                     + "".join(f"<li>{i}</li>" for i in items) + f"</ul><div class='flag' style='margin-top:8px'>📍 {where}</div></div>",
                     unsafe_allow_html=True)
    st.markdown("")
    st.markdown("#### 👥 Who uses what")
    st.dataframe(pd.DataFrame({
        "Workspace": ["🛠️ Admin Portal", "📣 Marketing", "🚚 Supply Chain & Inventory", "💰 Finance", "🏛️ C-Suite"],
        "Main question it answers": ["Is the system healthy, accurate and correctly tuned?", "Can my campaign go - and if not, what do I change?",
                                     "Will we have the stock and capacity for upcoming campaigns?", "Will this make money and do we have the budget?",
                                     "Is the system making us faster and more profitable?"],
        "Key analytics": ["Model comparison (4 algorithms), thresholds, queue, data quality, request-volume ARIMA",
                          "Instant decision + what-if levers, channel/type performance, K-Means archetypes",
                          "ARIMA demand forecast, ARIMA-adjusted stock cover, SKU reorder list",
                          "ROI & margin erosion diagnostics, cost of delay, pending commitments, revenue forecast",
                          "Cycle-time & loss-avoidance impact, lane mix, strategic insights"]}), hide_index=True)


def page_eda():
    title("EDA - Descriptive & Diagnostic Analytics", "What happened in 2024-2026, and why. Every chart has a plain-English reading underneath.")
    t1, t2, t3 = st.tabs(["📊 Descriptive - what happened", "🔬 Diagnostic - why it happened", "🧹 Data quality & dictionary"])
    with t1:
        c = st.columns(4)
        kpi(c[0], "Campaigns", f"{len(H):,}", f"{HL.shape[0]:,} with outcome")
        kpi(c[1], "Media budget", inr(H["budget_inr"].sum()), f"median {inr(H['budget_inr'].median())}")
        kpi(c[2], "Incremental profit", inr(HL["actual_incremental_profit_inr"].sum()), f"{HL['finance_ok'].mean():.0%} profitable",
            "good" if HL["actual_incremental_profit_inr"].sum() > 0 else "bad")
        kpi(c[3], "Avg approval", f"{H['total_approval_days'].mean():.1f} d", f"vs {H['lead_time_to_launch_days'].mean():.0f} d avg lead time")
        a, b = st.columns(2)
        with a:
            m = H.set_index("submit_date").resample("MS")["campaign_id"].count().reset_index()
            fig = px.line(m, x="submit_date", y="campaign_id", markers=True, title="Campaign requests per month",
                          labels={"submit_date": "", "campaign_id": "Requests"})
            fig.update_traces(line=dict(width=2, color=BLUE), marker=dict(size=7))
            show(fig, 330)
            peak = m.assign(mo=m.submit_date.dt.month).groupby("mo")["campaign_id"].mean().idxmax()
            insight(f"Requests peak every year in <b>{date(2000, peak, 1):%B}</b> - teams pile up festive (Dussehra/Diwali) campaigns. "
                    "This is exactly when a 13-day sequential approval hurts most, so the system must scale with seasonal load.")
        with b:
            ch = HL.groupby("channel").agg(n=("campaign_id", "count")).reset_index().sort_values("n")
            fig = px.bar(ch, x="n", y="channel", orientation="h", title="Campaigns by channel", labels={"n": "Campaigns", "channel": ""}, text="n")
            fig.update_traces(marker_color=BLUE, textposition="outside")
            show(fig, 330)
            insight(f"<b>{ch.iloc[-1]['channel']}</b> and <b>{ch.iloc[-2]['channel']}</b> carry the most campaigns, TV the fewest "
                    "(but the biggest budgets). Channel mix tells us where approval effort is concentrated.")
        a, b = st.columns(2)
        with a:
            fig = px.box(H, x="channel", y="budget_inr", log_y=True, points=False, title="Budget per campaign by channel (log scale)",
                         labels={"budget_inr": "Budget (₹, log scale)", "channel": ""})
            fig.update_traces(marker_color=BLUE, line_color=BLUE, fillcolor="rgba(42,120,214,.15)")
            show(fig, 340)
            insight("Budgets are <b>right-skewed</b> (a few very large TV campaigns), which is why we log-transform budget before modelling. "
                    "Box = middle 50% of campaigns, line = median.")
        with b:
            fig = px.histogram(H, x="total_approval_days", nbins=40, title="How long manual approvals took",
                               labels={"total_approval_days": "Approval days (submission → final sign-off)"})
            fig.update_traces(marker_color=BLUE, marker_line_color="white", marker_line_width=1)
            med = H["lead_time_to_launch_days"].median()
            fig.add_vline(x=med, line_dash="dash", line_color=ORANGE, annotation_text=f"median lead time {med:.0f} d", annotation_font_color=INK_2)
            fig.update_layout(yaxis_title="Campaigns")
            show(fig, 340)
            insight(f"Approvals average <b>{H['total_approval_days'].mean():.1f} days</b> with a long right tail. "
                    f"<b>{H['approval_sla_breach'].mean():.0%}</b> of campaigns were still waiting for sign-off on their planned launch date.")
        st.markdown("##### Summary statistics (key numeric fields)")
        cols = ["budget_inr", "discount_pct", "duration_days", "expected_uplift_pct", "inventory_coverage_ratio", "supplier_lead_time_days",
                "warehouse_utilization_pct", "projected_roi", "total_approval_days", "actual_roi", "fill_rate_pct"]
        st.dataframe(H[cols].describe().T.round(2), height=420)

    with t2:
        num = ["discount_pct", "log_budget", "duration_days", "lead_time_to_launch_days", "brand_fit_score", "expected_uplift_pct",
               "inventory_coverage_ratio", "supplier_lead_gap_days", "warehouse_utilization_pct", "logistics_capacity_available_pct",
               "gross_margin_after_disc_pct", "projected_roi", "total_approval_days", "launch_delay_days",
               "fill_rate_pct", "actual_roi", "supply_ok", "finance_ok", "campaign_success"]
        corr = HL[num].corr().round(2)
        fig = px.imshow(corr, color_continuous_scale=DIVERGING, zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                        title="Correlation heat-map (Pearson r) - blue = move together, red = move opposite")
        fig.update_traces(textfont_size=9, hovertemplate="%{y} vs %{x}: r = %{z}<extra></extra>")
        fig.update_layout(coloraxis_colorbar=dict(title="r"))
        show(fig, 640)
        top = corr["campaign_success"].drop(["campaign_success", "supply_ok", "finance_ok", "actual_roi", "fill_rate_pct"]).sort_values()
        insight(f"Strongest drivers of success: <b>{top.index[-1]}</b> (r = {top.iloc[-1]:+.2f}) and <b>{top.index[-2]}</b> (r = {top.iloc[-2]:+.2f}) push it up; "
                f"<b>{top.index[0]}</b> (r = {top.iloc[0]:+.2f}) and <b>{top.index[1]}</b> (r = {top.iloc[1]:+.2f}) pull it down. "
                f"Discount vs margin r = {corr.loc['discount_pct', 'gross_margin_after_disc_pct']:+.2f} (margin erosion); "
                f"inventory cover vs fill-rate r = {corr.loc['inventory_coverage_ratio', 'fill_rate_pct']:+.2f}. "
                "Correlation only shows association - the classification models in the Admin tab test these jointly.")
        a, b = st.columns(2)
        with a:
            s = HL.groupby("campaign_type")["campaign_success"].mean().sort_values().reset_index()
            fig = px.bar(s, x="campaign_success", y="campaign_type", orientation="h", title="Success rate by campaign type",
                         labels={"campaign_success": "Share delivered & profitable", "campaign_type": ""}, text_auto=".0%")
            fig.update_traces(marker_color=BLUE)
            fig.update_layout(xaxis_tickformat=".0%")
            show(fig, 340)
            insight(f"<b>{s.iloc[-1]['campaign_type']}</b> succeed most often ({s.iloc[-1]['campaign_success']:.0%}); "
                    f"<b>{s.iloc[0]['campaign_type']}</b> least ({s.iloc[0]['campaign_success']:.0%}) - deep promotions give margin to customers who would have bought anyway.")
        with b:
            bands = pd.cut(HL["discount_pct"], [-1, 5, 15, 25, 35, 60], labels=["0-5%", "6-15%", "16-25%", "26-35%", ">35%"])
            s = HL.groupby(bands, observed=True).agg(roi=("actual_roi", "median"), n=("campaign_id", "count")).reset_index()
            fig = px.bar(s, x="discount_pct", y="roi", title="Median actual ROI by discount band", text_auto=".2f",
                         labels={"discount_pct": "Discount band", "roi": "Median ROI (x budget)"}, hover_data={"n": True})
            fig.update_traces(marker_color=[BLUE if v >= 0 else ORANGE for v in s["roi"]])
            show(fig, 340)
            insight("ROI <b>peaks at modest 6-15% discounts</b> and turns <b>negative beyond ~25%</b>; zero-discount campaigns are mostly awareness plays "
                    "with slow payback. Orange bars = the median campaign loses money. This is the diagnostic basis for the 'reduce discount' lever.")
        a, b = st.columns(2)
        with a:
            x = HL.assign(replen=np.where(HL["supplier_lead_gap_days"] > 0, "Replenishment arrives AFTER launch", "Replenishment arrives before launch"))
            fig = px.scatter(x, x="inventory_coverage_ratio", y="fill_rate_pct", color="replen", opacity=0.55,
                             color_discrete_sequence=[ORANGE, BLUE], title="Stock cover vs. fill rate",
                             labels={"inventory_coverage_ratio": "Inventory cover (on-hand ÷ projected units)", "fill_rate_pct": "Fill rate %", "replen": ""})
            fig.update_traces(marker=dict(size=7, line=dict(width=1, color="white")))
            fig.add_hline(y=95, line_dash="dash", line_color=INK_2, annotation_text="95% in-full target", annotation_font_color=INK_2)
            show(fig, 380)
            insight("Below ~0.8 cover, fill-rate collapses - <b>especially when supplier lead time is longer than the time to launch</b> (orange), "
                    "because no replenishment can arrive in time. Supply Chain needs both numbers, which is why the engine checks them together.")
        with b:
            x = HL.assign(timing=np.where(HL["launched_late"] == 1, "Launched late", "Launched on time"))
            fig = px.box(x, x="cluster", y="actual_roi", color="timing", points=False, color_discrete_map={"Launched late": ORANGE, "Launched on time": BLUE},
                         title="Actual ROI: on-time vs late launches (by archetype)", labels={"actual_roi": "Actual ROI (x)", "cluster": "", "timing": ""})
            fig.update_yaxes(range=[-6, 10])
            show(fig, 380)
            gap = HL.groupby("launched_late")["actual_incremental_profit_inr"].mean()
            insight(f"Within every campaign archetype, late launches earn less. On average a late campaign made <b>{inr(gap.get(0, 0) - gap.get(1, 0))} less</b> "
                    "incremental profit than an on-time one - the cost of slow approvals, quantified.")
        a, b = st.columns(2)
        with a:
            wb = pd.cut(HL["warehouse_utilization_pct"], [0, 70, 80, 85, 90, 95, 100])
            s = HL.groupby(wb, observed=True)["on_time_delivery"].mean().reset_index()
            s["band"] = s["warehouse_utilization_pct"].astype(str)
            fig = px.bar(s, x="band", y="on_time_delivery", text_auto=".0%", title="On-time delivery by warehouse utilisation",
                         labels={"band": "Warehouse utilisation %", "on_time_delivery": "On-time delivery rate"})
            fig.update_traces(marker_color=BLUE)
            fig.update_layout(yaxis_tickformat=".0%")
            show(fig, 330)
            insight("Delivery reliability holds until warehouses pass ~90% utilisation, then drops sharply - a non-linear effect that "
                    "tree-based models capture better than a single correlation.")
        with b:
            x = HL[["projected_units", "actual_units_sold", "channel"]].copy()
            x["achieved"] = HL["actual_units_sold"] / HL["projected_units"]
            s = x.groupby("channel")["achieved"].median().sort_values().reset_index()
            fig = px.bar(s, x="achieved", y="channel", orientation="h", text_auto=".0%", title="Units sold as % of Marketing's projection",
                         labels={"achieved": "Median actual ÷ projected units", "channel": ""})
            fig.update_traces(marker_color=BLUE)
            fig.add_vline(x=1, line_dash="dash", line_color=INK_2)
            fig.update_layout(xaxis_tickformat=".0%")
            show(fig, 330)
            insight("Projections are <b>systematically optimistic</b> (most channels sell below 100% of plan). "
                    "The Marketing model learns this bias, so P(demand target met) is a reality-check on the plan.")
    with t3:
        st.markdown("##### Cleaning & transformation log (Step 2)")
        st.dataframe(D["log"], hide_index=True, height=420)
        insight("Each rule is logged with the number of rows it touched - duplicates removed, mixed date formats parsed, 278 channel spellings "
                "standardised, currency text converted, impossible values (e.g. 150% discount) nulled and imputed with group medians.")
        st.markdown("##### Columns in the analysis-ready dataset")
        dd = pd.DataFrame({"column": hist_all.columns, "dtype": hist_all.dtypes.astype(str).values,
                           "example": [str(hist_all[c].iloc[0])[:30] for c in hist_all.columns]})
        st.dataframe(dd, hide_index=True, height=360)


def page_admin():
    title("🛠️ Admin Portal", "Run the decision system: live queue, policy thresholds, model governance and data quality.")
    q = engine(D["pend"], use_arima=True)
    rep_rows = hist_all[hist_all.campaign_id.isin(D["replay"].campaign_id)]
    r = engine(rep_rows)
    c = st.columns(5)
    kpi(c[0], "Pending requests", f"{len(q)}", "submitted 26 Oct - 8 Nov 2026")
    kpi(c[1], "Auto-approved", f"{(q.lane == 'GREEN').mean():.0%}", "decided in < 1 hour", "good")
    kpi(c[2], "Need a human", f"{(q.lane == 'AMBER').mean():.0%}", "only flagged team reviews")
    kpi(c[3], "Sent back to redesign", f"{(q.lane == 'RED').mean():.0%}", "with prescriptive fixes")
    kpi(c[4], "Active GO model", st.session_state.choice["campaign_success"].replace("Gradient Boosted Trees", "Gradient Boosting"),
        f"test ROC-AUC {D['metrics'].set_index(['target', 'model']).loc[('campaign_success', st.session_state.choice['campaign_success']), 'test_roc_auc']:.3f}")
    st.markdown("")
    t1, t2, t3, t4 = st.tabs(["📥 Request queue", "🎚️ Policy & thresholds", "🤖 Model performance", "🧹 Data quality, capacity & audit"])
    with t1:
        a, b = st.columns([1, 2])
        with a:
            show(lane_bar(q, "Live queue by decision lane"), 300)
            insight("Lanes come from the GO model plus guard-rails. Only AMBER items create human work - and only for the team named in 'Routed to'.")
        with b:
            queue_table(q, "admin_q")
        order = q.assign(o=q.lane.map({"AMBER": 0, "RED": 1, "GREEN": 2})).sort_values(["o", "p_go"], ascending=[True, False])
        pick = st.selectbox("Inspect a request (AMBER first)", order["campaign_id"],
                            format_func=lambda c: f"{c} · {lane_label(q.set_index('campaign_id').loc[c, 'lane'])}", key="admin_pick")
        row = q[q.campaign_id == pick].iloc[0]
        decision_card(row, row, key="admin")
    with t2:
        p = st.session_state.policy
        st.markdown("Tune the guard-rails. Changes apply instantly to every dashboard (this session).")
        a, b, c_ = st.columns(3)
        p["green_prob"] = a.slider("Auto-approve if P(GO) ≥", 0.40, 0.95, float(p["green_prob"]), 0.01)
        p["red_prob"] = a.slider("Send back if P(GO) <", 0.05, float(p["green_prob"]) - 0.01, min(float(p["red_prob"]), float(p["green_prob"]) - 0.01), 0.01)
        p["dept_prob"] = a.slider("Pull a department in if its P(pass) <", 0.2, 0.9, float(p["dept_prob"]), 0.05)
        p["min_coverage"] = b.slider("Minimum stock cover", 0.3, 1.5, float(p["min_coverage"]), 0.05)
        p["max_wh_util"] = b.slider("Max warehouse utilisation %", 80.0, 100.0, float(p["max_wh_util"]), 0.5)
        p["min_roi"] = c_.slider("Minimum projected ROI (x)", -1.0, 2.0, float(p["min_roi"]), 0.1)
        p["min_headroom"] = c_.slider("Minimum budget headroom %", 0.0, 30.0, float(p["min_headroom"]), 1.0)
        p["min_brand_fit"] = c_.slider("Minimum brand-fit score", 1, 9, int(p["min_brand_fit"]))
        st.markdown("**Active algorithm per check**")
        cols = st.columns(4)
        for col, t in zip(cols, TARGETS):
            st.session_state.choice[t] = col.selectbox(TARGETS[t], ALGOS, index=ALGOS.index(st.session_state.choice[t]), key=f"m_{t}")
        if st.button("Reset to recommended defaults"):
            st.session_state.policy = dict(DEFAULT_POLICY)
            st.session_state.choice = {t: B["targets"][t]["best"] for t in TARGETS}
            st.rerun()
        st.markdown("##### Back-test: what this policy would have done on 587 held-out historical campaigns")
        s = r.groupby("lane").agg(campaigns=("campaign_id", "count"), success=("campaign_success", "mean"),
                                  profit=("actual_incremental_profit_inr", "sum")).reindex(["GREEN", "AMBER", "RED"]).dropna().reset_index()
        a, b = st.columns(2)
        with a:
            s["label"] = s["lane"].map(lambda x: f"{LANE_ICON[x]} {x}")
            fig = px.bar(s, x="label", y="success", color="lane", color_discrete_map=STATUS, text_auto=".0%",
                         title="Share that actually succeeded, by lane", labels={"success": "Actual success rate", "label": ""})
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1])
            show(fig, 330)
        with b:
            fig = px.bar(s, x="label", y="profit", color="lane", color_discrete_map=STATUS, title="Actual incremental profit, by lane",
                         labels={"profit": "₹", "label": ""}, text=s["profit"].map(inr))
            fig.update_layout(showlegend=False)
            show(fig, 330)
        insight(f"A good policy separates winners from losers: GREEN campaigns succeeded "
                f"<b>{s.set_index('lane')['success'].get('GREEN', np.nan):.0%}</b> of the time, RED only "
                f"<b>{s.set_index('lane')['success'].get('RED', np.nan):.0%}</b>. Raising the GREEN threshold makes auto-approval safer but sends more work to humans - "
                "watch both charts as you move the sliders.")
    with t3:
        tgt = st.selectbox("Check (target variable)", list(TARGETS), format_func=lambda t: TARGETS[t])
        m = D["metrics"][D["metrics"].target == tgt]
        a, b = st.columns([1.2, 1])
        with a:
            mm = m.melt(id_vars="model", value_vars=["test_roc_auc", "accuracy", "precision", "recall", "f1"], var_name="metric")
            mm["metric"] = mm["metric"].map({"test_roc_auc": "ROC-AUC", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"})
            fig = px.bar(mm, x="metric", y="value", color="model", barmode="group", color_discrete_sequence=SERIES[:4],
                         title="Hold-out test performance - 4 algorithms", labels={"value": "", "metric": ""}, category_orders={"model": ALGOS})
            fig.update_layout(yaxis_range=[0.5, 1])
            show(fig, 380)
        with b:
            rc = D["roc"][D["roc"].target == tgt]
            fig = px.line(rc, x="fpr", y="tpr", color="model", color_discrete_sequence=SERIES[:4], category_orders={"model": ALGOS},
                          title="ROC curves", labels={"fpr": "False-positive rate", "tpr": "True-positive rate"})
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#B9BEC7"))
            fig.update_traces(line=dict(width=2))
            show(fig, 380)
        st.dataframe(m[["model", "cv_roc_auc", "test_roc_auc", "accuracy", "precision", "recall", "f1", "best_params"]].round(3), hide_index=True)
        best = B["targets"][tgt]["best"]
        insight(f"<b>{best}</b> wins on this check (highest hold-out ROC-AUC). CV and test AUC are close, so the models are not over-fitting. "
                "KNN trails because distance in 17 mixed features is noisy; a single Decision Tree is the most explainable but least accurate tree model; "
                "Random Forest and Gradient Boosting average/boost many trees and capture non-linear thresholds (e.g. warehouse > 90%).")
        a, b = st.columns(2)
        with a:
            mdl = st.selectbox("Confusion matrix for", ALGOS, index=ALGOS.index(best))
            cm = D["cm"][(D["cm"].target == tgt) & (D["cm"].model == mdl)].iloc[0]
            z = [[cm.tn, cm.fp], [cm.fn, cm.tp]]
            fig = px.imshow(z, x=["Predicted FAIL", "Predicted PASS"], y=["Actual FAIL", "Actual PASS"], text_auto=True,
                            color_continuous_scale=SEQ_BLUE, title=f"Confusion matrix - {mdl}")
            fig.update_layout(coloraxis_showscale=False)
            show(fig, 340)
            insight(f"Of {cm.fp + cm.tn} campaigns that actually failed, the model wrongly passed <b>{cm.fp}</b> (false positives = risk of approving a loser); "
                    f"of {cm.tp + cm.fn} that succeeded it wrongly failed <b>{cm.fn}</b> (missed opportunities). Lanes + guard-rails reduce the cost of both.")
        with b:
            im = D["imp"][D["imp"].target == tgt].head(12).sort_values("importance")
            fig = px.bar(im, x="importance", y="feature", orientation="h", error_x="std", title=f"Top drivers (permutation importance, {best})",
                         labels={"importance": "Drop in ROC-AUC when shuffled", "feature": ""})
            fig.update_traces(marker_color=BLUE)
            show(fig, 340)
            insight("Permutation importance asks: how much worse does the model get if we scramble this feature? Large bars = features the decision really relies on.")
        with st.expander("Human-readable rules from a 3-level Decision Tree (overall GO)"):
            st.code(D["rules"])
    with t4:
        a, b = st.columns([1.3, 1])
        with a:
            v, f = D["vol"], D["fc"][D["fc"].product_category == "Campaign requests"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=v.week_start, y=v.requests, name="Actual requests / week", line=dict(color=BLUE, width=2)))
            fig.add_trace(go.Scatter(x=pd.concat([f.week_start, f.week_start[::-1]]), y=pd.concat([f.upper_95, f.lower_95[::-1]]),
                                     fill="toself", fillcolor="rgba(235,104,52,.15)", line=dict(width=0), name="95% interval", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=f.week_start, y=f.forecast_units, name="ARIMA forecast", line=dict(color=ORANGE, width=2, dash="dash")))
            fig.update_layout(title="Request volume - ARIMA capacity forecast (next 12 weeks)", yaxis_title="Requests per week", hovermode="x unified")
            show(fig, 360)
            ar = D["arsel"].set_index("series").loc["Campaign requests"]
            insight(f"ARIMA{ar['chosen_order']} with yearly Fourier terms forecasts ~<b>{f.forecast_units.mean():.0f} requests/week</b> over the next quarter. "
                    "Admins use this to size the AMBER review rota: with ~30% AMBER, that is the weekly human-review load.")
        with b:
            st.markdown("##### Data-quality checks")
            st.dataframe(D["log"][["rule", "column", "rows_affected"]], hide_index=True, height=360)
        st.markdown("##### Decision audit trail (live queue)")
        rng = np.random.default_rng(1)
        audit = q[["campaign_id", "submitted_by", "submit_date", "lane", "route_to"]].copy()
        audit["decided_at"] = audit["submit_date"] + pd.to_timedelta(rng.integers(1, 50, len(audit)), unit="s")
        audit["decided_by"] = np.where(audit.lane == "AMBER", "Engine → " + audit.route_to, "Engine (auto)")
        audit["model_version"] = "bundle-2026.11 · " + st.session_state.choice["campaign_success"]
        audit["lane"] = audit["lane"].map(lane_label)
        st.dataframe(audit.sort_values("decided_at", ascending=False), hide_index=True, height=300)


def page_marketing():
    title("📣 Marketing Workspace", "Submit a campaign and get a decision in seconds - with the exact changes that would make it approvable.")
    t1, t2, t3, t4 = st.tabs(["🚀 New campaign request", "📈 What works (performance)", "🧩 Campaign archetypes (K-Means)", "📋 My pending requests"])
    with t1:
        with st.form("req"):
            st.markdown("**Campaign plan**")
            c1, c2, c3, c4 = st.columns(4)
            region = c1.selectbox("Region", list(REGION_SHARE))
            category = c2.selectbox("Product category", sorted(hist_all.product_category.unique()))
            channel = c3.selectbox("Channel", sorted(hist_all.channel.unique()), index=5)
            ctype = c4.selectbox("Campaign type", sorted(hist_all.campaign_type.unique()), index=5)
            launch = c1.date_input("Planned launch date", TODAY + timedelta(days=21), min_value=TODAY + timedelta(days=3))
            duration = c2.slider("Duration (days)", 7, 60, 21)
            budget = c3.number_input("Media budget (₹)", 10000, 5000000, 250000, step=10000)
            discount = c4.slider("Discount %", 0, 60, 20)
            uplift = c1.slider("Expected sales uplift %", 0, 200, 70)
            brand = c2.slider("Brand-fit score (1-10)", 1, 10, 7)
            comp = c3.select_slider("Competitor intensity", ["Low", "Medium", "High"], "Medium")
            with st.expander("🔗 Auto-filled from Supply-Chain & ERP systems (edit only if you know better)"):
                e1, e2, e3, e4, e5 = st.columns(5)
                stock_pct = e1.slider("Stock allocated (% of projected units)", 30, 150, 95)
                lead = e2.number_input("Supplier lead time (days)", 0, 90, None, placeholder="auto: category median")
                wh = e3.number_input("Warehouse utilisation %", 0.0, 100.0, None, placeholder="auto: region median")
                logi = e4.number_input("Logistics capacity free %", 0.0, 100.0, None, placeholder="auto: 32%")
                head = e5.number_input("Budget headroom %", 0.0, 100.0, None, placeholder="auto: Finance ledger")
            go_btn = st.form_submit_button("⚡ Get instant decision", type="primary")
        if go_btn:
            t0 = time.perf_counter()
            hc = hist_all[hist_all.product_category == category]
            lw = pd.Timestamp(launch)
            f = D["fc"][(D["fc"].product_category == category) & (D["fc"].week_start >= lw - pd.Timedelta(days=6))
                        & (D["fc"].week_start <= lw + pd.Timedelta(days=duration))]
            base_weekly = (f.forecast_units.mean() if len(f) else D["sales"][D["sales"].product_category == category].units_sold.tail(8).mean()) * REGION_SHARE[region]
            cpr = hist_all[hist_all.channel == channel]["cost_per_reach_inr"].median()
            row = pd.DataFrame([dict(
                campaign_id="NEW-REQUEST", region=region, product_category=category, channel=channel, campaign_type=ctype,
                planned_launch_date=lw, submit_date=pd.Timestamp(TODAY), duration_days=duration,
                lead_time_to_launch_days=(launch - TODAY).days, festive_window=int(lw.month in {3, 8, 10, 11}),
                budget_inr=float(budget), discount_pct=discount, target_reach=int(budget / cpr),
                channel_hist_conv_rate_pct=hist_all[hist_all.channel == channel]["channel_hist_conv_rate_pct"].median(),
                brand_fit_score=brand, competitor_intensity=comp, unit_price_inr=hc.unit_price_inr.median(),
                unit_cost_inr=hc.unit_cost_inr.median(), baseline_weekly_units=int(base_weekly), expected_uplift_pct=float(uplift),
                inventory_on_hand_units=0.0,
                supplier_lead_time_days=lead if lead is not None else hc.supplier_lead_time_days.median(),
                warehouse_utilization_pct=wh if wh is not None else hist_all[hist_all.region == region].warehouse_utilization_pct.median(),
                logistics_capacity_available_pct=logi if logi is not None else 32.0,
                dept_budget_headroom_pct=head if head is not None else 30.0)])
            row = add_features(row)
            row["inventory_on_hand_units"] = round(row["projected_units"].iloc[0] * stock_pct / 100)
            row = add_features(row)
            res = run_engine(row, tuple(sorted(st.session_state.policy.items())), tuple(sorted(st.session_state.choice.items())), True)
            st.session_state.last_req = (res.iloc[0], time.perf_counter() - t0)
        if "last_req" in st.session_state:
            res, secs = st.session_state.last_req
            st.caption(f"Decision computed in {secs:.2f} s (vs. {H['total_approval_days'].mean():.1f} days manually). "
                       f"Baseline demand {res['baseline_weekly_units']:,} units/week from the ARIMA forecast · projected {res['projected_units']:,} units.")
            decision_card(res, res, key="mkt")
        else:
            insight("Fill in the plan and press <b>Get instant decision</b>. Try a 40% BOGO on TV to see a RED decision and its fixes, "
                    "or a 10% Loyalty campaign on Email to see an auto-approval.")
    with t2:
        s = HL.groupby("channel").agg(success=("campaign_success", "mean"), roi=("actual_roi", "median"), n=("campaign_id", "count")).reset_index()
        a, b = st.columns(2)
        with a:
            fig = px.bar(s.sort_values("success"), x="success", y="channel", orientation="h", text_auto=".0%", title="Success rate by channel",
                         labels={"success": "Delivered & profitable", "channel": ""})
            fig.update_traces(marker_color=BLUE)
            fig.update_layout(xaxis_tickformat=".0%")
            show(fig, 340)
        with b:
            fig = px.bar(s.sort_values("roi"), x="roi", y="channel", orientation="h", text_auto=".2f", title="Median actual ROI by channel",
                         labels={"roi": "Median ROI (x budget)", "channel": ""})
            fig.update_traces(marker_color=[BLUE if v >= 0 else ORANGE for v in s.sort_values("roi")["roi"]])
            show(fig, 340)
        insight(f"<b>{s.sort_values('roi').iloc[-1]['channel']}</b> delivers the best return per rupee, <b>{s.sort_values('roi').iloc[0]['channel']}</b> the worst "
                "(high fixed cost per regional campaign). Low-cost, high-intent channels win when the goal is profitable volume.")
        hm = HL.pivot_table(index="campaign_type", columns="channel", values="campaign_success", aggfunc="mean")
        fig = px.imshow(hm, text_auto=".0%", color_continuous_scale=SEQ_BLUE, zmin=0, zmax=1, aspect="auto",
                        title="Success rate - campaign type × channel (darker = more successful)")
        fig.update_layout(coloraxis_colorbar=dict(title="Success", tickformat=".0%"))
        show(fig, 380)
        insight("Use this as a design cheat-sheet: pick the darkest cells. BOGO and deep discounts are pale almost everywhere; "
                "Loyalty and Product Launch campaigns succeed across most channels.")
        x = HL.sample(min(900, len(HL)), random_state=1)
        fig = px.scatter(x, x="projected_units", y="actual_units_sold", color="outcome", color_discrete_map=OUTCOME, opacity=0.6, log_x=True, log_y=True,
                         title="Projected vs actual units (log-log)", labels={"projected_units": "Projected units", "actual_units_sold": "Actual units sold", "outcome": ""})
        lim = [x.projected_units.min(), x.projected_units.max()]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="#8A8F99"), name="perfect forecast"))
        fig.update_traces(marker=dict(size=7, line=dict(width=1, color="white")), selector=dict(mode="markers"))
        show(fig, 400)
        insight("Points below the dashed line sold less than planned. Most failures (orange) sit below it - over-optimistic uplift is a key failure mode.")
    with t3:
        cp = D["cprof"]
        a, b = st.columns([1.3, 1])
        with a:
            x = hist_all.sample(1500, random_state=2)
            fig = px.scatter(x, x="pc1", y="pc2", color="cluster", symbol="cluster", color_discrete_map=CLUSTER_COLORS, symbol_map=CLUSTER_SYMBOLS,
                             opacity=0.65, title="Campaign archetypes (K-Means, k = 4) - PCA projection",
                             labels={"pc1": "Principal component 1", "pc2": "Principal component 2", "cluster": ""},
                             hover_data={"campaign_id": True, "budget_inr": ":,", "discount_pct": True, "pc1": False, "pc2": False})
            fig.update_traces(marker=dict(size=7, line=dict(width=1, color="white")))
            show(fig, 430)
        with b:
            s = cp.sort_values("success_rate")
            fig = px.bar(s, x="success_rate", y="cluster", orientation="h", color="cluster", color_discrete_map=CLUSTER_COLORS, text_auto=".0%",
                         title="Historical success rate by archetype", labels={"success_rate": "Delivered & profitable", "cluster": ""})
            fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
            show(fig, 430)
        show_cp = cp.rename(columns={"cluster": "Archetype", "campaigns": "Campaigns", "success_rate": "Success", "median_budget": "Median budget ₹",
                                     "avg_discount": "Avg discount %", "avg_coverage": "Avg stock cover", "avg_margin": "Avg margin %",
                                     "avg_actual_roi": "Median ROI", "avg_approval_days": "Avg approval days"})
        st.dataframe(show_cp.drop(columns=["supply_ok_rate", "finance_ok_rate"]).round(2), hide_index=True)
        insight("K-Means groups campaigns by <b>design</b> (budget, discount, ROI, cover, margin, timing, duration, brand-fit) - no outcome is used. "
                "Yet the groups differ sharply in success: <b>Deep-Discount Volume Pushers</b> rarely make money and should default to Finance review, "
                "while <b>Lean Quick-Wins</b> and <b>Long-Run Loyalty & Launch Plays</b> are natural fast-track candidates. "
                "k = 4 was chosen from the elbow and silhouette curves (Admin → model tab) and for business readability.")
        a, b = st.columns(2)
        with a:
            fig = px.line(D["ksel"], x="k", y="inertia", markers=True, title="Elbow curve (within-cluster SSE)")
            fig.update_traces(line_color=BLUE, marker_size=8)
            fig.add_vline(x=4, line_dash="dash", line_color=ORANGE)
            show(fig, 280)
        with b:
            fig = px.line(D["ksel"], x="k", y="silhouette", markers=True, title="Silhouette score")
            fig.update_traces(line_color=BLUE, marker_size=8)
            fig.add_vline(x=4, line_dash="dash", line_color=ORANGE)
            show(fig, 280)
    with t4:
        q = engine(D["pend"], use_arima=True)
        who = st.selectbox("Submitted by", ["All"] + sorted(q.submitted_by.unique()))
        qq = q if who == "All" else q[q.submitted_by == who]
        queue_table(qq, "mkt_q")
        order = qq.assign(o=qq.lane.map({"AMBER": 0, "RED": 1, "GREEN": 2})).sort_values(["o", "p_go"], ascending=[True, False])
        pick = st.selectbox("Open a request for its fixes", order["campaign_id"],
                            format_func=lambda c: f"{c} · {lane_label(q.set_index('campaign_id').loc[c, 'lane'])}", key="mkt_pick")
        row = q[q.campaign_id == pick].iloc[0]
        decision_card(row, row, key="mkt2")


def page_supply():
    title("🚚 Supply Chain, Logistics & Inventory", "Will we have the stock and capacity to deliver upcoming campaigns in full and on time?")
    inv, q = D["inv"], engine(D["pend"], use_arima=True)
    c = st.columns(5)
    kpi(c[0], "SKUs critical", f"{(inv.stock_status == 'Critical').sum()}", "below safety stock", "bad")
    kpi(c[1], "SKUs to reorder", f"{(inv.stock_status == 'Reorder').sum()}", "below reorder point")
    kpi(c[2], "Median weeks of cover", f"{inv.weeks_of_cover.median():.1f}", "across 200 SKUs")
    kpi(c[3], "Historical fill rate", f"{HL.fill_rate_pct.median():.0f}%", f"{(HL.fill_rate_pct < 95).mean():.0%} campaigns short-shipped")
    kpi(c[4], "Pending with stock risk", f"{(q.arima_cov.fillna(q.inventory_coverage_ratio) < st.session_state.policy['min_coverage']).sum()}",
        "ARIMA-adjusted cover below floor", "bad")
    st.markdown("")
    t1, t2, t3, t4 = st.tabs(["📈 Demand forecast (ARIMA)", "📦 Upcoming campaign supply check", "🏷️ Inventory health", "🔬 Supply diagnostics"])
    with t1:
        cat = st.radio("Category", sorted(D["sales"].product_category.unique()), horizontal=True)
        s = D["sales"][D["sales"].product_category == cat]
        f = D["fc"][D["fc"].product_category == cat]
        h = D["hold"][D["hold"].product_category == cat]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.week_start, y=s.units_sold, name="Actual weekly units", line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=h.week_start, y=h.holdout_forecast, name="Back-test forecast (12-wk hold-out)", line=dict(color=AQUA, width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=pd.concat([f.week_start, f.week_start[::-1]]), y=pd.concat([f.upper_95, f.lower_95[::-1]]), fill="toself",
                                 fillcolor="rgba(235,104,52,.15)", line=dict(width=0), name="95% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=f.week_start, y=f.forecast_units, name="ARIMA forecast (16 wk)", line=dict(color=ORANGE, width=2.5)))
        fig.update_layout(title=f"{cat} - weekly demand and ARIMA forecast", yaxis_title="Units / week", hovermode="x unified")
        show(fig, 430)
        sel = D["arsel"].set_index("series").loc[cat]
        a, b = st.columns([1, 1.4])
        with a:
            st.dataframe(pd.DataFrame({"": ["ADF p-value (level)", "Differencing d", "Plain ARIMA order", "Plain ARIMA hold-out MAPE",
                                            "ARIMA + Fourier order", "ARIMA + Fourier MAPE", "Chosen"],
                                       "value": [f"{sel.adf_p_level:.3f}", str(int(sel.d)), sel.arima_order, f"{sel.arima_mape:.1f}%",
                                                 sel.fourier_order, f"{sel.fourier_mape:.1f}%", sel.chosen]}), hide_index=True)
        with b:
            insight(f"<b>How it was built:</b> the ADF test (p = {sel.adf_p_level:.3f}) decides whether the series needs differencing (d = {int(sel.d)}). "
                    f"A grid search over p, q ∈ 0-3 picks the order with the lowest AIC. Weekly data has a 52-week festive cycle that plain ARIMA cannot see, "
                    f"so we add yearly Fourier terms as regressors - hold-out error drops from <b>{sel.arima_mape:.1f}%</b> to <b>{sel.fourier_mape:.1f}%</b>. "
                    f"Next-16-week average: <b>{f.forecast_units.mean():,.0f} units/week</b> (Diwali peak included).")
    with t2:
        x = q.assign(cov_eff=q.arima_cov.fillna(q.inventory_coverage_ratio))
        demand_eff = x["inventory_on_hand_units"] / x["cov_eff"].clip(lower=0.01)      # demand implied by the effective cover
        x["need_units"] = np.clip(st.session_state.policy["min_coverage"] * demand_eff - x["inventory_on_hand_units"], 0, None).round()
        fig = px.scatter(x, x="inventory_coverage_ratio", y="cov_eff", color="lane", color_discrete_map=STATUS, symbol="lane",
                         symbol_map={"GREEN": "circle", "AMBER": "diamond", "RED": "x"}, size="projected_units", size_max=26,
                         hover_data={"campaign_id": True, "product_category": True, "region": True, "planned_launch_date": True},
                         title="Pending campaigns: Marketing's stock cover vs ARIMA-adjusted cover",
                         labels={"inventory_coverage_ratio": "Cover vs Marketing's projection", "cov_eff": "Cover vs ARIMA-based demand", "lane": "Lane"})
        floor = st.session_state.policy["min_coverage"]
        fig.add_hline(y=floor, line_dash="dash", line_color=INK_2, annotation_text=f"minimum cover {floor:.0%}", annotation_font_color=INK_2)
        fig.add_shape(type="line", x0=0.3, y0=0.3, x1=2, y1=2, line=dict(dash="dot", color="#B9BEC7"))
        show(fig, 430)
        insight("Points <b>below the diagonal</b> are campaigns where the ARIMA forecast (with trend + festive season) expects more demand than Marketing's "
                "baseline - their real stock cover is lower than it looks. Points under the dashed line need extra stock before launch.")
        need = x[x.cov_eff < floor].sort_values("cov_eff")
        st.markdown(f"##### 🧭 Prescriptive: pre-position stock for {len(need)} at-risk campaigns")
        st.dataframe(need[["campaign_id", "product_category", "region", "planned_launch_date", "inventory_on_hand_units", "cov_eff",
                           "need_units", "supplier_lead_time_days", "lead_time_to_launch_days", "lane"]].assign(lane=lambda d: d.lane.map(lane_label)),
                     hide_index=True, column_config={"cov_eff": st.column_config.NumberColumn("ARIMA cover", format="percent"),
                                                     "need_units": st.column_config.NumberColumn("Extra units to order", format="%d"),
                                                     "planned_launch_date": st.column_config.DateColumn("Launch", format="DD MMM")})
        insight("Extra units = quantity that lifts ARIMA-adjusted cover to the policy floor. If supplier lead time is longer than the days to launch, "
                "the order cannot arrive in time - those rows need a later launch date or a stock transfer from another region.")
    with t3:
        a, b = st.columns(2)
        with a:
            s = inv.groupby(["product_category", "stock_status"]).size().reset_index(name="SKUs")
            fig = px.bar(s, x="product_category", y="SKUs", color="stock_status", title="SKU stock status by category",
                         color_discrete_map={"Healthy": STATUS["GREEN"], "Reorder": STATUS["AMBER"], "Critical": STATUS["RED"]},
                         category_orders={"stock_status": ["Healthy", "Reorder", "Critical"]}, labels={"product_category": "", "stock_status": ""})
            fig.update_traces(marker_line_color="white", marker_line_width=2)
            show(fig, 360)
            insight("Status = on-hand vs safety stock and reorder point (ROP = weekly demand × lead time + safety stock). "
                    "Red SKUs should not be promoted until replenished.")
        with b:
            fig = px.histogram(inv, x="weeks_of_cover", nbins=30, title="Weeks of cover across SKUs", labels={"weeks_of_cover": "Weeks of cover"})
            fig.update_traces(marker_color=BLUE, marker_line_color="white", marker_line_width=1)
            fig.update_layout(yaxis_title="SKUs")
            show(fig, 360)
            insight("Most SKUs carry 1-4 weeks of stock. A typical 3-week campaign with +60% uplift needs ~5 weeks of normal demand - "
                    "so campaigns almost always need dedicated stock allocation.")
        st.markdown("##### Reorder list")
        st.dataframe(inv[inv.stock_status != "Healthy"].sort_values("weeks_of_cover")[
            ["sku_id", "product_category", "region", "warehouse", "on_hand_units", "safety_stock", "reorder_point", "weeks_of_cover",
             "supplier_lead_time_days", "suggested_order_qty", "stock_status"]], hide_index=True, height=320)
    with t4:
        a, b = st.columns(2)
        with a:
            g = HL.assign(gap=pd.cut(HL.supplier_lead_gap_days, [-60, -14, -7, 0, 7, 14, 60],
                                     labels=["≥14 d early", "7-13 d early", "0-6 d early", "1-7 d late", "8-14 d late", ">14 d late"]))
            s = g.groupby("gap", observed=True)["supply_ok"].mean().reset_index()
            fig = px.bar(s, x="gap", y="supply_ok", text_auto=".0%", title="Supply check pass-rate vs. replenishment timing",
                         labels={"gap": "Supplier delivery relative to launch", "supply_ok": "In-full & on-time rate"})
            fig.update_traces(marker_color=BLUE)
            fig.update_layout(yaxis_tickformat=".0%")
            show(fig, 350)
            insight("When replenishment lands after launch, the campaign must live on opening stock - pass-rate falls. "
                    "This is why 'push launch back 7-14 days' is a lever the engine can recommend.")
        with b:
            s = HL.groupby("product_category").agg(fill=("fill_rate_pct", "median"), late=("on_time_delivery", lambda v: 1 - v.mean())).reset_index()
            fig = px.bar(s.sort_values("late"), x="late", y="product_category", orientation="h", text_auto=".1%",
                         title="Late-delivery rate by category", labels={"late": "Share delivered late", "product_category": ""})
            fig.update_traces(marker_color=ORANGE)
            fig.update_layout(xaxis_tickformat=".0%")
            show(fig, 350)
            insight("Late-delivery risk is driven mostly by warehouse congestion and free logistics capacity rather than by category - "
                    "a cross-category capacity view is needed during festive peaks.")


def page_finance():
    title("💰 Finance", "Is each campaign worth the money, and what is slow approval costing us?")
    q = engine(D["pend"], use_arima=True)
    late_gap = HL.groupby("launched_late")["actual_incremental_profit_inr"].mean()
    delay_cost = (late_gap.get(0, 0) - late_gap.get(1, 0)) * HL.launched_late.sum()
    c = st.columns(5)
    kpi(c[0], "Incremental profit", inr(HL.actual_incremental_profit_inr.sum()), "all completed campaigns",
        "good" if HL.actual_incremental_profit_inr.sum() > 0 else "bad")
    kpi(c[1], "Profitable campaigns", f"{HL.finance_ok.mean():.0%}", f"median ROI {HL.actual_roi.median():.2f}x")
    kpi(c[2], "Money-losing campaigns", inr(HL.loc[HL.actual_incremental_profit_inr < 0, "actual_incremental_profit_inr"].sum()),
        f"{(HL.actual_incremental_profit_inr < 0).sum():,} campaigns", "bad")
    kpi(c[3], "Estimated cost of delay", inr(delay_cost), f"{HL.launched_late.sum():,} late launches", "bad")
    kpi(c[4], "Pending budget", inr(q.budget_inr.sum()), f"{inr(q[q.lane == 'RED'].budget_inr.sum())} in RED lane")
    st.markdown("")
    t1, t2, t3, t4 = st.tabs(["📊 Profitability", "⏱️ Cost of delay", "🧾 Pending commitments", "🔮 Revenue forecast"])
    with t1:
        a, b = st.columns(2)
        with a:
            s = HL.groupby("channel")["actual_incremental_profit_inr"].sum().sort_values().reset_index()
            fig = px.bar(s, x="actual_incremental_profit_inr", y="channel", orientation="h", title="Total incremental profit by channel",
                         labels={"actual_incremental_profit_inr": "₹", "channel": ""}, text=s.actual_incremental_profit_inr.map(inr))
            fig.update_traces(marker_color=[BLUE if v >= 0 else ORANGE for v in s.actual_incremental_profit_inr])
            show(fig, 360)
            insight("Orange = channel destroyed value in total. The portfolio is carried by a few channels; the rest need tighter guard-rails.")
        with b:
            fig = px.scatter(HL.sample(min(1000, len(HL)), random_state=3), x="discount_pct", y="gross_margin_after_disc_pct", color="outcome",
                             color_discrete_map=OUTCOME, opacity=0.55, title="Margin erosion: discount vs margin after discount",
                             labels={"discount_pct": "Discount %", "gross_margin_after_disc_pct": "Gross margin after discount %", "outcome": ""})
            fig.update_traces(marker=dict(size=7, line=dict(width=1, color="white")))
            show(fig, 360)
            r_ = HL[["discount_pct", "gross_margin_after_disc_pct"]].corr().iloc[0, 1]
            insight(f"Each extra discount point removes margin almost one-for-one (r = {r_:.2f}). Below ~30% margin, failures (orange) dominate - "
                    "Finance's guard-rail should watch margin after discount, not just headline revenue.")
        x = HL.assign(proj=HL.projected_roi.clip(-5, 10), act=HL.actual_roi.clip(-8, 12)).sample(min(1000, len(HL)), random_state=4)
        fig = px.scatter(x, x="proj", y="act", color="outcome", color_discrete_map=OUTCOME, opacity=0.55, title="Projected ROI (at submission) vs actual ROI",
                         labels={"proj": "Projected ROI (x)", "act": "Actual ROI (x)", "outcome": ""})
        fig.add_trace(go.Scatter(x=[-5, 10], y=[-5, 10], mode="lines", line=dict(dash="dash", color="#8A8F99"), name="projection = actual"))
        fig.update_traces(marker=dict(size=7, line=dict(width=1, color="white")), selector=dict(mode="markers"))
        show(fig, 400)
        insight("Most points sit below the dashed line: business cases over-state ROI. The Finance model learns the real relationship, "
                "so P(profitable) is a better approval signal than the spreadsheet ROI alone.")
    with t2:
        a, b = st.columns(2)
        with a:
            g = HL.assign(band=pd.cut(HL.launch_delay_days, [-0.1, 0, 3, 7, 14, 60], labels=["On time", "1-3 d", "4-7 d", "8-14 d", ">14 d"]))
            s = g.groupby("band", observed=True).agg(roi=("actual_roi", "median"), n=("campaign_id", "count")).reset_index()
            fig = px.bar(s, x="band", y="roi", text_auto=".2f", title="Median actual ROI by launch delay", hover_data={"n": True},
                         labels={"band": "Launch delay caused by approvals", "roi": "Median ROI (x)"})
            fig.update_traces(marker_color=[BLUE if v >= 0 else ORANGE for v in s.roi])
            show(fig, 360)
        with b:
            s = HL.groupby(["festive_window", "launched_late"])["actual_incremental_profit_inr"].mean().reset_index()
            s["window"] = s.festive_window.map({0: "Non-festive", 1: "Festive window"})
            s["timing"] = s.launched_late.map({0: "On time", 1: "Late"})
            fig = px.bar(s, x="window", y="actual_incremental_profit_inr", color="timing", barmode="group",
                         color_discrete_map={"On time": BLUE, "Late": ORANGE}, title="Avg profit per campaign: festive vs non-festive",
                         labels={"actual_incremental_profit_inr": "₹ per campaign", "window": "", "timing": ""})
            show(fig, 360)
        insight(f"Every extra day of approval delay erodes return, and missing a festive window is the most expensive. Historically "
                f"{HL.launched_late.mean():.0%} of campaigns launched late, costing an estimated <b>{inr(delay_cost)}</b> in profit "
                "(difference in average profit × number of late launches). The new system decides in hours, so this leakage disappears.")
    with t3:
        s = q.groupby("lane").agg(budget=("budget_inr", "sum"), n=("campaign_id", "count"), exp_profit=("projected_incremental_profit_inr", "sum")).reset_index()
        s["label"] = s.lane.map(lambda v: f"{LANE_ICON[v]} {v}")
        a, b = st.columns([1, 1.6])
        with a:
            fig = px.bar(s, x="label", y="budget", color="lane", color_discrete_map=STATUS, text=s.budget.map(inr),
                         title="Budget requested in the live queue, by lane", labels={"budget": "₹", "label": ""},
                         category_orders={"label": [f"{LANE_ICON[v]} {v}" for v in ["GREEN", "AMBER", "RED"]]})
            fig.update_layout(showlegend=False)
            show(fig, 340)
        with b:
            fq = q[q.route_to.str.contains("Finance")].sort_values("p_finance")
            st.markdown(f"**{len(fq)} requests need Finance's eyes** (Finance flag or P(profitable) below threshold)")
            st.dataframe(fq[["campaign_id", "channel", "campaign_type", "budget_inr", "discount_pct", "projected_roi", "p_finance", "dept_budget_headroom_pct", "flags"]],
                         hide_index=True, height=300, column_config={"p_finance": st.column_config.ProgressColumn("P(profitable)", format="percent", min_value=0, max_value=1),
                                                                      "budget_inr": st.column_config.NumberColumn("Budget ₹", format="%d"),
                                                                      "projected_roi": st.column_config.NumberColumn("Proj. ROI", format="%.2fx")})
        insight("Finance no longer reviews every campaign - only those the model or guard-rails flag. GREEN budget is released automatically; "
                "RED budget is held until Marketing applies a fix.")
    with t4:
        price = hist_all.groupby("product_category").unit_price_inr.median()
        f = D["fc"][D["fc"].product_category.isin(price.index)].copy()
        f["revenue"] = f.forecast_units * f.product_category.map(price)
        f["lo"], f["hi"] = f.lower_95 * f.product_category.map(price), f.upper_95 * f.product_category.map(price)
        fig = px.line(f, x="week_start", y="revenue", color="product_category", color_discrete_sequence=SERIES[:5], markers=True,
                      title="ARIMA revenue outlook - next 16 weeks (forecast units × list price)",
                      labels={"week_start": "", "revenue": "₹ per week", "product_category": ""})
        fig.update_traces(line=dict(width=2), marker=dict(size=6))
        fig.update_layout(hovermode="x unified")
        show(fig, 420)
        tot = f.groupby("week_start").revenue.sum()
        insight(f"Baseline (pre-campaign) revenue is forecast at <b>{inr(tot.sum())}</b> over 16 weeks, peaking in the Diwali weeks at "
                f"<b>{inr(tot.max())}</b>/week. Finance uses this to set quarterly campaign budgets in proportion to the demand they can lift.")


def page_csuite():
    title("🏛️ C-Suite View", "Is the new decision system making us faster and more profitable?")
    rep = hist_all[hist_all.campaign_id.isin(D["replay"].campaign_id)]
    r = engine(rep)
    r["new_days"] = r.lane.map(NEW_DAYS)
    old_days, new_days = r.total_approval_days.mean(), r.new_days.mean()
    loss_avoided = -r.loc[(r.lane == "RED") & (r.actual_incremental_profit_inr < 0), "actual_incremental_profit_inr"].sum()
    missed = r.loc[(r.lane == "RED") & (r.actual_incremental_profit_inr > 0), "actual_incremental_profit_inr"].sum()
    green_succ = r.loc[r.lane == "GREEN", "campaign_success"].mean()
    c = st.columns(5)
    kpi(c[0], "Approval cycle time", f"{new_days * 24:.0f} h", f"from {old_days:.1f} days (-{1 - new_days / old_days:.0%})", "good")
    kpi(c[1], "Late launches", "≈ 0%", f"from {r.launched_late.mean():.0%} - decisions beat every lead time", "good")
    kpi(c[2], "Losses caught before launch", inr(loss_avoided), "RED-lane campaigns that lost money", "good")
    kpi(c[3], "Auto-approved success rate", f"{green_succ:.0%}", f"vs {r.campaign_success.mean():.0%} for all approved today", "good")
    kpi(c[4], "Human reviews needed", f"{(r.lane == 'AMBER').mean():.0%}", "of requests (was 100% × 3 teams)")
    st.caption("Impact is back-tested on 587 historical campaigns the models never saw during training (25% hold-out).")
    st.markdown("")
    a, b = st.columns(2)
    with a:
        s = pd.DataFrame({"process": ["Manual (sequential)", "Engine: GREEN", "Engine: AMBER", "Engine: RED"],
                          "days": [old_days, NEW_DAYS["GREEN"], NEW_DAYS["AMBER"], NEW_DAYS["RED"]]})
        fig = px.bar(s, x="days", y="process", orientation="h", text=s.days.map(lambda d: f"{d:.1f} d" if d >= 1 else f"{d * 24:.0f} h"),
                     title="Time to decision", labels={"days": "Days", "process": ""})
        fig.update_traces(marker_color=[ORANGE, STATUS["GREEN"], STATUS["AMBER"], STATUS["RED"]], textposition="outside")
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_range=[0, old_days * 1.18])
        show(fig, 330)
        insight("Parallel model-based checks replace three sequential sign-offs. Even the slowest lane (AMBER, 1 day) is faster than the shortest campaign lead time (7 days).")
    with b:
        v = pd.DataFrame({"scenario": ["Approve everything (as-is)", "Engine: GREEN + AMBER only"],
                          "profit": [r.actual_incremental_profit_inr.sum(), r.loc[r.lane != "RED", "actual_incremental_profit_inr"].sum()]})
        fig = px.bar(v, x="scenario", y="profit", text=v.profit.map(inr), title="Portfolio profit on the hold-out campaigns",
                     labels={"profit": "Incremental profit (₹)", "scenario": ""})
        fig.update_traces(marker_color=[ORANGE, BLUE])
        show(fig, 330)
        insight(f"Holding back RED campaigns (until they are redesigned) avoids <b>{inr(loss_avoided)}</b> of losses at the cost of "
                f"<b>{inr(missed)}</b> profit from the few RED campaigns that would have worked - a net gain of <b>{inr(loss_avoided - missed)}</b>, "
                "before counting the extra profit from fixing and relaunching them.")
    a, b = st.columns(2)
    with a:
        qt = HL.assign(q=HL.submit_date.dt.to_period("Q").astype(str)).groupby("q").agg(
            success=("campaign_success", "mean"), late=("launched_late", "mean")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qt.q, y=qt.success, name="Delivered & profitable", mode="lines+markers", line=dict(color=BLUE, width=2), marker_size=8))
        fig.add_trace(go.Scatter(x=qt.q, y=qt.late, name="Launched late", mode="lines+markers", line=dict(color=ORANGE, width=2), marker_size=8))
        fig.update_layout(title="Quarterly trend (as-is process)", yaxis_tickformat=".0%", hovermode="x unified", yaxis_title="Share of campaigns")
        show(fig, 340)
        insight("Under the manual process neither success nor punctuality improved over time - the problem is structural, not seasonal.")
    with b:
        q = engine(D["pend"], use_arima=True)
        show(lane_bar(q, "This fortnight's 40 requests"), 340)
        insight(f"{(q.lane == 'GREEN').sum()} requests go live immediately, {(q.lane == 'AMBER').sum()} need one targeted review, "
                f"{(q.lane == 'RED').sum()} go back with specific fixes - worth {inr(q[q.lane == 'RED'].budget_inr.sum())} of budget held for redesign.")
    st.markdown("#### 🎯 Strategic recommendations")
    best_ch = HL.groupby("channel").actual_roi.median().idxmax()
    worst_type = HL.groupby("campaign_type").campaign_success.mean().idxmin()
    st.markdown(f"""
<div class='card'>
<b>1. Make the engine the single front door for campaign approval.</b> Replace sequential sign-offs with parallel model checks; humans review only AMBER cases (≈{(r.lane == 'AMBER').mean():.0%}).<br><br>
<b>2. Cap discount depth.</b> ROI turns negative beyond ~25% discount and <b>{worst_type}</b> campaigns succeed least often - require Finance co-design for these.<br><br>
<b>3. Shift mix toward high-return channels.</b> <b>{best_ch}</b> has the best median ROI; large regional TV spends rarely pay back.<br><br>
<b>4. Plan festive stock with ARIMA, not last year's run-rate.</b> Forecasts include trend and Diwali seasonality, catching stock gaps weeks before launch.<br><br>
<b>5. Govern the models.</b> Re-train quarterly, monitor ROC-AUC in the Admin portal, and review thresholds each season.
</div>""", unsafe_allow_html=True)


{PAGES[0]: page_overview, PAGES[1]: page_eda, PAGES[2]: page_admin, PAGES[3]: page_marketing,
 PAGES[4]: page_supply, PAGES[5]: page_finance, PAGES[6]: page_csuite}[page]()
