"""Visual system for the dashboard: palette, Plotly template, CSS and small UI helpers."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---- palette (validated reference palette; categorical order is fixed, never cycled) ----
INK, INK_2, MUTED, GRID, SURFACE = "#1B2333", "#52514e", "#8A8F99", "#E6E9EF", "#FFFFFF"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW = SERIES[:4]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#c43b3a"], [0.25, "#ec9a99"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]
STATUS = {"GREEN": "#0ca30c", "AMBER": "#fab219", "RED": "#d03b3b"}
LANE_ICON = {"GREEN": "✅", "AMBER": "⚠️", "RED": "⛔"}
LANE_TEXT = {"GREEN": "Auto-approve", "AMBER": "Fast-track review", "RED": "Redesign required"}
OUTCOME = {"Success": BLUE, "Failure": ORANGE}
CLUSTER_COLORS = {"Lean Quick-Win Campaigns": BLUE, "Long-Run Loyalty & Launch Plays": AQUA,
                  "Big-Budget Low-Return Bets": YELLOW, "Deep-Discount Volume Pushers": ORANGE}
CLUSTER_SYMBOLS = {"Lean Quick-Win Campaigns": "circle", "Long-Run Loyalty & Launch Plays": "diamond",
                   "Big-Budget Low-Return Bets": "square", "Deep-Discount Volume Pushers": "x"}

pio.templates["aurora"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color=INK),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", xref="container", y=0.98, yanchor="top", yref="container", pad=dict(l=10)),
    xaxis=dict(automargin=True, gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, tickfont=dict(color=INK_2), title_font=dict(color=INK_2)),
    yaxis=dict(automargin=True, gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, tickfont=dict(color=INK_2), title_font=dict(color=INK_2)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK_2), title_text=""),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=12)),
    margin=dict(l=10, r=10, t=78, b=10), bargap=0.25,
))
pio.templates.default = "aurora"

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1350px;}
h1, h2, h3 {color: #1B2333; letter-spacing: -0.01em;}
.app-title {font-size: 1.9rem; font-weight: 750; color: #1B2333; margin-bottom: .1rem;}
.app-sub {color: #52514e; font-size: 1.02rem; margin-bottom: 1.1rem;}
.kpi {background: #FFFFFF; border: 1px solid #E3E8F0; border-radius: 14px; padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(16,24,40,.04); height: 100%;}
.kpi .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; color: #52514e; font-weight: 600;}
.kpi .value {font-size: 1.65rem; font-weight: 750; color: #1B2333; line-height: 1.25; margin-top: 2px;}
.kpi .note {font-size: .82rem; color: #52514e; margin-top: 2px;}
.kpi .good {color: #0a7f0a; font-weight: 650;} .kpi .bad {color: #b42f2f; font-weight: 650;}
.insight {background: #F0F6FF; border-left: 4px solid #2a78d6; border-radius: 8px; padding: 10px 14px;
          margin: 4px 0 18px 0; color: #1B2333; font-size: .93rem; line-height: 1.5;}
.insight b {color: #184f95;}
.lane {display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: .9rem;
       border: 1.5px solid; background: #fff;}
.lane-GREEN {color: #0a7f0a; border-color: #0ca30c; background: #EEF9EE;}
.lane-AMBER {color: #8a5a00; border-color: #fab219; background: #FFF7E6;}
.lane-RED   {color: #b42f2f; border-color: #d03b3b; background: #FDEEEE;}
.card {background: #FFFFFF; border: 1px solid #E3E8F0; border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; height: 100%;}
.card h4 {margin: 0 0 6px 0; font-size: 1.02rem;}
.card .big {font-size: 1.5rem; font-weight: 750;}
.flag {font-size: .88rem; color: #52514e; margin: 3px 0;}
.stage {border-radius: 14px; padding: 14px 16px; height: 100%; border: 1px solid #E3E8F0; background: #fff;}
.stage .q {font-weight: 700; color: #184f95; font-size: .95rem;}
.stage .t {font-size: 1.15rem; font-weight: 750; margin: 2px 0 6px 0;}
.stage ul {padding-left: 18px; margin: 0; font-size: .9rem; color: #33373d;}
section[data-testid="stSidebar"] .stRadio label p {font-size: .98rem;}
div[data-testid="stTabs"] button p {font-size: .95rem; font-weight: 600;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def inr(x, dec=1):
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e7:
        return f"{sign}₹{x / 1e7:.{dec}f} Cr"
    if x >= 1e5:
        return f"{sign}₹{x / 1e5:.{dec}f} L"
    return f"{sign}₹{x:,.0f}"


def kpi(col, label, value, note="", tone=None):
    cls = {"good": "good", "bad": "bad"}.get(tone, "")
    col.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div>"
                 f"<div class='note {cls}'>{note}</div></div>", unsafe_allow_html=True)


def insight(text):
    st.markdown(f"<div class='insight'>📖 {text}</div>", unsafe_allow_html=True)


def lane_badge(lane):
    return f"<span class='lane lane-{lane}'>{LANE_ICON[lane]} {lane} · {LANE_TEXT[lane]}</span>"


def lane_label(lane):
    return f"{LANE_ICON[lane]} {lane}"


def show(fig, height=None):
    # Streamlit's frontend can drop template backgrounds, so set the key styling explicitly on the layout
    fig.update_layout(template=pio.templates["aurora"], paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color=INK))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, automargin=True)
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, theme=None, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def title(text, sub=""):
    st.markdown(f"<div class='app-title'>{text}</div><div class='app-sub'>{sub}</div>", unsafe_allow_html=True)
