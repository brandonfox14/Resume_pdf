import streamlit as st
from pathlib import Path
import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

# PDF generation (no browser automation)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Visual Resume", layout="wide")

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
PDF_OUT = ROOT / "visual_resume.pdf"

# -----------------------------
# Helpers
# -----------------------------
def b64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def safe_logo_html(path: Path, height_px: int = 30) -> str:
    enc = b64(path)
    if not enc:
        return f'<div style="height:{height_px}px; width:{height_px}px; border:1px solid rgba(0,0,0,0.15); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:10px; opacity:0.6;">No logo</div>'
    return f'<img src="data:image/png;base64,{enc}" style="height:{height_px}px; width:auto;" />'

def rounded25(x: float) -> int:
    return int(round(x / 25.0) * 25)

def fig_to_png_bytes(fig, dpi=220) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 1.2rem; max-width: 1180px; }

      .name { font-size: 44px; font-weight: 850; line-height: 1.05; margin-bottom: 0.15rem; }
      .contact { font-size: 14px; opacity: 0.85; margin-bottom: 0.95rem; }

      .box {
        border: 1px solid rgba(49, 51, 63, 0.25);
        border-radius: 16px;
        padding: 14px 14px 10px 14px;
        background: rgba(255,255,255,0.02);
      }
      .box-title { font-size: 16px; font-weight: 760; margin-bottom: 0.55rem; }
      .subtle { font-size: 12px; opacity: 0.75; }

      .logo-row { display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
      .logo-item { display:flex; gap:10px; align-items:center; }
      .logo-item img { height:28px; width:auto; }

      .job-head { display:flex; gap:12px; align-items:center; margin-bottom: 8px; }
      .job-company { font-size: 15px; font-weight: 860; line-height: 1.1; }
      .job-title { font-size: 13px; font-weight: 650; opacity: 0.92; }
      .job-dates { font-size: 12px; opacity: 0.75; }

      ul { margin-top: 0.35rem; padding-left: 1.15rem; }
      li { margin-bottom: 0.18rem; font-size: 12.5px; }

      .pill {
        display:inline-block;
        border: 1px solid rgba(49, 51, 63, 0.25);
        border-radius: 999px;
        padding: 3px 10px;
        margin: 2px 6px 2px 0;
        font-size: 12px;
        opacity: 0.9;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header content
# -----------------------------
NAME = "Brandon Fox"
CONTACT = "Madison, WI • brandonfox14@icloud.com • (608) 516-9676"

st.markdown(f'<div class="name">{NAME}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="contact">{CONTACT}</div>', unsafe_allow_html=True)

# ============================================================
# EDUCATION
# ============================================================
st.markdown('<div class="box">', unsafe_allow_html=True)
st.markdown('<div class="box-title">Education</div>', unsafe_allow_html=True)

mich_logo = ASSETS / "schools" / "Mich.png"
isu_logo = ASSETS / "schools" / "isu.png"

edu_html = f"""
<div class="logo-row">
  <div class="logo-item">
    {safe_logo_html(mich_logo, 28)}
    <div>
      <div><b>University of Michigan</b> — M.S. Applied Data Science (MADS)</div>
      <div class="subtle">Predictive modeling • experimentation • production analytics</div>
    </div>
  </div>
  <div class="logo-item">
    {safe_logo_html(isu_logo, 28)}
    <div>
      <div><b>Illinois State University</b> — (Add degree / major here if desired)</div>
      <div class="subtle">Economics + regression analysis • stakeholder communication</div>
    </div>
  </div>
</div>
"""
st.markdown(edu_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ============================================================
# VIS 1: Verona half-court shot chart
# ============================================================
def draw_half_court(ax):
    ax.set_xlim(-25, 25)
    ax.set_ylim(0, 47)
    ax.set_aspect('equal')
    ax.axis("off")

    ax.add_patch(Rectangle((-25, 0), 50, 47, fill=False, linewidth=1.2))

    hoop_y = 5.25
    ax.add_patch(Circle((0, hoop_y), 0.75, fill=False, linewidth=1.2))
    ax.add_patch(Rectangle((-3, hoop_y-0.75), 6, 0.1, fill=True, linewidth=0))

    ax.add_patch(Rectangle((-8, 0), 16, 19, fill=False, linewidth=1.2))
    ax.add_patch(Arc((0, 19), 12, 12, theta1=0, theta2=180, linewidth=1.2))
    ax.add_patch(Arc((0, 19), 12, 12, theta1=180, theta2=360, linewidth=1.2, linestyle="--", alpha=0.6))
    ax.add_patch(Arc((0, hoop_y), 8, 8, theta1=0, theta2=180, linewidth=1.2))

    ax.plot([-22, -22], [0, 14], linewidth=1.2)
    ax.plot([22, 22], [0, 14], linewidth=1.2)
    ax.add_patch(Arc((0, hoop_y), 44, 44, theta1=22, theta2=158, linewidth=1.2))

def assign_zone(x, y):
    hoop_y = 5.25
    r = np.sqrt(x**2 + (y - hoop_y)**2)

    if y <= 14 and x <= -22:
        return 3
    if y <= 14 and x >= 22:
        return 4

    is_three = (r >= 22) and (y > 14)
    if is_three:
        return 5

    if r <= 4:
        return 0
    if (-8 <= x <= 8) and (0 <= y <= 19):
        return 1
    return 2

def verona_shot_fig(seed=42):
    rng = np.random.default_rng(seed)

    n_shots = 53
    made_total = 24
    n_threes = 11
    made_threes = 4
    n_twos = n_shots - n_threes
    made_twos = made_total - made_threes  # 20

    three_x, three_y = [], []
    for _ in range(n_threes):
        corner = rng.random() < 0.35
        if corner:
            side = -1 if rng.random() < 0.5 else 1
            x = side * (22 + rng.normal(0, 0.7))
            y = rng.uniform(2, 13.5)
        else:
            angle = rng.uniform(np.deg2rad(30), np.deg2rad(150))
            r = rng.normal(23.3, 0.8)
            hoop_y = 5.25
            x = r * np.cos(angle)
            y = hoop_y + r * np.sin(angle)
            y = np.clip(y, 14.5, 46)
        three_x.append(float(x))
        three_y.append(float(y))

    two_x, two_y = [], []
    for _ in range(n_twos):
        bucket = rng.choice(["rim", "paint", "mid"], p=[0.40, 0.25, 0.35])
        hoop_y = 5.25
        if bucket == "rim":
            angle = rng.uniform(np.deg2rad(20), np.deg2rad(160))
            r = rng.uniform(1.5, 6.0)
            x = r * np.cos(angle)
            y = hoop_y + r * np.sin(angle)
            y = np.clip(y, 1, 16)
        elif bucket == "paint":
            x = rng.uniform(-7.5, 7.5)
            y = rng.uniform(6, 18.5)
        else:
            x = rng.uniform(-20, 20)
            y = rng.uniform(16, 30)
        two_x.append(float(x))
        two_y.append(float(y))

    x = np.array(two_x + three_x)
    y = np.array(two_y + three_y)
    is_three = np.array([False]*n_twos + [True]*n_threes)

    made = np.array([False]*n_shots)
    three_idx = np.where(is_three)[0]
    two_idx = np.where(~is_three)[0]
    made_three_idx = rng.choice(three_idx, size=made_threes, replace=False)
    made_two_idx = rng.choice(two_idx, size=made_twos, replace=False)
    made[made_three_idx] = True
    made[made_two_idx] = True

    zones = np.array([assign_zone(xi, yi) for xi, yi in zip(x, y)])
    zone_pct = {}
    for z in [0, 1, 2, 3, 4, 5]:
        idx = zones == z
        att = idx.sum()
        pct = None if att == 0 else made[idx].mean()
        zone_pct[z] = (att, pct)

    pcts = [(z, zone_pct[z][1]) for z in zone_pct if zone_pct[z][0] >= 3 and zone_pct[z][1] is not None]
    hot_zones, cold_zones = set(), set()
    if len(pcts) >= 2:
        pcts_sorted = sorted(pcts, key=lambda t: t[1])
        cold_zones.add(pcts_sorted[0][0])
        hot_zones.add(pcts_sorted[-1][0])

    fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=220)
    draw_half_court(ax)

    if 0 in hot_zones or 1 in hot_zones:
        ax.add_patch(Rectangle((-8, 0), 16, 19, alpha=0.12, linewidth=0))
    if 0 in cold_zones or 1 in cold_zones:
        ax.add_patch(Rectangle((-8, 0), 16, 19, alpha=0.08, linewidth=0, hatch=".."))

    if 2 in hot_zones:
        ax.add_patch(Rectangle((-20, 16), 40, 15, alpha=0.10, linewidth=0))
    if 2 in cold_zones:
        ax.add_patch(Rectangle((-20, 16), 40, 15, alpha=0.07, linewidth=0, hatch=".."))

    if 3 in hot_zones:
        ax.add_patch(Rectangle((-25, 0), 3, 14, alpha=0.12, linewidth=0))
    if 3 in cold_zones:
        ax.add_patch(Rectangle((-25, 0), 3, 14, alpha=0.07, linewidth=0, hatch=".."))
    if 4 in hot_zones:
        ax.add_patch(Rectangle((22, 0), 3, 14, alpha=0.12, linewidth=0))
    if 4 in cold_zones:
        ax.add_patch(Rectangle((22, 0), 3, 14, alpha=0.07, linewidth=0, hatch=".."))

    if 5 in hot_zones:
        ax.add_patch(Rectangle((-25, 14), 50, 33, alpha=0.08, linewidth=0))
    if 5 in cold_zones:
        ax.add_patch(Rectangle((-25, 14), 50, 33, alpha=0.06, linewidth=0, hatch=".."))

    ax.scatter(x[~made], y[~made], s=18, marker="x", linewidths=1.0, alpha=0.9)
    ax.scatter(x[made], y[made], s=20, marker="o", alpha=0.9)

    fg = made.mean()
    three_fg = made[is_three].mean()
    ax.set_title(
        f"Verona HS Shot Chart (53 FGA) — FG {made.sum()}/53 ({fg:.0%}), 3PT {made[is_three].sum()}/{n_threes} ({three_fg:.0%})",
        fontsize=9
    )

    fig.tight_layout()
    return fig

# ============================================================
# VIS 2: March Metrics growth chart (AUC + ACC)
# ============================================================
def march_metrics_growth(seed=7):
    rng = np.random.default_rng(seed)
    years = np.array([2021, 2022, 2023, 2024, 2025, 2026])

    auc_2021, acc_2021 = 0.61, 0.58
    auc_2026, acc_2026 = 0.851, 0.792

    t = np.linspace(0, 1, len(years))
    auc = auc_2021 + (auc_2026 - auc_2021) * (t**1.15)
    acc = acc_2021 + (acc_2026 - acc_2021) * (t**1.10)

    auc += rng.normal(0, 0.008, size=len(years))
    acc += rng.normal(0, 0.008, size=len(years))

    auc[0], acc[0] = auc_2021, acc_2021
    auc[-1], acc[-1] = auc_2026, acc_2026

    for i in range(1, len(years)):
        auc[i] = max(auc[i], auc[i-1] + 0.01)
        acc[i] = max(acc[i], acc[i-1] + 0.01)

    auc[-1], acc[-1] = auc_2026, acc_2026

    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=220)
    ax.plot(years, auc, marker="o", linewidth=2.0, label="Avg AUC (Win+Spread+OU)")
    ax.plot(years, acc, marker="o", linewidth=2.0, label="Avg ACC (Win+Spread+OU)")
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(years)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_title("March Metrics — Model Quality Growth (Simulated trend, endpoints fixed)", fontsize=10)
    ax.set_xlabel("Season")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig, pd.DataFrame({"Season": years, "Avg AUC": auc, "Avg ACC": acc})

# ============================================================
# VIS 3: Sushi Primos 4-location comparison (rounded to 25)
# ============================================================
def sushi_locations(seed=12):
    rng = np.random.default_rng(seed)

    current = dict(
        Location="Current (Rounded)",
        Traffic=rounded25(475),
        FootTraffic=rounded25(625),
        Customers=rounded25(125),
        ProfitPreRent=rounded25(40000),
        Rent=rounded25(24500),
        Residential=rounded25(3500),  # placeholder baseline (rounded) for demo
        SqFt=rounded25(1575),
    )

    goals = dict(
        Traffic=1000,
        FootTraffic=250,
        Residential=3500,
        Rent=30000,
        SqFt_min=1750,
        SqFt_max=3250
    )

    def predict_customers(traffic, foot):
        base = (traffic / current["Traffic"]) * 0.35 + (foot / current["FootTraffic"]) * 0.65
        pred = current["Customers"] * base
        pred *= rng.uniform(0.95, 1.05)
        return rounded25(pred)

    def profit_pre_rent(customers):
        ppc = rng.uniform(320, 390)
        prof = customers * ppc
        return rounded25(prof)

    # Option A (Best Fit)
    A_traffic = rounded25(rng.uniform(1025, 1400))
    A_foot = rounded25(rng.uniform(275, 520))
    A_res = rounded25(rng.uniform(3750, 5200))
    A_sqft = rounded25(rng.uniform(1850, 3100))
    A_rent = rounded25(rng.uniform(22000, 29500))
    A_cust = predict_customers(A_traffic, A_foot)
    A_profit = profit_pre_rent(A_cust)

    # Option B (Too Big / Expensive; lower profit than best)
    B_traffic = rounded25(rng.uniform(1050, 1350))
    B_foot = rounded25(rng.uniform(260, 480))
    B_res = rounded25(rng.uniform(3600, 4800))
    B_sqft = rounded25(rng.uniform(3400, 4200))  # too big
    B_rent = rounded25(rng.uniform(34000, 45000))  # too expensive
    B_cust = predict_customers(B_traffic, B_foot)
    B_profit = rounded25(rng.uniform(A_profit * 0.78, A_profit * 0.92))

    # Option C (Low Residential / High Foot)
    C_traffic = rounded25(rng.uniform(950, 1250))
    C_foot = rounded25(rng.uniform(700, 1050))
    C_res = rounded25(rng.uniform(1200, 2400))  # too few people nearby
    C_sqft = rounded25(rng.uniform(1900, 3100))
    C_rent = rounded25(rng.uniform(24000, 31000))
    C_cust = predict_customers(C_traffic, C_foot)
    C_profit = profit_pre_rent(C_cust)

    # Option D (Slightly low across; cheapest rent)
    D_traffic = rounded25(rng.uniform(775, 975))
    D_foot = rounded25(rng.uniform(200, 245))
    D_res = rounded25(rng.uniform(3000, 3450))
    D_sqft = rounded25(rng.uniform(1750, 2000))
    D_rent = rounded25(rng.uniform(16000, 21000))
    D_cust = predict_customers(D_traffic, D_foot)
    D_profit = profit_pre_rent(D_cust)

    df = pd.DataFrame([
        current,
        dict(Location="Option A (Best Fit)", Traffic=A_traffic, FootTraffic=A_foot, Customers=A_cust,
             ProfitPreRent=A_profit, Rent=A_rent, Residential=A_res, SqFt=A_sqft),
        dict(Location="Option B (Too Big / Expensive)", Traffic=B_traffic, FootTraffic=B_foot, Customers=B_cust,
             ProfitPreRent=B_profit, Rent=B_rent, Residential=B_res, SqFt=B_sqft),
        dict(Location="Option C (Low Residential / High Foot)", Traffic=C_traffic, FootTraffic=C_foot, Customers=C_cust,
             ProfitPreRent=C_profit, Rent=C_rent, Residential=C_res, SqFt=C_sqft),
        dict(Location="Option D (Low Across / Cheapest Rent)", Traffic=D_traffic, FootTraffic=D_foot, Customers=D_cust,
             ProfitPreRent=D_profit, Rent=D_rent, Residential=D_res, SqFt=D_sqft),
    ])

    df["ProfitMinusRent"] = df["ProfitPreRent"] - df["Rent"]
    checks = {
        "Traffic": df["Traffic"] >= goals["Traffic"],
        "FootTraffic": df["FootTraffic"] >= goals["FootTraffic"],
        "Residential": df["Residential"] >= goals["Residential"],
        "Rent": df["Rent"] < goals["Rent"],
        "SqFt": df["SqFt"].between(goals["SqFt_min"], goals["SqFt_max"]),
    }
    for k, v in checks.items():
        df[f"Meets_{k}"] = v

    return df, goals

def sushi_fig(df):
    tmp = df[df["Location"] != "Current (Rounded)"].copy()
    goal_cols = ["Meets_Traffic", "Meets_FootTraffic", "Meets_Residential", "Meets_Rent", "Meets_SqFt"]
    tmp["GoalsMet"] = tmp[goal_cols].sum(axis=1)

    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=220)
    ax.scatter(tmp["GoalsMet"], tmp["ProfitMinusRent"])
    for _, r in tmp.iterrows():
        ax.annotate(r["Location"].replace("Option ", ""), (r["GoalsMet"], r["ProfitMinusRent"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Goals Met (out of 5)")
    ax.set_ylabel("Avg Monthly Profit − Rent (Rounded)")
    ax.set_title("Sushi Primos — Location Tradeoffs (Rounded + Slightly Altered)", fontsize=10)
    fig.tight_layout()
    return fig

# ============================================================
# VIS 4: ISU retention synthetic demo charts
# ============================================================
def retention_data():
    majors = ["Education", "STEM", "Undecided", "Other"]
    groups = [
        ("Female", "In-State"),
        ("Male", "In-State"),
        ("Female", "Out-of-State"),
        ("Male", "Out-of-State"),
    ]
    data = {
        ("Female", "In-State"):     [93, 90, 72, 84],
        ("Male", "In-State"):       [91, 92, 70, 82],
        ("Female", "Out-of-State"): [86, 84, 66, 78],
        ("Male", "Out-of-State"):   [83, 81, 60, 75],
    }
    df = pd.DataFrame(
        [{"Major": m, "Gender": g, "Residency": r, "Retention": data[(g, r)][i]}
         for i, m in enumerate(majors)
         for (g, r) in groups]
    )
    return df

def retention_figs(df):
    majors = ["Education", "STEM", "Undecided", "Other"]
    order = [("Female", "In-State"), ("Male", "In-State"), ("Female", "Out-of-State"), ("Male", "Out-of-State")]
    labels = ["F / In", "M / In", "F / Out", "M / Out"]

    figs = []
    for major in majors:
        sub = df[df["Major"] == major].copy()
        vals = [float(sub[(sub["Gender"] == g) & (sub["Residency"] == r)]["Retention"].iloc[0]) for g, r in order]

        fig, ax = plt.subplots(figsize=(3.2, 2.3), dpi=220)
        ax.bar(labels, vals)
        ax.set_ylim(50, 100)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_title(f"{major} (Freshman→Sophomore)", fontsize=9)
        ax.set_ylabel("Retention %")
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        fig.tight_layout()
        figs.append(fig)
    return figs

# ============================================================
# Job box renderer
# ============================================================
def job_box(col, company_name, job_title, dates, logo_path, pills, fig_or_figs, bullets, footnote=None):
    with col:
        st.markdown('<div class="box">', unsafe_allow_html=True)

        head_html = f"""
        <div class="job-head">
          {safe_logo_html(logo_path, 30)}
          <div>
            <div class="job-company">{company_name}</div>
            <div class="job-title">{job_title}</div>
            <div class="job-dates">{dates}</div>
          </div>
        </div>
        """
        st.markdown(head_html, unsafe_allow_html=True)

        if pills:
            st.markdown("".join([f'<span class="pill">{p}</span>' for p in pills]), unsafe_allow_html=True)

        if isinstance(fig_or_figs, list):
            cols = st.columns(len(fig_or_figs), gap="small")
            for c, f in zip(cols, fig_or_figs):
                with c:
                    st.pyplot(f, use_container_width=True)
        else:
            st.pyplot(fig_or_figs, use_container_width=True)

        bullets_html = "<ul>" + "".join([f"<li>{b}</li>" for b in bullets]) + "</ul>"
        st.markdown(bullets_html, unsafe_allow_html=True)

        if footnote:
            st.markdown(f'<div class="subtle">{footnote}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Build visuals
# ============================================================
verona_fig = verona_shot_fig(seed=42)
mm_fig, _ = march_metrics_growth(seed=7)

sushi_df, sushi_goals = sushi_locations(seed=12)
sushi_plot = sushi_fig(sushi_df)

ret_df = retention_data()
ret_figs = retention_figs(ret_df)

# ============================================================
# 2x2 job grid
# ============================================================
row1 = st.columns(2, gap="large")
row2 = st.columns(2, gap="large")

job_box(
    row1[0],
    "Verona Area High School",
    "JV Assistant Basketball Coach",
    "10/2024 to Present",
    ASSETS / "companies" / "verona.png",
    pills=["Player development", "Practice KPIs", "Shot quality"],
    fig_or_figs=verona_fig,
    bullets=[
        "Built practice + game tracking to quantify shot selection and efficiency.",
        "Used simple heat zones to communicate shot quality and decision-making.",
        "Standardized postgame review to drive repeatable improvement."
    ],
)

job_box(
    row1[1],
    "March Metrics",
    "Data Scientist",
    "08/2020 to Present",
    ASSETS / "companies" / "metrics.png",
    pills=["NCAA Hoops", "Model evaluation", "AUC/ACC tracking"],
    fig_or_figs=mm_fig,
    bullets=[
        "Moneyline/Win: AUC 0.979 • ACC 0.920 (2026 season).",
        "Spread: AUC 0.721 • ACC 0.671 (2026 season).",
        "Total (O/U): AUC 0.854 • ACC 0.783 (2026 season).",
        "Avg (Win+Spread+OU): AUC 0.851 • ACC 0.792 (2026 season).",
    ],
    footnote="Growth chart is simulated; 2021 and 2026 endpoints fixed to the stated averages."
)

job_box(
    row2[0],
    "Sushi Primos",
    "Data Science Consultant (Site Selection)",
    "2024",
    ASSETS / "companies" / "sushi.png",
    pills=["Location scoring", "Traffic + footfall", "Profit vs rent"],
    fig_or_figs=sushi_plot,
    bullets=[
        "Compared 4 candidate locations using rounded KPIs (traffic, foot traffic, customers, rent, residential, square footage).",
        "Balanced predicted demand with rent constraints and operational fit (executive decision support).",
        "Values are rounded and slightly altered to preserve confidentiality while maintaining ranking."
    ],
    footnote="All KPIs are rounded to nearest 25 and slightly altered for confidentiality."
)

job_box(
    row2[1],
    "Illinois State University",
    "Graduate Assistant — Data & Statistical Research (Personal Demo)",
    "2023–2024",
    ASSETS / "companies" / "isu.png",
    pills=["Synthetic demo", "Retention analysis", "Segmented insights"],
    fig_or_figs=ret_figs,
    bullets=[
        "Built a retention-rate segmentation demo by major, residency, and gender.",
        "Education majors lead retention; STEM close behind; undecided lowest.",
        "In-state advantage is consistent; female advantage is modest but present."
    ],
    footnote="Important: these charts use my own synthetic/demo data — not ISU data."
)

# ============================================================
# Export (NO Playwright)
# ============================================================
st.markdown("---")
st.markdown('<div class="box">', unsafe_allow_html=True)
st.markdown('<div class="box-title">Export</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtle">Two options: (1) open your browser print dialog to “Save as PDF”, or (2) generate a one-page PDF download built from these visuals.</div>',
    unsafe_allow_html=True
)

colA, colB = st.columns(2, gap="large")

# ---- Option A: Browser Print dialog
with colA:
    st.markdown("**Option A (fastest): Print / Save as PDF**")
    st.components.v1.html(
        """
        <button onclick="window.print()" style="
          padding:10px 14px; border-radius:10px; border:1px solid rgba(49,51,63,0.25);
          background:white; cursor:pointer; font-weight:600;">
          Print / Save as PDF
        </button>
        <div style="font-size:12px; opacity:0.7; margin-top:6px;">
          Choose “Save as PDF” in the print dialog.
        </div>
        """,
        height=80
    )

# ---- Option B: Generate a PDF file (ReportLab)
def build_pdf(path: Path):
    # Convert the 4 visuals into PNGs
    # Re-create figs so they are not closed by earlier rendering
    v_fig = verona_shot_fig(seed=42)
    m_fig, _ = march_metrics_growth(seed=7)
    s_fig = sushi_fig(sushi_df)
    r_figs = retention_figs(ret_df)

    v_png = fig_to_png_bytes(v_fig)
    m_png = fig_to_png_bytes(m_fig)
    s_png = fig_to_png_bytes(s_fig)

    # For retention, stitch 4 small charts into one horizontal strip image by placing into the PDF directly
    r_pngs = [fig_to_png_bytes(f) for f in r_figs]

    c = canvas.Canvas(str(path), pagesize=letter)
    W, H = letter

    # Margins
    margin = 36  # 0.5 inch
    x0 = margin
    y = H - margin

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(x0, y - 10, NAME)
    c.setFont("Helvetica", 10)
    c.drawString(x0, y - 28, CONTACT)
    y -= 48

    # Education line (simple text; logos optional here)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y, "Education")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(x0, y, "University of Michigan — M.S. Applied Data Science (MADS)")
    y -= 12
    c.drawString(x0, y, "Illinois State University — (degree/major)")
    y -= 18

    # Layout 2x2 images
    gap = 10
    box_w = (W - 2*margin - gap) / 2
    box_h = 210  # tuned for one page

    def draw_box_title(title, subtitle, x, y_top):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y_top - 14, title)
        c.setFont("Helvetica", 9)
        c.drawString(x, y_top - 28, subtitle)

    def draw_img(img_bytes, x, y_top, w, h):
        img = ImageReader(io.BytesIO(img_bytes))
        c.drawImage(img, x, y_top - h, width=w, height=h, preserveAspectRatio=True, mask='auto')

    # Row 1
    y_row1 = y
    x_left = x0
    x_right = x0 + box_w + gap

    draw_box_title("Verona Area HS — JV Assistant Coach", "10/2024–Present", x_left, y_row1)
    draw_img(v_png, x_left, y_row1 - 34, box_w, box_h - 34)

    draw_box_title("March Metrics — Data Scientist", "08/2020–Present", x_right, y_row1)
    draw_img(m_png, x_right, y_row1 - 34, box_w, box_h - 34)

    # Row 2
    y_row2 = y_row1 - box_h - 18
    draw_box_title("Sushi Primos — DS Consultant", "2024 • rounded & slightly altered", x_left, y_row2)
    draw_img(s_png, x_left, y_row2 - 34, box_w, box_h - 34)

    draw_box_title("ISU — GA Research (Personal Demo)", "Synthetic portfolio data (not ISU data)", x_right, y_row2)
    # place retention charts in a 2x2 within that box
    # We draw them as 2 rows of 2 images
    small_gap = 6
    small_w = (box_w - small_gap) / 2
    small_h = (box_h - 34 - small_gap) / 2

    y_img_top = y_row2 - 34
    draw_img(r_pngs[0], x_right, y_img_top, small_w, small_h)
    draw_img(r_pngs[1], x_right + small_w + small_gap, y_img_top, small_w, small_h)
    draw_img(r_pngs[2], x_right, y_img_top - small_h - small_gap, small_w, small_h)
    draw_img(r_pngs[3], x_right + small_w + small_gap, y_img_top - small_h - small_gap, small_w, small_h)

    c.showPage()
    c.save()

with colB:
    st.markdown("**Option B: Generate PDF Download (no print dialog)**")
    if st.button("Generate PDF file", type="primary"):
        build_pdf(PDF_OUT)
        st.success("PDF generated.")

    if PDF_OUT.exists():
        st.download_button(
            label="Download visual_resume.pdf",
            data=PDF_OUT.read_bytes(),
            file_name="visual_resume.pdf",
            mime="application/pdf"
        )

st.markdown("</div>", unsafe_allow_html=True)

# Optional table
with st.expander("Sushi Primos — Rounded KPI table (confidentiality-preserving)"):
    show_cols = ["Location", "Traffic", "FootTraffic", "Customers", "Residential", "SqFt", "ProfitPreRent", "Rent", "ProfitMinusRent"]
    st.dataframe(sushi_df[show_cols], use_container_width=True)
    st.markdown(
        """
        <div class="subtle">
        Targets: Traffic ≥ 1,000/day • Foot Traffic ≥ 250/day • Residential (0.5 mi) ≥ 3,500 • Rent &lt; $30,000/mo • SqFt 1,750–3,250.
        </div>
        """,
        unsafe_allow_html=True
    )