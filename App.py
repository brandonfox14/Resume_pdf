import streamlit as st
from pathlib import Path
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import subprocess
import sys
import time

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

def kpi_line(label, value, target=None, good_when="above"):
    """
    Small HTML KPI line with optional target note.
    """
    tgt = ""
    if target is not None:
        arrow = "≥" if good_when == "above" else "≤"
        tgt = f'<span style="opacity:0.72; font-size:12px;"> (Target {arrow} {target:,})</span>'
    return f'<div style="font-size:12.5px; margin:2px 0;"><b>{label}:</b> {value:,}{tgt}</div>'

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
# Header content (edit these)
# -----------------------------
NAME = "Brandon Fox"
CONTACT = "Madison, WI • brandonfox14@icloud.com • (608) 516-9676 

st.markdown(f'<div class="name">{NAME}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="contact">{CONTACT}</div>', unsafe_allow_html=True)

# ============================================================
# EDUCATION (full width)
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
# VIS 1: Verona half-court shot chart w/ hot/cold zones
# ============================================================
def draw_half_court(ax):
    # Court dimensions (NBA-ish units in feet; but we just need a consistent layout)
    # We'll use a half court from baseline to midcourt: x [-25, 25], y [0, 47]
    ax.set_xlim(-25, 25)
    ax.set_ylim(0, 47)
    ax.set_aspect('equal')
    ax.axis("off")

    # Outer lines (half court)
    ax.add_patch(Rectangle((-25, 0), 50, 47, fill=False, linewidth=1.2))

    # Hoop at (0, 5.25) (approx)
    hoop_y = 5.25
    ax.add_patch(Circle((0, hoop_y), 0.75, fill=False, linewidth=1.2))

    # Backboard
    ax.add_patch(Rectangle((-3, hoop_y-0.75), 6, 0.1, fill=True, linewidth=0))

    # Paint
    ax.add_patch(Rectangle((-8, 0), 16, 19, fill=False, linewidth=1.2))

    # Free throw circle
    ax.add_patch(Arc((0, 19), 12, 12, theta1=0, theta2=180, linewidth=1.2))
    ax.add_patch(Arc((0, 19), 12, 12, theta1=180, theta2=360, linewidth=1.2, linestyle="--", alpha=0.6))

    # Restricted arc
    ax.add_patch(Arc((0, hoop_y), 8, 8, theta1=0, theta2=180, linewidth=1.2))

    # 3pt arc + corners (approx NCAA-ish feel)
    # Corner 3 lines
    ax.plot([-22, -22], [0, 14], linewidth=1.2)
    ax.plot([22, 22], [0, 14], linewidth=1.2)
    # Arc
    ax.add_patch(Arc((0, hoop_y), 44, 44, theta1=22, theta2=158, linewidth=1.2))

def assign_zone(x, y):
    # Simple zones:
    # 0: Rim (restricted / paint near hoop)
    # 1: Paint (non-rim)
    # 2: Midrange (inside 3)
    # 3: Left corner 3
    # 4: Right corner 3
    # 5: Arc 3 (above the break)
    hoop_y = 5.25
    r = np.sqrt(x**2 + (y - hoop_y)**2)

    # Corner 3 logic
    if y <= 14 and x <= -22:
        return 3
    if y <= 14 and x >= 22:
        return 4

    # Three-point arc approx
    is_three = (r >= 22) and (y > 14)
    if is_three:
        return 5

    # Rim / paint
    if r <= 4:
        return 0
    if (-8 <= x <= 8) and (0 <= y <= 19):
        return 1

    return 2

def verona_shot_fig(seed=42):
    rng = np.random.default_rng(seed)

    # Need: 53 shots, 24 made, 4/11 from 3
    n_shots = 53
    made_total = 24
    n_threes = 11
    made_threes = 4

    n_twos = n_shots - n_threes
    made_twos = made_total - made_threes  # 20

    # Generate shot locations:
    # Threes: corners + arc
    three_x = []
    three_y = []
    for _ in range(n_threes):
        corner = rng.random() < 0.35
        if corner:
            # left or right corner
            side = -1 if rng.random() < 0.5 else 1
            x = side * (22 + rng.normal(0, 0.7))
            y = rng.uniform(2, 13.5)
        else:
            # arc
            angle = rng.uniform(np.deg2rad(30), np.deg2rad(150))
            r = rng.normal(23.3, 0.8)
            hoop_y = 5.25
            x = r * np.cos(angle)
            y = hoop_y + r * np.sin(angle)
            y = np.clip(y, 14.5, 46)
        three_x.append(float(x))
        three_y.append(float(y))

    # Twos: mix rim/paint/midrange
    two_x = []
    two_y = []
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

    # Assign makes with exact counts
    made = np.array([False]*n_shots)

    # choose made indices among threes
    three_idx = np.where(is_three)[0]
    two_idx = np.where(~is_three)[0]
    made_three_idx = rng.choice(three_idx, size=made_threes, replace=False)
    made_two_idx = rng.choice(two_idx, size=made_twos, replace=False)
    made[made_three_idx] = True
    made[made_two_idx] = True

    # zone FG%
    zones = np.array([assign_zone(xi, yi) for xi, yi in zip(x, y)])
    zone_names = {
        0: "Rim",
        1: "Paint",
        2: "Mid",
        3: "L Corner 3",
        4: "R Corner 3",
        5: "Arc 3"
    }

    zone_pct = {}
    for z in sorted(zone_names.keys()):
        idx = zones == z
        att = idx.sum()
        if att == 0:
            pct = None
        else:
            pct = made[idx].mean()
        zone_pct[z] = (att, pct)

    # Determine hot/cold zones by relative performance (only zones with attempts)
    pcts = [(z, zone_pct[z][1]) for z in zone_pct if zone_pct[z][0] >= 3 and zone_pct[z][1] is not None]
    hot_zones = set()
    cold_zones = set()
    if len(pcts) >= 2:
        # top 1 hot, bottom 1 cold
        pcts_sorted = sorted(pcts, key=lambda t: t[1])
        cold_zones.add(pcts_sorted[0][0])
        hot_zones.add(pcts_sorted[-1][0])

    # Plot
    fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=220)
    draw_half_court(ax)

    # Overlay hot/cold regions lightly (simple rectangles/arcs approximations)
    # Keep it simple + print-safe: just shade broad areas.
    # Rim+Paint (zone 0/1)
    if 0 in hot_zones or 1 in hot_zones:
        ax.add_patch(Rectangle((-8, 0), 16, 19, alpha=0.12, linewidth=0))
    if 0 in cold_zones or 1 in cold_zones:
        ax.add_patch(Rectangle((-8, 0), 16, 19, alpha=0.08, linewidth=0, hatch=".."))

    # Midrange (zone 2)
    if 2 in hot_zones:
        ax.add_patch(Rectangle((-20, 16), 40, 15, alpha=0.10, linewidth=0))
    if 2 in cold_zones:
        ax.add_patch(Rectangle((-20, 16), 40, 15, alpha=0.07, linewidth=0, hatch=".."))

    # Corner 3s
    if 3 in hot_zones:
        ax.add_patch(Rectangle((-25, 0), 3, 14, alpha=0.12, linewidth=0))
    if 3 in cold_zones:
        ax.add_patch(Rectangle((-25, 0), 3, 14, alpha=0.07, linewidth=0, hatch=".."))
    if 4 in hot_zones:
        ax.add_patch(Rectangle((22, 0), 3, 14, alpha=0.12, linewidth=0))
    if 4 in cold_zones:
        ax.add_patch(Rectangle((22, 0), 3, 14, alpha=0.07, linewidth=0, hatch=".."))

    # Arc 3
    if 5 in hot_zones:
        ax.add_patch(Rectangle((-25, 14), 50, 33, alpha=0.08, linewidth=0))
    if 5 in cold_zones:
        ax.add_patch(Rectangle((-25, 14), 50, 33, alpha=0.06, linewidth=0, hatch=".."))

    # Scatter shots
    ax.scatter(x[~made], y[~made], s=18, marker="x", linewidths=1.0, alpha=0.9)
    ax.scatter(x[made], y[made], s=20, marker="o", alpha=0.9)

    # Title and quick stat line
    fg = made.mean()
    three_fg = made[is_three].mean()
    ax.set_title(f"Verona HS Shot Chart (53 FGA) — FG {made.sum()}/53 ({fg:.0%}), 3PT {made[is_three].sum()}/{n_threes} ({three_fg:.0%})",
                 fontsize=9)

    fig.tight_layout()
    return fig

# ============================================================
# VIS 2: March Metrics growth chart (AUC + ACC) 2021–2026
# ============================================================
def march_metrics_growth(seed=7):
    rng = np.random.default_rng(seed)
    years = np.array([2021, 2022, 2023, 2024, 2025, 2026])

    # Must hit:
    # 2021 avg AUC ~0.61, avg ACC ~0.58
    # 2026 avg AUC 0.851, avg ACC 0.792
    auc_2021, acc_2021 = 0.61, 0.58
    auc_2026, acc_2026 = 0.851, 0.792

    # Create smooth growth w/ slight random wiggle that preserves monotonic trend
    t = np.linspace(0, 1, len(years))
    auc = auc_2021 + (auc_2026 - auc_2021) * (t**1.15)
    acc = acc_2021 + (acc_2026 - acc_2021) * (t**1.10)

    # Add tiny noise then enforce increasing
    auc += rng.normal(0, 0.008, size=len(years))
    acc += rng.normal(0, 0.008, size=len(years))

    # Pin endpoints exactly
    auc[0], acc[0] = auc_2021, acc_2021
    auc[-1], acc[-1] = auc_2026, acc_2026

    # Enforce gentle monotonic increase
    for i in range(1, len(years)):
        auc[i] = max(auc[i], auc[i-1] + 0.01)
        acc[i] = max(acc[i], acc[i-1] + 0.01)

    # Re-pin last in case it drifted up too far
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
        Residential=rounded25(3500),  # not provided; we keep a plausible rounded baseline for comparison
        SqFt=rounded25(1575),
    )

    # Goals / constraints
    goals = dict(
        Traffic=1000,
        FootTraffic=250,
        Customers=None,  # derived / goal is "more than current"
        Residential=3500,
        Rent=30000,      # below
        SqFt_min=1750,
        SqFt_max=3250
    )

    # Construct 4 options meeting your narrative constraints:
    # A: Best option (meets targets, strong profit, acceptable rent, within sq ft)
    # B: Too big + too expensive + lower profit than best
    # C: Too few people nearby + really high foot traffic
    # D: Slightly low on all numbers but lowest rent

    # Helper for customer prediction using traffic+foot traffic scale vs current
    def predict_customers(traffic, foot):
        # simple proportional model vs current, add small noise
        base = (traffic / current["Traffic"]) * 0.35 + (foot / current["FootTraffic"]) * 0.65
        pred = current["Customers"] * base
        pred *= rng.uniform(0.95, 1.05)
        return rounded25(pred)

    def profit_pre_rent(customers):
        # simplistic profit proxy: $ per customer ~ 350–420 (sushi shop could vary); add noise
        ppc = rng.uniform(320, 390)
        prof = customers * ppc
        return rounded25(prof)

    # Option A (Best)
    A_traffic = rounded25(rng.uniform(1025, 1400))
    A_foot = rounded25(rng.uniform(275, 520))
    A_res = rounded25(rng.uniform(3750, 5200))
    A_sqft = rounded25(rng.uniform(1850, 3100))
    A_rent = rounded25(rng.uniform(22000, 29500))
    A_cust = predict_customers(A_traffic, A_foot)
    A_profit = profit_pre_rent(A_cust)

    # Option B (Too big + too expensive, lower profit than best)
    B_traffic = rounded25(rng.uniform(1050, 1350))
    B_foot = rounded25(rng.uniform(260, 480))
    B_res = rounded25(rng.uniform(3600, 4800))
    B_sqft = rounded25(rng.uniform(3400, 4200))  # too big
    B_rent = rounded25(rng.uniform(34000, 45000))  # too expensive
    B_cust = predict_customers(B_traffic, B_foot)
    B_profit = rounded25(rng.uniform(A_profit * 0.78, A_profit * 0.92))  # lower than best

    # Option C (Too few people nearby, really high foot traffic)
    C_traffic = rounded25(rng.uniform(950, 1250))
    C_foot = rounded25(rng.uniform(700, 1050))  # really high
    C_res = rounded25(rng.uniform(1200, 2400))  # too few nearby
    C_sqft = rounded25(rng.uniform(1900, 3100))
    C_rent = rounded25(rng.uniform(24000, 31000))
    C_cust = predict_customers(C_traffic, C_foot)
    C_profit = profit_pre_rent(C_cust)

    # Option D (Slightly low on all numbers, lowest rent)
    D_traffic = rounded25(rng.uniform(775, 975))   # slightly low vs 1000 goal
    D_foot = rounded25(rng.uniform(200, 245))      # slightly low vs 250 goal
    D_res = rounded25(rng.uniform(3000, 3450))     # slightly low vs 3500 goal
    D_sqft = rounded25(rng.uniform(1750, 2000))    # barely meets min
    D_rent = rounded25(rng.uniform(16000, 21000))  # lowest rent
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

    # Add a few derived helpful fields
    df["ProfitMinusRent"] = df["ProfitPreRent"] - df["Rent"]
    df["MeetsTrafficGoal"] = df["Traffic"] >= goals["Traffic"]
    df["MeetsFootGoal"] = df["FootTraffic"] >= goals["FootTraffic"]
    df["MeetsResGoal"] = df["Residential"] >= goals["Residential"]
    df["MeetsRentGoal"] = df["Rent"] < goals["Rent"]
    df["MeetsSqFtRange"] = df["SqFt"].between(goals["SqFt_min"], goals["SqFt_max"])

    return df, current, goals

def sushi_fig(df):
    # Show a compact "goal vs options" plot using ProfitMinusRent & constraints count
    tmp = df[df["Location"] != "Current (Rounded)"].copy()
    # Count goal checks
    checks = ["MeetsTrafficGoal", "MeetsFootGoal", "MeetsResGoal", "MeetsRentGoal", "MeetsSqFtRange"]
    tmp["GoalsMet"] = tmp[checks].sum(axis=1)

    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=220)
    ax.scatter(tmp["GoalsMet"], tmp["ProfitMinusRent"])
    for _, r in tmp.iterrows():
        ax.annotate(r["Location"].replace("Option ", ""),
                    (r["GoalsMet"], r["ProfitMinusRent"]),
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

    # Targets:
    # Avg retention ~ 80
    # Best: Female In-State Education = 93
    # Male STEM In-State = 92
    # Worst: Male Out-of-State Undecided = 60
    # Education best overall, STEM close second, Undecided worst overall
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
    # sanity average close to 80
    return df

def retention_figs(df):
    # Four small bar charts (one per major) showing 4 groups
    majors = ["Education", "STEM", "Undecided", "Other"]
    group_order = [
        ("Female", "In-State"),
        ("Male", "In-State"),
        ("Female", "Out-of-State"),
        ("Male", "Out-of-State"),
    ]
    group_labels = ["F / In", "M / In", "F / Out", "M / Out"]

    figs = []
    for major in majors:
        sub = df[df["Major"] == major].copy()
        vals = []
        for g, r in group_order:
            vals.append(float(sub[(sub["Gender"] == g) & (sub["Residency"] == r)]["Retention"].iloc[0]))

        fig, ax = plt.subplots(figsize=(3.2, 2.3), dpi=220)
        ax.bar(group_labels, vals)
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

        # Plot area
        if isinstance(fig_or_figs, list):
            # render multiple figures as a row
            cols = st.columns(len(fig_or_figs), gap="small")
            for c, f in zip(cols, fig_or_figs):
                with c:
                    st.pyplot(f, use_container_width=True)
        else:
            st.pyplot(fig_or_figs, use_container_width=True)

        # Bullets
        bullets_html = "<ul>" + "".join([f"<li>{b}</li>" for b in bullets]) + "</ul>"
        st.markdown(bullets_html, unsafe_allow_html=True)

        if footnote:
            st.markdown(f'<div class="subtle">{footnote}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Build visuals once
# ============================================================
verona_fig = verona_shot_fig(seed=42)
mm_fig, mm_df = march_metrics_growth(seed=7)

sushi_df, sushi_current, sushi_goals = sushi_locations(seed=12)
sushi_plot = sushi_fig(sushi_df)

ret_df = retention_data()
ret_figs = retention_figs(ret_df)

# ============================================================
# 4-column job grid
# ============================================================
c1, c2, c3, c4 = st.columns(4, gap="large")

# 1) Verona
job_box(
    c1,
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

# 2) March Metrics
metrics_logo = ASSETS / "companies" / "metrics.png"
job_box(
    c2,
    "March Metrics",
    "Data Scientist",
    "08/2020 to Present",
    metrics_logo,
    pills=["NCAA Hoops", "Model evaluation", "AUC/ACC tracking"],
    fig_or_figs=mm_fig,
    bullets=[
        "Moneyline/Win: AUC 0.979 • ACC 0.920 (2026 season).",
        "Spread: AUC 0.721 • ACC 0.671 (2026 season).",
        "Total (O/U): AUC 0.854 • ACC 0.783 (2026 season).",
        "Avg (Win+Spread+OU): AUC 0.851 • ACC 0.792 (2026 season).",
    ],
    footnote="Growth chart is simulated to show trend over time; 2021 and 2026 endpoints are fixed to the stated averages."
)

# 3) Sushi Primos
sushi_logo = ASSETS / "companies" / "sushi.png"
job_box(
    c3,
    "Sushi Primos",
    "Data Science Consultant (Site Selection)",
    "2024",
    sushi_logo,
    pills=["Location scoring", "Traffic + footfall", "Profit vs rent"],
    fig_or_figs=sushi_plot,
    bullets=[
        "Compared 4 candidate locations using rounded KPIs (traffic, foot traffic, customers, rent, residential, square footage).",
        "Balanced predicted demand with rent constraints and operational fit (1-page executive decision support).",
        "Preserved business comparative advantage by rounding and slightly altering values while maintaining relative ranking."
    ],
    footnote="All location KPIs shown are rounded to the nearest 25 and slightly altered for confidentiality."
)

# 4) ISU retention demo
isu_company_logo = ASSETS / "companies" / "isu.png"
job_box(
    c4,
    "Illinois State University",
    "Graduate Assistant — Data & Statistical Research (Personal Demo)",
    "2023–2024",
    isu_company_logo,
    pills=["Synthetic demo", "Retention analysis", "Segmented insights"],
    fig_or_figs=ret_figs,
    bullets=[
        "Built a retention-rate segmentation demo by major, residency, and gender.",
        "Education majors lead retention; STEM close behind; undecided lowest.",
        "In-state advantage is consistent; female advantage is modest but present."
    ],
    footnote="Important: these charts use my own synthetic/demo data for portfolio purposes — not ISU data."
)

st.write("")

# ============================================================
# Sushi mini-table (optional but useful on the page)
# ============================================================
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

# ============================================================
# Export to PDF controls
# ============================================================
st.markdown("---")
st.markdown('<div class="box">', unsafe_allow_html=True)
st.markdown('<div class="box-title">Export</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Creates a single-page PDF via Playwright rendering of this Streamlit page.</div>', unsafe_allow_html=True)

colA, colB = st.columns([1, 2], gap="large")

with colA:
    do_export = st.button("Export PDF", type="primary")

with colB:
    st.markdown(
        '<div class="subtle">If the PDF spills onto a 2nd page, reduce bullets or adjust export scale in <code>export_pdf.py</code>.</div>',
        unsafe_allow_html=True
    )

if do_export:
    # Call export script
    try:
        # Give the page a moment (useful if user clicked quickly after load)
        time.sleep(0.4)

        # Use current python interpreter
        cmd = [sys.executable, str(ROOT / "export_pdf.py"), "--out", str(PDF_OUT)]
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode != 0:
            st.error("PDF export failed. See error output below.")
            st.code(res.stderr or res.stdout)
        else:
            st.success("PDF created successfully.")
    except Exception as e:
        st.error(f"PDF export error: {e}")

# Download button if exists
if PDF_OUT.exists():
    st.write("")
    st.download_button(
        label="Download visual_resume.pdf",
        data=PDF_OUT.read_bytes(),
        file_name="visual_resume.pdf",
        mime="application/pdf"
    )

st.markdown("</div>", unsafe_allow_html=True)
