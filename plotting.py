"""
plotting.py
-----------
Reproduces all figures from Haris et al. (2023) using the DE-optimised layouts.

Figures produced (active in this pipeline):
  Fig 2 - Cosine efficiency 4-panel (vernal eq, summer sol, autumnal eq, winter sol)
           ← Radial Staggered, optimised layout, four design-day columns
  Fig 4 - Power variation throughout year (Fermat spiral, 4-panel, four design days)
           ← Fermat Spiral, optimised layout
  Fig 5 - Optimised layouts side-by-side (RS + FS)
  Fig 6 - DE convergence curves (both layouts overlaid on one axes, 100 generations)
  Fig 7 - Efficiency comparison bar chart (before/after, both layouts)
  Fig 8 - DNI data from CSV (monthly/seasonal averages)

Figures removed from the active pipeline (unoptimised GA baselines):
  Fig 1 - plot_attenuation_rs()   → attenuation map, RS unoptimised default params
  Fig 3 - plot_attenuation_fs()   → attenuation map, FS unoptimised default params
  These functions are retained below for reference / standalone use but are no
  longer called from main.py.  In the original GA paper (Haris et al. 2023) they
  correspond to Figs 1 & 3 showing the pre-optimisation field.  The DE pipeline
  skips them because we only visualise the optimised result.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

from heliostat_field import FieldParams, radial_staggered_layout, fermat_spiral_layout
from efficiency import (overall_efficiency, attenuation_efficiency,
                        cosine_efficiency, field_total_power_mw)
from solar_geometry import (solar_elevation, solar_azimuth,
                             DESIGN_POINTS, LATITUDE_DEG)

CMAP_EFF = "RdYlGn"
CMAP_ATT = "RdYlGn"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":    "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":     300,
    "savefig.dpi":    300,
})


def _scatter_field(ax, positions, values, title, cmap=CMAP_EFF,
                   vmin=None, vmax=None, cbar_label="Efficiency (%)"):
    sc = ax.scatter(positions[:, 0], positions[:, 1],
                    c=values, cmap=cmap, s=6,
                    vmin=vmin, vmax=vmax, linewidths=0)
    plt.colorbar(sc, ax=ax, label=cbar_label, fraction=0.04, pad=0.04)
    ax.set_title(title, pad=6)
    ax.set_xlabel("Distance from Tower (m)")
    ax.set_ylabel("Distance from Tower (m)")
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.axvline(0, color="k", lw=0.4, ls="--")
    for txt, xy in [("N", (0, 1)), ("S", (0, -1)), ("E", (1, 0)), ("W", (-1, 0))]:
        r = max(np.abs(positions).max() * 1.05, 50)
        ax.text(xy[0]*r*0.92, xy[1]*r*0.92, txt, ha="center", va="center",
                fontsize=7, color="gray")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 1 – Attenuation efficiency of unoptimised radial staggered layout
# NOTE: This function is retained for reference only.  It is NOT called from
#       main.py in the DE pipeline.  It corresponds to the pre-optimisation
#       baseline that was plotted in the original GA paper (Haris et al. 2023,
#       Fig 1).  Call it standalone if you need the unoptimised comparison map.
# ──────────────────────────────────────────────────────────────────────────────

def plot_attenuation_rs(params: FieldParams = None):
    """
    [REFERENCE ONLY – not called in active pipeline]
    Attenuation efficiency map for the *unoptimised* radial-staggered layout
    (default FieldParams).  Equivalent to Fig 1 in Haris et al. (2023).
    """
    if params is None:
        params = FieldParams()
    pos  = radial_staggered_layout(params, max_radius=450)
    f_at = attenuation_efficiency(pos, params.tower_height) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_field(ax, pos, f_at,
                   "Attenuation Efficiency – Radial Staggered (Unoptimised)",
                   cbar_label="Attenuation Efficiency (%)",
                   vmin=96, vmax=100)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_attenuation_radial_staggered.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 2 – Cosine + overall efficiency 4-panel (RS, optimised layout)
#
#  Four columns = four solar design days:
#    Vernal Equinox  (day 80,  Mar 21)
#    Summer Solstice (day 172, Jun 21)
#    Autumnal Equinox(day 266, Sep 23)
#    Winter Solstice (day 355, Dec 21)
#  Row 0: overall efficiency map   Row 1: cosine efficiency map
#
#  Accepts optional `params` so it can be driven by the DE-optimised result.
# ──────────────────────────────────────────────────────────────────────────────

def plot_cosine_4panel_rs(params: FieldParams = None):
    """
    Cosine + overall efficiency 4-panel for the Radial Staggered layout.
    Pass the DE-optimised FieldParams to plot the post-optimisation field;
    omit (or pass None) to fall back to default (unoptimised) params.
    Iterates over all four design days defined in solar_geometry.DESIGN_POINTS.
    """
    if params is None:
        params = FieldParams()
    pos     = radial_staggered_layout(params, max_radius=450)
    dp_list = list(DESIGN_POINTS.items())   # 4 design points

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for col, (name, info) in enumerate(dp_list):
        day   = info["day"]
        alpha = solar_elevation(day)
        A     = solar_azimuth(day)

        cos_eff = cosine_efficiency(pos, params.tower_height, alpha, A) * 100
        ov_eff  = overall_efficiency(pos, params, day) * 100

        ax_ov  = axes[0, col]
        ax_cos = axes[1, col]

        _scatter_field(ax_ov, pos, ov_eff,
                       f"Overall Eff.\n{name}", vmin=50, vmax=100,
                       cbar_label="Efficiency (%)")
        _scatter_field(ax_cos, pos, cos_eff,
                       f"Cosine Eff.\n{name} (11 AM)", vmin=50, vmax=110,
                       cbar_label="Cosine Eff. (%)")

    fig.suptitle(
        "Efficiency Maps – Radial Staggered Layout | Quetta, Pakistan\n"
        "Four Design Days: Vernal Eq. / Summer Sol. / Autumnal Eq. / Winter Sol.",
        fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_cosine_4panel_radial_staggered.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 3 – Attenuation efficiency of unoptimised Fermat spiral layout
# NOTE: This function is retained for reference only.  It is NOT called from
#       main.py in the DE pipeline.  It corresponds to the pre-optimisation
#       baseline that appears in the original GA paper (Haris et al. 2023,
#       Fig 3).  Call it standalone if you need the unoptimised comparison map.
# ──────────────────────────────────────────────────────────────────────────────

def plot_attenuation_fs(params: FieldParams = None):
    """
    [REFERENCE ONLY – not called in active pipeline]
    Attenuation efficiency map for the *unoptimised* Fermat spiral layout
    (default FieldParams).  Equivalent to Fig 3 in Haris et al. (2023).
    """
    if params is None:
        params = FieldParams()
    pos  = fermat_spiral_layout(params, n_heliostats=1300)
    f_at = attenuation_efficiency(pos, params.tower_height) * 100

    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_field(ax, pos, f_at,
                   "Attenuation Efficiency – Fermat's Spiral (Unoptimised)",
                   cbar_label="Attenuation Efficiency (%)",
                   vmin=90, vmax=100)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_attenuation_fermat_spiral.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 4 – Power variation throughout the year (Fermat spiral, 4-panel)
#
#  Four columns = same four solar design days as Fig 2.
#  Row 0: power per heliostat (kW)   Row 1: overall efficiency map
#
#  Accepts optional `params` so it can be driven by the DE-optimised result.
# ──────────────────────────────────────────────────────────────────────────────

def plot_power_4panel_fs(design_dni: dict, params: FieldParams = None):
    """
    Power + efficiency 4-panel for the Fermat Spiral layout.
    Pass the DE-optimised FieldParams to plot the post-optimisation field;
    omit (or pass None) to fall back to default (unoptimised) params.
    Iterates over all four design days defined in solar_geometry.DESIGN_POINTS.
    """
    if params is None:
        params = FieldParams()
    pos     = fermat_spiral_layout(params, n_heliostats=1300)
    dp_list = list(DESIGN_POINTS.items())

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for col, (name, info) in enumerate(dp_list):
        day = info["day"]
        dni = design_dni.get(name, 860.0)

        from efficiency import power_per_heliostat
        pw = power_per_heliostat(pos, params, day, dni) / 1000   # kW
        ov = overall_efficiency(pos, params, day) * 100

        ax_pw = axes[0, col]
        ax_ov = axes[1, col]

        _scatter_field(ax_pw, pos, pw,
                       f"Power/Heliostat (kW)\n{name}",
                       cbar_label="Power (kW)", vmin=0)
        _scatter_field(ax_ov, pos, ov,
                       f"Overall Efficiency (%)\n{name}",
                       vmin=30, vmax=100)

    fig.suptitle(
        "Power Variation – Fermat's Spiral Layout | Quetta, Pakistan\n"
        "Four Design Days: Vernal Eq. / Summer Sol. / Autumnal Eq. / Winter Sol.",
        fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_power_4panel_fermat.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 5 – Optimised field layouts side-by-side
# ──────────────────────────────────────────────────────────────────────────────

def plot_optimised_layouts(rs_result, fs_result, design_dni: dict):
    from heliostat_field import radial_staggered_layout, fermat_spiral_layout

    rs_pos = radial_staggered_layout(rs_result.best_params, max_radius=600)
    fs_pos = fermat_spiral_layout(fs_result.best_params, n_heliostats=1300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = [
        f"Optimised – Radial Staggered\n"
        f"DE: TH={rs_result.best_params.tower_height:.0f}m "
        f"LH={rs_result.best_params.heliostat_length:.1f}m "
        f"η={rs_result.best_efficiency:.1f}%",

        f"Optimised – Fermat's Spiral\n"
        f"DE: TH={fs_result.best_params.tower_height:.0f}m "
        f"LH={fs_result.best_params.heliostat_length:.1f}m "
        f"η={fs_result.best_efficiency:.1f}%",
    ]

    for ax, pos, params, title in zip(
            axes,
            [rs_pos, fs_pos],
            [rs_result.best_params, fs_result.best_params],
            titles):
        ov = overall_efficiency(pos, params, DESIGN_POINTS["Vernal Equinox"]["day"]) * 100
        _scatter_field(ax, pos, ov, title, vmin=50, vmax=100)
        ax.plot(0, 0, "k^", ms=10, label="Receiver")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Optimised Field Layouts – DE Optimisation | Quetta, Pakistan (DNI≈950 W/m²)",
        fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_optimised_layouts.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 6 – DE convergence curves  (both layouts overlaid, 100 generations)
#
#  Both Radial Staggered and Fermat Spiral convergence curves are drawn on
#  a single axes, matching Fig 6(b) of Haris et al. (2023).
#  The x-axis always spans 0 → max_generations (100) because de_optimizer.py
#  pads the history list to full length on early convergence.
# ──────────────────────────────────────────────────────────────────────────────

def plot_convergence(rs_result, fs_result):
    """
    Plot DE convergence for Radial Staggered and Fermat Spiral on one axes.

    convergence_history[0]  = best fitness of the initial population (gen 0)
    convergence_history[g+1] = population-best after generation g
    Length is always max_generations + 1 (padded in de_optimizer if early stop).
    """
    rs_hist = np.array(rs_result.convergence_history)
    fs_hist = np.array(fs_result.convergence_history)

    # x-axis: generation index (0 = initial population)
    rs_gens = np.arange(len(rs_hist))
    fs_gens = np.arange(len(fs_hist))

    colour_rs = "#d62728"    # red  – matches paper colour convention
    colour_fs = "#1f77b4"    # blue

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(rs_gens, rs_hist,
            color=colour_rs, lw=1.8, label="Radial staggered")
    ax.fill_between(rs_gens, rs_hist,
                    alpha=0.12, color=colour_rs)

    ax.plot(fs_gens, fs_hist,
            color=colour_fs, lw=1.8, linestyle="--", label="Fermat spiral")
    ax.fill_between(fs_gens, fs_hist,
                    alpha=0.12, color=colour_fs)

    # Annotate final best-fitness values
    ax.annotate(f"{rs_result.best_efficiency:.2f}%",
                xy=(rs_gens[-1], rs_hist[-1]),
                xytext=(-38, 6), textcoords="offset points",
                fontsize=7.5, color=colour_rs,
                arrowprops=dict(arrowstyle="-", color=colour_rs, lw=0.8))
    ax.annotate(f"{fs_result.best_efficiency:.2f}%",
                xy=(fs_gens[-1], fs_hist[-1]),
                xytext=(-38, -14), textcoords="offset points",
                fontsize=7.5, color=colour_fs,
                arrowprops=dict(arrowstyle="-", color=colour_fs, lw=0.8))

    ax.set_xlabel("Generation", fontsize=9)
    ax.set_ylabel("Best η (annual mean efficiency, %)", fontsize=9)
    ax.set_title("(b) DE convergence – Radial Staggered & Fermat Spiral\n"
                 "Quetta, Pakistan | pop=30 · F=0.8 · CR=0.7 · max_gen=100",
                 fontsize=9)
    ax.set_xlim(0, max(len(rs_hist), len(fs_hist)) - 1)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, linewidth=0.6)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_de_convergence.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 7 – Before/After efficiency comparison bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_efficiency_comparison(before_rs: float, after_rs: float,
                               before_fs: float, after_fs: float,
                               n_before_rs: int, n_after_rs: int,
                               n_before_fs: int, n_after_fs: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(2)
    w = 0.35
    colours = ["#457b9d", "#e63946"]

    # Efficiency bars
    ax = axes[0]
    bars1 = ax.bar(x - w/2, [before_rs, before_fs], w, label="Before DE",
                   color=colours[0], alpha=0.85)
    bars2 = ax.bar(x + w/2, [after_rs, after_fs],   w, label="After DE",
                   color=colours[1], alpha=0.85)
    ax.bar_label(bars1, fmt="%.2f%%", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Radial Staggered", "Fermat's Spiral"])
    ax.set_ylabel("Annual Mean Efficiency (%)")
    ax.set_title("Efficiency: Before vs After DE Optimisation")
    ax.legend()
    ax.set_ylim(0, max(after_rs, after_fs) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    # Heliostat count bars
    ax2 = axes[1]
    bars3 = ax2.bar(x - w/2, [n_before_rs, n_before_fs], w, label="Before DE",
                    color=colours[0], alpha=0.85)
    bars4 = ax2.bar(x + w/2, [n_after_rs,  n_after_fs],  w, label="After DE",
                    color=colours[1], alpha=0.85)
    ax2.bar_label(bars3, padding=3, fontsize=8)
    ax2.bar_label(bars4, padding=3, fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Radial Staggered", "Fermat's Spiral"])
    ax2.set_ylabel("Number of Heliostats")
    ax2.set_title("Heliostat Count: Before vs After DE Optimisation")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    for ax_ in axes:
        ax_.spines["top"].set_visible(False)
        ax_.spines["right"].set_visible(False)

    fig.suptitle("DE Optimisation Results – Quetta, Pakistan", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig7_efficiency_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 8 – DNI data visualisation from CSV
# ──────────────────────────────────────────────────────────────────────────────

def plot_dni_data(df):
    """Plot seasonal and diurnal DNI patterns from the CSV dataset."""
    import pandas as pd

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A – Monthly average DNI
    ax1 = fig.add_subplot(gs[0, 0])
    monthly    = df.groupby("month")["dni"].mean()
    months_lbl = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    colours_m  = plt.cm.plasma(np.linspace(0.15, 0.85, 12))
    ax1.bar(monthly.index, monthly.values, color=colours_m)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(months_lbl, rotation=45, ha="right")
    ax1.set_ylabel("Average DNI (W/m²)")
    ax1.set_title("Monthly Average DNI – Quetta")
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel B – Design point DNI box plots
    ax2 = fig.add_subplot(gs[0, 1])
    dp_data, dp_labels = [], []
    windows = {
        "Vernal\nEquinox":  (3,  21),
        "Summer\nSolstice": (6,  21),
        "Autumnal\nEquinox":(9,  23),
        "Winter\nSolstice": (12, 21),
    }
    for label, (m, d) in windows.items():
        sub = df[(df["month"] == m) &
                 (df["day"].between(d-2, d+2)) &
                 (df["hour"] >= 9) & (df["hour"] <= 15)]["dni"]
        dp_data.append(sub.dropna().values)
        dp_labels.append(label)

    bp = ax2.boxplot(dp_data, labels=dp_labels, patch_artist=True,
                     medianprops=dict(color="black", lw=2))
    for patch, color in zip(bp["boxes"],
                            ["#e9c46a", "#f4a261", "#e76f51", "#264653"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax2.set_ylabel("DNI (W/m²)")
    ax2.set_title("DNI at Design Points (±2 days, 9–15h)")
    ax2.grid(True, axis="y", alpha=0.3)

    # Panel C – Diurnal DNI profile for each season
    ax3 = fig.add_subplot(gs[1, 0])
    colours_s = ["#e9c46a", "#f4a261", "#e76f51", "#264653"]
    for (label, (m, d)), col in zip(windows.items(), colours_s):
        sub    = df[(df["month"] == m) & (df["day"].between(d-5, d+5))]
        hourly = sub.groupby(sub["hour"].round(0))["dni"].mean()
        ax3.plot(hourly.index, hourly.values,
                 label=label.replace("\n", " "), color=col, lw=2)
    ax3.set_xlabel("Solar Hour")
    ax3.set_ylabel("DNI (W/m²)")
    ax3.set_title("Diurnal DNI Profile by Season")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel D – Time-series of daily peak DNI
    ax4 = fig.add_subplot(gs[1, 1])
    daily_peak = (df[df["hour"].between(10, 12)]
                  .groupby(df["time"].dt.date)["dni"].max())
    daily_peak.index = pd.to_datetime(daily_peak.index)
    ax4.plot(daily_peak.index, daily_peak.values,
             lw=0.6, color="#2a9d8f", alpha=0.7)
    roll = daily_peak.rolling(30, min_periods=5).mean()
    ax4.plot(roll.index, roll.values, color="#e76f51", lw=2, label="30-day mean")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Peak Midday DNI (W/m²)")
    ax4.set_title("Daily Peak DNI (10–12h) Over Dataset Period")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "Solar DNI Dataset Analysis – Quetta (BUITEMS, 2015–2017)\n"
        "Source: World Bank ESMAP Tier-2 Station",
        fontsize=10)
    path = os.path.join(OUT_DIR, "fig8_dni_analysis.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
