"""
plotting.py
-----------
Reproduces all figures from Haris et al. (2023) using the DE-optimised layouts.

Active figures in this pipeline
--------------------------------
  Fig 1b  – Unoptimised field layouts 1×2 (RS left | FS right) + tower icon ★
  Fig 2u  – Unoptimised Radial Staggered 2×2 (one panel per design day)
  Fig 2   – Optimised Radial Staggered 2×2 (one panel per design day, avg DNI)
  Fig 4   – Optimised Fermat Spiral 2×2 (power/heliostat, avg DNI)
  Fig 5   – Optimised layouts side-by-side (RS + FS) with tower icon
  Fig 6   – DE convergence curves (both layouts overlaid, 100 generations)
  Fig 7   – Efficiency comparison bar chart (before/after, both layouts)

Removed from active pipeline
-----------------------------
  Fig 1   – plot_attenuation_rs()  [kept as reference function, not called]
  Fig 3   – plot_attenuation_fs()  [kept as reference function, not called]
  Fig 8   – DNI dataset analysis   [removed per request]

Design notes
-------------
* The 2×2 efficiency / power grids use a SINGLE avg DNI (mean of all four
  design-point DNIs) so DNI is not counted twice per layout.
* A red ★ tower icon is placed at (0, 0) in every scatter map.
* Shared colour scales across panels within each figure for easy comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

from heliostat_field import FieldParams, radial_staggered_layout, fermat_spiral_layout
from efficiency import (overall_efficiency, attenuation_efficiency,
                        cosine_efficiency, field_total_power_mw,
                        power_per_heliostat)
from solar_geometry import (solar_elevation, solar_azimuth,
                             DESIGN_POINTS, LATITUDE_DEG)

CMAP_EFF = "RdYlGn"
CMAP_ATT = "RdYlGn"
CMAP_PWR = "plasma"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _draw_tower(ax, size: float = 14):
    """Draw a red ★ receiver tower icon at the origin."""
    ax.plot(0, 0,
            marker="*",
            markersize=size,
            color="#c0392b",
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=10,
            label="Tower / Receiver")


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
    r = max(float(np.abs(positions).max()) * 1.05, 50)
    for txt, xy in [("N", (0, 1)), ("S", (0, -1)), ("E", (1, 0)), ("W", (-1, 0))]:
        ax.text(xy[0]*r*0.92, xy[1]*r*0.92, txt, ha="center", va="center",
                fontsize=7, color="gray")


def _avg_design_dni(design_dni: dict) -> float:
    """Return the mean DNI across all four design-point values."""
    return float(np.mean(list(design_dni.values())))


# ──────────────────────────────────────────────────────────────────────────────
# Fig 1 – Attenuation efficiency of unoptimised RS layout  [REFERENCE ONLY]
# ──────────────────────────────────────────────────────────────────────────────

def plot_attenuation_rs(params: FieldParams = None):
    """[REFERENCE ONLY] Attenuation map for unoptimised Radial Staggered."""
    if params is None:
        params = FieldParams()
    pos  = radial_staggered_layout(params, max_radius=450)
    f_at = attenuation_efficiency(pos, params.tower_height) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_field(ax, pos, f_at,
                   "Attenuation Efficiency – Radial Staggered (Unoptimised)",
                   cbar_label="Attenuation Efficiency (%)", vmin=96, vmax=100)
    _draw_tower(ax)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_attenuation_radial_staggered.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 1b – Unoptimised field layouts 1×2 (RS | FS) with tower icon
# ──────────────────────────────────────────────────────────────────────────────

def plot_unoptimised_layouts_1x2(params: FieldParams = None):
    """
    1×2 side-by-side of the unoptimised Radial Staggered and Fermat Spiral
    layouts, coloured by overall efficiency at the Vernal Equinox (11 AM).
    A red ★ tower icon marks the central receiver at (0, 0) in both panels.
    Shared colour scale across both axes for direct comparison.
    """
    if params is None:
        params = FieldParams()

    rs_pos = radial_staggered_layout(params, max_radius=450)
    fs_pos = fermat_spiral_layout(params, n_heliostats=1300)

    day = DESIGN_POINTS["Vernal Equinox"]["day"]

    rs_ov = overall_efficiency(rs_pos, params, day) * 100
    fs_ov = overall_efficiency(fs_pos, params, day) * 100

    vmin = float(min(rs_ov.min(), fs_ov.min()))
    vmax = float(max(rs_ov.max(), fs_ov.max()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pairs = [
        (axes[0], rs_pos, rs_ov, "Radial Staggered"),
        (axes[1], fs_pos, fs_ov, "Fermat's Spiral"),
    ]
    for ax, pos, ov, label in pairs:
        _scatter_field(ax, pos, ov,
                       f"Unoptimised – {label}\n"
                       f"Overall Efficiency (%)  •  Vernal Equinox, 11 AM\n"
                       f"N = {len(pos):,} heliostats",
                       vmin=vmin, vmax=vmax,
                       cbar_label="Overall Efficiency (%)")
        _draw_tower(ax, size=16)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Unoptimised Field Layouts – Radial Staggered & Fermat's Spiral\n"
        "Default FieldParams  |  Quetta, Pakistan",
        fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1b_unoptimised_layouts_1x2.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 2u – Unoptimised Radial Staggered 2×2 (one panel per design day)
# ──────────────────────────────────────────────────────────────────────────────

def plot_unoptimised_rs_2x2(params: FieldParams = None):
    """
    2×2 overall-efficiency grid for the *unoptimised* Radial Staggered layout.
    One panel per seasonal design day; shared colour scale; tower icon in each.
    """
    if params is None:
        params = FieldParams()

    pos     = radial_staggered_layout(params, max_radius=450)
    dp_list = list(DESIGN_POINTS.items())     # 4 items → 2 rows × 2 cols

    # Pre-compute shared colour limits
    all_eff_vals = np.concatenate([
        overall_efficiency(pos, params, info["day"]) * 100
        for _, info in dp_list
    ])
    vmin = float(max(0.0, all_eff_vals.min()))
    vmax = float(all_eff_vals.max())

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes_flat = axes.flatten()

    for ax, (name, info) in zip(axes_flat, dp_list):
        day      = info["day"]
        ov       = overall_efficiency(pos, params, day) * 100
        mean_eff = float(np.mean(ov))
        elev     = np.degrees(solar_elevation(day))

        _scatter_field(ax, pos, ov,
                       f"{name}  (Day {day})\n"
                       f"Solar Elev. {elev:.1f}°   Mean η = {mean_eff:.2f}%",
                       vmin=vmin, vmax=vmax,
                       cbar_label="Overall Efficiency (%)")
        _draw_tower(ax)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Unoptimised Radial Staggered Layout – Overall Efficiency (2×2)\n"
        f"Default FieldParams  |  Quetta, Pakistan  |  11:00 AM solar time\n"
        f"N = {len(pos):,} heliostats",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "fig2u_unoptimised_rs_2x2.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 2 – Optimised Radial Staggered 2×2 (avg DNI, one panel per design day)
# ──────────────────────────────────────────────────────────────────────────────

def plot_cosine_4panel_rs(params: FieldParams = None, design_dni: dict = None):
    """
    2×2 overall-efficiency grid for the *optimised* Radial Staggered layout.
    Uses a single avg_dni (mean of all four design-point DNIs) annotated in
    each sub-title, so DNI is not double-counted across panels.
    Shared colour scale; red ★ tower icon in every panel.

    Parameters
    ----------
    params     : DE-optimised FieldParams (None → default/unoptimised)
    design_dni : dict from solar_geometry.average_design_point_dni()
    """
    if params is None:
        params = FieldParams()

    pos     = radial_staggered_layout(params, max_radius=600)
    dp_list = list(DESIGN_POINTS.items())

    avg_dni = _avg_design_dni(design_dni) if design_dni else 889.0

    # Shared colour limits
    all_eff_vals = np.concatenate([
        overall_efficiency(pos, params, info["day"]) * 100
        for _, info in dp_list
    ])
    vmin = float(max(0.0, all_eff_vals.min()))
    vmax = float(all_eff_vals.max())

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes_flat = axes.flatten()

    for ax, (name, info) in zip(axes_flat, dp_list):
        day      = info["day"]
        ov       = overall_efficiency(pos, params, day) * 100
        mean_eff = float(np.mean(ov))
        elev     = np.degrees(solar_elevation(day))

        _scatter_field(ax, pos, ov,
                       f"{name}  (Day {day})\n"
                       f"Solar Elev. {elev:.1f}°   Avg DNI {avg_dni:.0f} W/m²\n"
                       f"Mean η = {mean_eff:.2f}%",
                       vmin=vmin, vmax=vmax,
                       cbar_label="Overall Efficiency (%)")
        _draw_tower(ax)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Optimised Radial Staggered Layout – Overall Efficiency (2×2)\n"
        f"DE-Optimised  |  Quetta, Pakistan  |  Avg DNI {avg_dni:.0f} W/m²  "
        f"|  N = {len(pos):,} heliostats",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "fig2_cosine_4panel_radial_staggered.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 3 – Attenuation efficiency of unoptimised FS layout  [REFERENCE ONLY]
# ──────────────────────────────────────────────────────────────────────────────

def plot_attenuation_fs(params: FieldParams = None):
    """[REFERENCE ONLY] Attenuation map for unoptimised Fermat Spiral."""
    if params is None:
        params = FieldParams()
    pos  = fermat_spiral_layout(params, n_heliostats=1300)
    f_at = attenuation_efficiency(pos, params.tower_height) * 100

    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_field(ax, pos, f_at,
                   "Attenuation Efficiency – Fermat's Spiral (Unoptimised)",
                   cbar_label="Attenuation Efficiency (%)", vmin=90, vmax=100)
    _draw_tower(ax)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_attenuation_fermat_spiral.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 4 – Optimised Fermat Spiral 2×2 – power/heliostat, avg DNI
# ──────────────────────────────────────────────────────────────────────────────

def plot_power_4panel_fs(design_dni: dict, params: FieldParams = None):
    """
    2×2 power-per-heliostat grid for the *optimised* Fermat Spiral layout.
    A single avg_dni (mean across all four design points) is applied to every
    panel — the panels differ only in solar geometry (elevation / azimuth).
    Shared colour scale; red ★ tower icon in every panel.
    """
    if params is None:
        params = FieldParams()

    pos     = fermat_spiral_layout(params, n_heliostats=1300)
    dp_list = list(DESIGN_POINTS.items())

    avg_dni = _avg_design_dni(design_dni)

    # Shared colour limits
    all_pw_vals = np.concatenate([
        power_per_heliostat(pos, params, info["day"], avg_dni) / 1000
        for _, info in dp_list
    ])
    vmin, vmax = 0.0, float(all_pw_vals.max())

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes_flat = axes.flatten()

    for ax, (name, info) in zip(axes_flat, dp_list):
        day      = info["day"]
        pw       = power_per_heliostat(pos, params, day, avg_dni) / 1000   # kW
        mean_pw  = float(np.mean(pw))
        elev     = np.degrees(solar_elevation(day))

        _scatter_field(ax, pos, pw,
                       f"{name}  (Day {day})\n"
                       f"Solar Elev. {elev:.1f}°   Avg DNI {avg_dni:.0f} W/m²\n"
                       f"Mean power = {mean_pw:.2f} kW/heliostat",
                       cmap=CMAP_PWR,
                       vmin=vmin, vmax=vmax,
                       cbar_label="Power per Heliostat (kW)")
        _draw_tower(ax)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Optimised Fermat's Spiral Layout – Power per Heliostat (2×2)\n"
        f"DE-Optimised  |  Quetta, Pakistan  |  Avg DNI {avg_dni:.0f} W/m²  "
        f"|  N = {len(pos):,} heliostats",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "fig4_power_4panel_fermat.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 5 – Optimised layouts side-by-side (RS + FS) with tower icon
# ──────────────────────────────────────────────────────────────────────────────

def plot_optimised_layouts(rs_result, fs_result, design_dni: dict):
    rs_pos = radial_staggered_layout(rs_result.best_params, max_radius=600)
    fs_pos = fermat_spiral_layout(fs_result.best_params, n_heliostats=1300)

    avg_dni = _avg_design_dni(design_dni)
    day     = DESIGN_POINTS["Vernal Equinox"]["day"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    data = [
        (axes[0], rs_pos, rs_result.best_params,
         f"Optimised – Radial Staggered\n"
         f"TH={rs_result.best_params.tower_height:.0f} m  "
         f"LH={rs_result.best_params.heliostat_length:.1f} m  "
         f"η={rs_result.best_efficiency:.1f}%"),
        (axes[1], fs_pos, fs_result.best_params,
         f"Optimised – Fermat's Spiral\n"
         f"TH={fs_result.best_params.tower_height:.0f} m  "
         f"LH={fs_result.best_params.heliostat_length:.1f} m  "
         f"η={fs_result.best_efficiency:.1f}%"),
    ]
    for ax, pos, par, title in data:
        ov = overall_efficiency(pos, par, day) * 100
        _scatter_field(ax, pos, ov, title, vmin=50, vmax=100)
        _draw_tower(ax, size=18)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Optimised Field Layouts – DE  |  Quetta, Pakistan  |  "
        f"Avg DNI {avg_dni:.0f} W/m²",
        fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_optimised_layouts.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 5a – Optimised Radial Staggered layout  (standalone single panel)
# ──────────────────────────────────────────────────────────────────────────────

def plot_rs_alone(rs_result, design_dni: dict):
    """
    Single-panel scatter map of the optimised Radial Staggered field,
    coloured by overall efficiency at the Vernal Equinox (11 AM).
    Annual mean efficiency (averaged across all four design days) is
    computed from the actual efficiency arrays and shown in the title.
    """
    params  = rs_result.best_params
    pos     = radial_staggered_layout(params, max_radius=600)
    avg_dni = _avg_design_dni(design_dni)

    # Per-day efficiencies for the annual mean label
    dp_list  = list(DESIGN_POINTS.items())
    day_effs = [overall_efficiency(pos, params, info["day"]) * 100
                for _, info in dp_list]
    annual_mean = float(np.mean([e.mean() for e in day_effs]))

    # Display: Vernal Equinox
    vernal_day = DESIGN_POINTS["Vernal Equinox"]["day"]
    ov         = overall_efficiency(pos, params, vernal_day) * 100
    mean_eff   = float(np.mean(ov))
    elev       = np.degrees(solar_elevation(vernal_day))

    fig, ax = plt.subplots(figsize=(8, 7))
    _scatter_field(ax, pos, ov,
                   f"Optimised Radial Staggered Layout\n"
                   f"Vernal Equinox  (Day {vernal_day})  –  Solar Elev. {elev:.1f}°\n"
                   f"Mean η (this day) = {mean_eff:.2f}%    "
                   f"Annual mean η = {annual_mean:.2f}%",
                   vmin=0, vmax=100,
                   cbar_label="Overall Efficiency (%)")
    _draw_tower(ax, size=18)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"DE-Optimised RS Field  |  TH={params.tower_height:.0f} m  "
        f"LH={params.heliostat_length:.1f} m  "
        f"WR={params.width_ratio:.2f}  DS={params.security_dist:.3f}\n"
        f"N = {len(pos):,} heliostats  |  Avg DNI {avg_dni:.0f} W/m²  |  Quetta, Pakistan",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, "fig5a_rs_alone.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 5b – RS and FS overlaid on one axes
# ──────────────────────────────────────────────────────────────────────────────

def plot_rs_fs_overlaid(rs_result, fs_result, design_dni: dict):
    """
    Single axes with both the optimised Radial Staggered (circles, ●) and
    Fermat Spiral (triangles, ▲) fields plotted together.
    Both are coloured by overall efficiency at the Vernal Equinox.
    Shared colour scale; mean efficiency computed from actual arrays and
    shown in the legend for each layout.  Tower icon at origin.
    """
    rs_params = rs_result.best_params
    fs_params = fs_result.best_params
    rs_pos    = radial_staggered_layout(rs_params, max_radius=600)
    fs_pos    = fermat_spiral_layout(fs_params, n_heliostats=1300)

    avg_dni    = _avg_design_dni(design_dni)
    vernal_day = DESIGN_POINTS["Vernal Equinox"]["day"]

    rs_ov = overall_efficiency(rs_pos, rs_params, vernal_day) * 100
    fs_ov = overall_efficiency(fs_pos, fs_params, vernal_day) * 100

    rs_mean = float(np.mean(rs_ov))
    fs_mean = float(np.mean(fs_ov))

    vmin = float(min(rs_ov.min(), fs_ov.min()))
    vmax = float(max(rs_ov.max(), fs_ov.max()))

    fig, ax = plt.subplots(figsize=(9, 8))

    sc_rs = ax.scatter(rs_pos[:, 0], rs_pos[:, 1],
                       c=rs_ov, cmap=CMAP_EFF, s=8,
                       vmin=vmin, vmax=vmax,
                       marker="o", linewidths=0,
                       label=f"Radial Staggered  (mean η = {rs_mean:.2f}%,  N={len(rs_pos):,})")
    ax.scatter(fs_pos[:, 0], fs_pos[:, 1],
               c=fs_ov, cmap=CMAP_EFF, s=10,
               vmin=vmin, vmax=vmax,
               marker="^", linewidths=0,
               label=f"Fermat's Spiral  (mean η = {fs_mean:.2f}%,  N={len(fs_pos):,})")

    plt.colorbar(sc_rs, ax=ax, label="Overall Efficiency (%)", fraction=0.04, pad=0.04)

    ax.set_xlabel("Distance from Tower (m)")
    ax.set_ylabel("Distance from Tower (m)")
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.axvline(0, color="k", lw=0.4, ls="--")

    r = float(np.abs(np.vstack([rs_pos, fs_pos])).max()) * 1.05
    for txt, xy in [("N", (0, 1)), ("S", (0, -1)), ("E", (1, 0)), ("W", (-1, 0))]:
        ax.text(xy[0]*r*0.92, xy[1]*r*0.92, txt, ha="center", va="center",
                fontsize=7, color="gray")

    _draw_tower(ax, size=18)
    ax.legend(fontsize=8, loc="upper right")

    elev = np.degrees(solar_elevation(vernal_day))
    ax.set_title(
        f"RS ● vs FS ▲ – Overall Efficiency Overlay\n"
        f"Vernal Equinox  (Day {vernal_day})  –  Solar Elev. {elev:.1f}°  |  "
        f"Avg DNI {avg_dni:.0f} W/m²",
        fontsize=10)
    fig.suptitle(
        "Optimised Layouts Overlaid – Radial Staggered & Fermat's Spiral\n"
        "Quetta, Pakistan  |  DE-Optimised",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, "fig5b_rs_fs_overlaid.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Fig 6 – DE convergence curves (both layouts overlaid, 100 generations)
# ──────────────────────────────────────────────────────────────────────────────

def plot_convergence(rs_result, fs_result):
    """
    DE convergence for Radial Staggered and Fermat Spiral on a single axes.
    x-axis always spans 0 → max_generations (padded in de_optimizer.py).
    """
    rs_hist = np.array(rs_result.convergence_history)
    fs_hist = np.array(fs_result.convergence_history)

    rs_gens = np.arange(len(rs_hist))
    fs_gens = np.arange(len(fs_hist))

    colour_rs = "#d62728"
    colour_fs = "#1f77b4"

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(rs_gens, rs_hist, color=colour_rs, lw=1.8, label="Radial Staggered")
    ax.fill_between(rs_gens, rs_hist, alpha=0.12, color=colour_rs)

    ax.plot(fs_gens, fs_hist, color=colour_fs, lw=1.8,
            linestyle="--", label="Fermat's Spiral")
    ax.fill_between(fs_gens, fs_hist, alpha=0.12, color=colour_fs)

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
    ax.set_ylabel("Best η  (annual mean efficiency, %)", fontsize=9)
    ax.set_title("DE Convergence – Radial Staggered & Fermat's Spiral\n"
                 "Quetta, Pakistan  |  pop=30 · F=0.8 · CR=0.7 · max_gen=100",
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

    ax = axes[0]
    bars1 = ax.bar(x - w/2, [before_rs, before_fs], w, label="Before DE",
                   color=colours[0], alpha=0.85)
    bars2 = ax.bar(x + w/2, [after_rs,  after_fs],  w, label="After DE",
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

# NOTE: plot_dni_data (Fig 8) has been intentionally removed from this pipeline.
