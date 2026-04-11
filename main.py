"""
main.py
-------
End-to-end pipeline – Differential Evolution optimisation of heliostat fields.

Steps
-----
1. Load DNI data from CSV
2. Compute unoptimised efficiency scalars (for Fig 7 and new unoptimised figures)
3. Run DE optimisation on Radial Staggered layout (100 generations)
4. Run DE optimisation on Fermat Spiral layout    (100 generations)
5. Produce figures:
     Fig 1b – Unoptimised RS + FS layouts, 1×2 side-by-side with tower icon ★
     Fig 2u – Unoptimised Radial Staggered, 2×2 grid (4 design days)
     Fig 2  – Optimised Radial Staggered, 2×2 grid, avg DNI (4 design days)
     Fig 4  – Optimised Fermat Spiral,    2×2 grid, avg DNI (power/heliostat)
     Fig 5  – Optimised layouts side-by-side (RS + FS) with tower icon
     Fig 6  – DE convergence curves (both layouts overlaid, 100 generations)
     Fig 7  – Before/After efficiency comparison bar chart
6. Print summary table (Table 6 equivalent)

Removed from this pipeline:
  Fig 8 – DNI dataset analysis  (removed per request)
  Fig 1 – plot_attenuation_rs() (unoptimised RS baseline, available in plotting.py)
  Fig 3 – plot_attenuation_fs() (unoptimised FS baseline, available in plotting.py)

Run from the project root:
    python main.py
    python main.py --csv path/to/your_data.csv
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from solar_geometry   import load_dni_data, average_design_point_dni, DESIGN_POINTS
from heliostat_field  import FieldParams, radial_staggered_layout, fermat_spiral_layout
from efficiency       import annual_mean_efficiency, field_mean_efficiency, field_total_power_mw
from de_optimizer     import differential_evolution
from plotting         import (
    plot_unoptimised_layouts_1x2,   # Fig 1b – unoptimised RS+FS 1×2 with tower icon
    plot_unoptimised_rs_2x2,        # Fig 2u – unoptimised RS 2×2 (4 design days)
    plot_cosine_4panel_rs,          # Fig 2  – optimised RS 2×2 (avg DNI)
    plot_power_4panel_fs,           # Fig 4  – optimised FS 2×2 (power, avg DNI)
    plot_optimised_layouts,         # Fig 5  – optimised layouts side-by-side
    plot_rs_alone,                  # Fig 5a – optimised RS standalone single panel
    plot_rs_fs_overlaid,            # Fig 5b – RS + FS overlaid on one axes
    plot_convergence,               # Fig 6  – DE convergence
    plot_efficiency_comparison,     # Fig 7  – before/after efficiency bars
    # plot_attenuation_rs,          # Fig 1  – REMOVED (unoptimised RS baseline)
    # plot_attenuation_fs,          # Fig 3  – REMOVED (unoptimised FS baseline)
    # plot_dni_data,                # Fig 8  – REMOVED (DNI analysis)
)

_HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "solar-measurementspakistanquettawb-esmapqc.csv")
OUT_DIR  = os.path.join(_HERE, "outputs")


# ─── 1. Load DNI data ─────────────────────────────────────────────────────────

def load_data(csv_path: str = CSV_PATH):
    print("Loading DNI data …")
    df         = load_dni_data(csv_path)
    design_dni = average_design_point_dni(df)

    paper_vals = {
        "Vernal Equinox":  858.47,
        "Summer Solstice": 965.64,
        "Autumnal Equinox":875.71,
        "Winter Solstice": 856.63,
    }
    print("  Design-point DNI values (W/m²):")
    for k, v in design_dni.items():
        print(f"    {k:22s}: {v:.2f}  (paper: {paper_vals[k]:.2f})")
    return df, design_dni


# ─── 2. Unoptimised efficiency scalars (for Fig 7 and unoptimised figures) ────

def compute_unopt_efficiency():
    """
    Return annual-mean efficiency and heliostat count for the unoptimised
    (default FieldParams) RS and FS layouts.
    Used for Fig 7 before/after comparison and to drive the new unoptimised
    figures (Fig 1b and Fig 2u).  No optimisation maps are generated here.
    """
    base   = FieldParams()
    rs_pos = radial_staggered_layout(base, max_radius=450)
    fs_pos = fermat_spiral_layout(base, n_heliostats=1300)

    rs_eff = annual_mean_efficiency(rs_pos, base)
    fs_eff = annual_mean_efficiency(fs_pos, base)

    print(f"  Unoptimised baseline  RS: {len(rs_pos)} heliostats, η={rs_eff:.2f}%")
    print(f"  Unoptimised baseline  FS: {len(fs_pos)} heliostats, η={fs_eff:.2f}%")
    return base, rs_eff, fs_eff, len(rs_pos), len(fs_pos)


# ─── 3. DE Optimisation ───────────────────────────────────────────────────────

def run_optimisation(design_dni: dict, base: FieldParams):
    print("\n" + "═"*60)
    print("  Running DE Optimisation – Radial Staggered  (100 generations)")
    print("═"*60)
    rs_result = differential_evolution(
        layout_type     = "radial_staggered",
        design_dni      = design_dni,
        base_params     = base,
        pop_size        = 30,
        max_generations = 100,
        F               = 0.8,
        CR              = 0.7,
        tol             = 1e-4,
        seed            = 42,
        verbose         = True,
    )

    print("\n" + "═"*60)
    print("  Running DE Optimisation – Fermat's Spiral   (100 generations)")
    print("═"*60)
    fs_result = differential_evolution(
        layout_type     = "fermat_spiral",
        design_dni      = design_dni,
        base_params     = base,
        pop_size        = 30,
        max_generations = 100,
        F               = 0.8,
        CR              = 0.7,
        tol             = 1e-4,
        seed            = 42,
        verbose         = True,
    )
    return rs_result, fs_result


# ─── 4. Summary table (Table 6 equivalent) ────────────────────────────────────

def print_summary_table(rs_eff_unopt, rs_result,
                        fs_eff_unopt, fs_result,
                        n_rs_before, n_fs_before):
    print("\n" + "═"*80)
    print("  SUMMARY TABLE – Differential Evolution Results")
    print("  (Replicates Table 6 from Haris et al. 2023, with DE instead of GA)")
    print("═"*80)
    header = (f"{'Layout':18s} | {'TH(m)':6s} | {'WR':5s} | {'LH(m)':6s} | "
              f"{'DS':5s} | {'η_before(%)':11s} | {'η_after(%)':10s} | "
              f"{'N_before':8s} | {'N_after':7s}")
    print(header)
    print("─" * len(header))

    for label, eff_b, result, n_b in [
        ("Radial Staggered", rs_eff_unopt, rs_result, n_rs_before),
        ("Fermat's Spiral",  fs_eff_unopt, fs_result, n_fs_before),
    ]:
        p = result.best_params
        print(f"{label:18s} | {p.tower_height:6.1f} | {p.width_ratio:5.2f} | "
              f"{p.heliostat_length:6.2f} | {p.security_dist:5.3f} | "
              f"{eff_b:11.2f} | {result.best_efficiency:10.2f} | "
              f"{n_b:8d} | {result.n_heliostats:7d}")
        imp  = result.best_efficiency - eff_b
        red  = n_b - result.n_heliostats
        sign = "+" if imp >= 0 else ""
        print(f"{'  → Improvement':18s} {'':6s} {'':5s} {'':6s} {'':5s} "
              f"{'':11s} {sign}{imp:.2f}% {'':8s} −{red}")

    print("═"*80)
    print(f"\n  NOTE: GA (original paper) → RS improved by 8.52%, FS by 14.62%")
    print(f"        DE (this work)       → compare above figures")


# ─── 5. Generate all active figures ───────────────────────────────────────────

def generate_figures(df, design_dni,
                     base_params,
                     rs_result, fs_result,
                     rs_eff_unopt, fs_eff_unopt,
                     n_rs_before, n_fs_before):
    print("\nGenerating figures …")

    # ── NEW ──────────────────────────────────────────────────────────────────
    # Fig 1b – Unoptimised RS + FS layouts side-by-side (1×2) with tower icon
    plot_unoptimised_layouts_1x2(params=base_params)

    # Fig 2u – Unoptimised Radial Staggered 2×2 (one panel per design day)
    plot_unoptimised_rs_2x2(params=base_params)

    # ── UPDATED (2×2 grid, avg DNI) ──────────────────────────────────────────
    # Fig 2 – Optimised Radial Staggered 2×2
    plot_cosine_4panel_rs(params=rs_result.best_params, design_dni=design_dni)

    # Fig 4 – Optimised Fermat Spiral 2×2 (power per heliostat, avg DNI)
    plot_power_4panel_fs(design_dni, params=fs_result.best_params)

    # ── UNCHANGED ─────────────────────────────────────────────────────────────
    # Fig 5 – Optimised layouts side-by-side with tower icon
    plot_optimised_layouts(rs_result, fs_result, design_dni)

    # Fig 5a – Optimised RS standalone (single panel, mean efficiency annotated)
    plot_rs_alone(rs_result, design_dni)

    # Fig 5b – RS and FS overlaid on one axes (mean efficiency per layout annotated)
    plot_rs_fs_overlaid(rs_result, fs_result, design_dni)

    # Fig 6 – DE convergence curves
    plot_convergence(rs_result, fs_result)

    # Fig 7 – Before/after efficiency + heliostat count
    plot_efficiency_comparison(
        rs_eff_unopt,          rs_result.best_efficiency,
        fs_eff_unopt,          fs_result.best_efficiency,
        n_rs_before,           rs_result.n_heliostats,
        n_fs_before,           fs_result.n_heliostats,
    )

    # Fig 8 – REMOVED (DNI dataset analysis) per request

    print(f"\n  All figures saved to: {os.path.abspath(OUT_DIR)}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Heliostat DE Optimisation")
    parser.add_argument("--csv", default=CSV_PATH,
                        help="Path to the DNI CSV file")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Heliostat Field Layout Optimisation – Differential Evolution║")
    print("║  Site: Quetta, Balochistan, Pakistan (30.18°N, 66.97°E)    ║")
    print("║  Target: 50 MW Central Receiver Solar Thermal Power Plant   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # 1. Load DNI data
    df, design_dni = load_data(args.csv)

    # 2. Unoptimised efficiency scalars + unoptimised base params
    print("\nComputing unoptimised efficiency scalars …")
    base, rs_eff_unopt, fs_eff_unopt, n_rs_before, n_fs_before = \
        compute_unopt_efficiency()

    # 3. DE optimisation
    rs_result, fs_result = run_optimisation(design_dni, base)

    # 4. Summary table
    print_summary_table(rs_eff_unopt, rs_result,
                        fs_eff_unopt, fs_result,
                        n_rs_before, n_fs_before)

    # 5. All active figures
    generate_figures(df, design_dni,
                     base,
                     rs_result, fs_result,
                     rs_eff_unopt, fs_eff_unopt,
                     n_rs_before, n_fs_before)

    print("\nDone ✓")


if __name__ == "__main__":
    main()
