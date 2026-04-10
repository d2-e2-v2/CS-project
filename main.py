"""
main.py
-------
End-to-end pipeline – Differential Evolution optimisation of heliostat fields.

Steps
-----
1. Load DNI data from CSV
2. Run DE optimisation on Radial Staggered layout (100 generations)
3. Run DE optimisation on Fermat Spiral layout    (100 generations)
4. Produce active figures:
     Fig 2 – Cosine + overall efficiency 4-panel (RS, optimised, 4 design days)
     Fig 4 – Power variation 4-panel              (FS, optimised, 4 design days)
     Fig 5 – Optimised layouts side-by-side
     Fig 6 – DE convergence curves (both layouts overlaid, 100 generations)
     Fig 7 – Before/After efficiency comparison bar chart
     Fig 8 – DNI dataset analysis
5. Print summary table replicating Table 6 from the paper

Removed from this pipeline (unoptimised GA baselines):
  Figs 1 & 3 – plot_attenuation_rs() / plot_attenuation_fs()
    These visualised the pre-optimisation field using default FieldParams,
    reproducing Figs 1 & 3 from Haris et al. (2023) where the GA paper showed
    the unoptimised attenuation maps.  Because this codebase replaces GA with
    DE and focuses on the optimised result, those calls have been removed.
    The functions still exist in plotting.py for standalone / reference use.

  compute_baselines() – also removed from the active pipeline.
    It computed annual_mean_efficiency for the default (unoptimised) FieldParams
    to serve as the "before" bar in Fig 7.  We now use a lightweight helper
    (compute_unopt_efficiency) that returns only the scalar efficiency values
    needed for the comparison chart, without generating full efficiency maps.

Run from the project root:
    python main.py
    python main.py --csv path/to/your_data.csv

No external API calls.  All computation is local.
"""

import sys
import os

# ── Ensure the project directory is on the path ───────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from solar_geometry   import load_dni_data, average_design_point_dni, DESIGN_POINTS
from heliostat_field  import FieldParams, radial_staggered_layout, fermat_spiral_layout
from efficiency       import annual_mean_efficiency, field_mean_efficiency, field_total_power_mw
from de_optimizer     import differential_evolution
from plotting         import (
    plot_cosine_4panel_rs,    # Fig 2 – RS layout, 4 design days  (optimised)
    plot_power_4panel_fs,     # Fig 4 – FS layout, 4 design days  (optimised)
    plot_optimised_layouts,   # Fig 5
    plot_convergence,         # Fig 6 – DE convergence, 100 generations
    plot_efficiency_comparison,  # Fig 7
    plot_dni_data,            # Fig 8
    # plot_attenuation_rs,    # Fig 1 – REMOVED: unoptimised RS baseline (GA paper)
    # plot_attenuation_fs,    # Fig 3 – REMOVED: unoptimised FS baseline (GA paper)
)

# ── Resolve paths ──────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "solar-measurementspakistanquettawb-esmapqc.csv")
OUT_DIR  = os.path.join(_HERE, "outputs")


# ─── 1. Load data ─────────────────────────────────────────────────────────────

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


# ─── 2. Unoptimised efficiency scalars (for Fig 7 "before" bars) ──────────────
#
#  NOTE: The full compute_baselines() function that generated efficiency maps
#  and per-design-point tables (Table 4 in the paper) has been removed from
#  the active pipeline.  It replicated the pre-optimisation (unoptimised GA
#  baseline) analysis from Haris et al. (2023).  The layout generation and
#  annual_mean_efficiency calls below are kept *only* to provide the scalar
#  "before DE" values required by Fig 7.
#
#  If you need the full unoptimised baseline analysis (Tables 3-4, Figs 1-3
#  from the GA paper) run plot_attenuation_rs() and plot_attenuation_fs()
#  directly from plotting.py, or reinstate compute_baselines() below.

def compute_unopt_efficiency():
    """
    Return annual-mean efficiency and heliostat count for the unoptimised
    (default FieldParams) RS and FS layouts.  Used only for the Fig 7
    before/after comparison.  No figures are generated here.
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
        imp = result.best_efficiency - eff_b
        red = n_b - result.n_heliostats
        sign = "+" if imp >= 0 else ""
        print(f"{'  → Improvement':18s} {'':6s} {'':5s} {'':6s} {'':5s} "
              f"{'':11s} {sign}{imp:.2f}% {'':8s} −{red}")

    print("═"*80)
    print(f"\n  NOTE: GA (original paper) → RS improved by 8.52%, FS by 14.62%")
    print(f"        DE (this work)       → compare above figures")


# ─── 5. Generate all active figures ───────────────────────────────────────────

def generate_figures(df, design_dni,
                     rs_result, fs_result,
                     rs_eff_unopt, fs_eff_unopt,
                     n_rs_before, n_fs_before):
    print("\nGenerating figures …")

    # Fig 1 – REMOVED (unoptimised RS attenuation map, GA paper baseline)
    # plot_attenuation_rs()
    # → Haris et al. 2023, Fig 1: pre-GA-optimisation attenuation map for
    #   Radial Staggered layout with default FieldParams.  Available via
    #   plot_attenuation_rs() in plotting.py if the unoptimised baseline
    #   comparison is needed.

    # Fig 2 – Cosine + overall efficiency, RS layout, 4 design days (optimised)
    plot_cosine_4panel_rs(params=rs_result.best_params)

    # Fig 3 – REMOVED (unoptimised FS attenuation map, GA paper baseline)
    # plot_attenuation_fs()
    # → Haris et al. 2023, Fig 3: pre-GA-optimisation attenuation map for
    #   Fermat Spiral layout with default FieldParams.  Available via
    #   plot_attenuation_fs() in plotting.py if the unoptimised baseline
    #   comparison is needed.

    # Fig 4 – Power per heliostat + overall efficiency, FS layout, 4 design days (optimised)
    plot_power_4panel_fs(design_dni, params=fs_result.best_params)

    # Fig 5 – Optimised field layouts side-by-side
    plot_optimised_layouts(rs_result, fs_result, design_dni)

    # Fig 6 – DE convergence curves, both layouts on one axes, 100 generations
    plot_convergence(rs_result, fs_result)

    # Fig 7 – Before/After efficiency + heliostat count comparison
    plot_efficiency_comparison(
        rs_eff_unopt,          rs_result.best_efficiency,
        fs_eff_unopt,          fs_result.best_efficiency,
        n_rs_before,           rs_result.n_heliostats,
        n_fs_before,           fs_result.n_heliostats,
    )

    # Fig 8 – DNI dataset analysis
    plot_dni_data(df)

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

    # 2. Unoptimised efficiency scalars (lightweight – no maps generated)
    print("\nComputing unoptimised efficiency scalars for Fig 7 …")
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
                     rs_result, fs_result,
                     rs_eff_unopt, fs_eff_unopt,
                     n_rs_before, n_fs_before)

    print("\nDone ✓")


if __name__ == "__main__":
    main()
