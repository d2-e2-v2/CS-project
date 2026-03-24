# Heliostat Field Layout Optimisation – Differential Evolution

**Based on:** Haris et al., *"Genetic Algorithm Optimization of Heliostat Field Layout for the Design of a Central Receiver Solar Thermal Power Plant"*, Heliyon 9 (2023) e21488  
**Modification:** The original **Genetic Algorithm (GA)** is replaced with **Differential Evolution (DE)** — a more efficient, parameter-light evolutionary algorithm.

---

## Overview

This codebase designs and optimises a **50 MW Central Receiver Solar Thermal Power Plant** for **Quetta, Balochistan, Pakistan** (30.18°N, 66.97°E). Two heliostat field layouts are evaluated:

1. **Radial Staggered** – concentric-ring arrangement  
2. **Fermat's Spiral** – golden-angle sunflower distribution

The DE algorithm tunes four heliostat field parameters to maximise annual optical efficiency while targeting 50 MW thermal output.

---

## Project Structure

```
heliostat_de/
├── src/
│   ├── solar_geometry.py    # Solar position, DNI loading
│   ├── heliostat_field.py   # Layout generation (RS + FS)
│   ├── efficiency.py        # Cosine, attenuation, spillage, overall efficiency
│   ├── de_optimizer.py      # Differential Evolution algorithm
│   ├── plotting.py          # All 8 figures
│   └── main.py              # End-to-end pipeline
├── data/
│   └── quetta_solar.csv     # World Bank ESMAP Tier-2 DNI data
├── outputs/                 # Generated figures (created on run)
└── README.md
```

---

## Setup & Run

```bash
# Install dependencies (Python 3.9+)
pip install numpy scipy pandas matplotlib

# Run the full pipeline
cd heliostat_de
python src/main.py
```

**No API calls are made.** All computation is local and self-contained.

---

## Algorithm: Differential Evolution (DE/rand/1/bin)

### Why DE instead of GA?

| Property | Genetic Algorithm (paper) | Differential Evolution (this work) |
|---|---|---|
| Mutation | Random gene swap | `v = x_r1 + F·(x_r2 − x_r3)` |
| Crossover | Tournament selection | Binomial crossover (rate CR) |
| Parameters | Population, mutation rate, crossover | Population, F, CR |
| Convergence | Moderate | Generally faster on continuous problems |
| Implementation | More complex (encoding) | Simple vectorised operations |

### DE Parameters

| Parameter | Value | Description |
|---|---|---|
| `pop_size` | 30 | Population size (same as GA in paper) |
| `F` | 0.8 | Differential weight (mutation scale) |
| `CR` | 0.7 | Crossover probability |
| `max_generations` | 100 | Maximum iterations |
| `tol` | 1e-4 | Convergence tolerance (Δη per 5 gens) |
| `seed` | 42 | Reproducibility seed |

### Chromosome / Genes

```
[TH, LH, WR, DS]
```

| Gene | Symbol | Range | Description |
|---|---|---|---|
| Tower Height | TH | 50–300 m | Height of central receiver tower |
| Heliostat Length | LH | 5–20 m | Length of each heliostat mirror |
| Width-to-Length Ratio | WR | 1–2 | Aspect ratio of heliostat |
| Security Distance | DS | 0.1–0.5 | Minimum gap between heliostats |

### Fitness Function

```
fitness = annual_mean_efficiency − λ · |P_field − P_target| / P_target
```

Where:
- `annual_mean_efficiency` = mean of η over 4 design points (%)
- `P_target` = 50 MW
- `λ` = 5.0 (penalty weight)

---

## Efficiency Calculations

All equations replicate the paper exactly:

### 1. Cosine Efficiency (Eq. 5)
```
cos 2θᵢ = [(z₀−z₁)sin α − e₁ cos α sin A − n₁ cos α cos A]
           / √[(z₀−z₁)² + e₁² + n₁²]
```

### 2. Solar Elevation (Eq. 6)
```
α = arcsin(sin δ · sin φ + cos δ · cos ω · cos φ)
```

### 3. Atmospheric Attenuation (Eq. 14, Vittitoe & Biggs 1978)
```
τₐ = 0.99326 − 0.1046·S + 0.017·S² − 0.002845·S³    [S in km]
```

### 4. Spillage Factor (Eq. 12-13)
Gaussian approximation with σᵣ = 2.51 mrad

### 5. Overall Field Efficiency (Eq. 11)
```
η = cos ω · fᵦ · f_sp · f_at
```
where `fᵦ = 0.97` (blocking factor)

### 6. Power per Heliostat (Eq. 10)
```
P = I · ρ · cos ω · f_sp · f_sb · f_at · A_h    [W]
```
where `ρ = 0.88` (mirror reflectivity)

---

## Output Figures

| File | Description |
|---|---|
| `fig1_attenuation_radial_staggered.png` | Attenuation map, RS layout unoptimised |
| `fig2_cosine_4panel_radial_staggered.png` | Cosine + overall efficiency, 4 design points |
| `fig3_attenuation_fermat_spiral.png` | Attenuation map, FS layout unoptimised |
| `fig4_power_4panel_fermat.png` | Power variation, 4 seasons, FS layout |
| `fig5_optimised_layouts.png` | Optimised RS + FS side-by-side |
| `fig6_de_convergence.png` | DE convergence curves (both layouts) |
| `fig7_efficiency_comparison.png` | Before/after efficiency & heliostat count |
| `fig8_dni_analysis.png` | DNI dataset analysis from CSV |

---

## Design Points (Table 3 from paper)

| Solar Event | Day | Average DNI (W/m²) |
|---|---|---|
| Vernal Equinox | 80 (Mar 21) | 858.47 |
| Summer Solstice | 172 (Jun 21) | 965.64 |
| Autumnal Equinox | 266 (Sep 23) | 875.71 |
| Winter Solstice | 355 (Dec 21) | 856.63 |

---

## Site Information

- **Location:** BUITEMS, Quetta, Balochistan, Pakistan
- **Coordinates:** 30.1798°N, 66.9750°E
- **Altitude:** ~1,670 m ASL
- **DNI Range:** 1500–2750 W/m²/day (annual)
- **DNI Equipment:** Tier-2 Rotating Shadowband Irradiometer (installed 2015)
- **Data Source:** World Bank ESMAP ([energydata.info](https://energydata.info/dataset/pakistan-solar-radiation-measurement-data))

---

## API Calls

**This project makes zero external API calls.** All computation is:
- Local Python (NumPy, SciPy, Pandas, Matplotlib)
- DNI data read from the bundled CSV file
- No internet access required

---

## References

1. Haris et al. (2023). Genetic algorithm optimization of heliostat field layout. *Heliyon 9*, e21488. https://doi.org/10.1016/j.heliyon.2023.e21488
2. Storn & Price (1997). Differential Evolution. *J. Global Optimization 11*, 341–359. https://link.springer.com/article/10.1023/A:1008202821328#citeas
3. Siala & Elayeb (2001). Mathematical formulation of a no-blocking heliostat field. https://ideas.repec.org/a/eee/renene/v23y2001i1p77-92.html
4. Kistler (1986). DELSOL3 User's Manual. Sandia National Laboratories. 
5. Vittitoe & Biggs (1978). Terrestrial Propagation Loss. Sandia National Lab. https://www.sciencedirect.com/science/article/pii/S0038092X1100377X
