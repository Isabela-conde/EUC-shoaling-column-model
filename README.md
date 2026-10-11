# EUC shoaling: 1-D column experiments

Does surface warming shoal the Equatorial Undercurrent by moving the pressure
gradient (Peng et al. 2022) or through the stratification and the mixing above
the core (Saenko et al. 2011)? A single-column KPP model at 0°, 140°W, forced
with ACCESS-OM2 profiles, separates the two. Wind stress is held at its control
value in every run (fixed-wind experiments, as in Peng et al.).

Companion to the ACCESS-OM2 warming experiments (control / tropical warming).

## Layout

| path | what |
|---|---|
| `scripts/euc_scaling.py` | Pedlosky inertial-scaling test on the ACCESS runs (fractional trends, exponent fit) |
| `scripts/extract_column_inputs.py` (+ notebook) | builds `inputs/euc_column_inputs_140W.nc`: T, S, u, w, PGF profiles + τ, SST at 140°W for control and warming |
| `scripts/euc_column_kpp.jl` | the column experiments (OceanTurb.jl KPP + LMD94 interior shear mixing) and diagnostics |
| `inputs/` | NetCDF inputs (not tracked) |
| `figures/` | output figures |

## Column model

    dU/dt = PGF(z) − w dU/dz − U/τ_d + d/dz(ν dU/dz),   top flux τ/ρ₀
    dT/dt = (T*(z) − T)/τ_T + mixing,                   top flux λ_fb (SST − SST*)/(ρ₀ cp)

No restoring on U. The Rayleigh term stands in for horizontal advection and
meridional divergence; its size relative to the PGF at the core is printed.

OceanTurb's KPP has no interior shear-instability mixing; `euc_column_kpp.jl`
adds the LMD94 term used by MOM5/ACCESS (ν₀ = 5e-3 m² s⁻¹, Ri₀ = 0.7) and
MOM5's background values.

## Experiments

| run | PGF(z) | stratification (T*, S*, SST*, w) | isolates |
|---|---|---|---|
| ctrl | control | control | baseline, compare to ACCESS jet |
| pgf | warming | control | pressure-gradient route |
| strat | control | warming | stratification route |
| both | warming | warming | sum, compare to ACCESS shoaling |
| stratwc | control | warming T, S, SST; control w | check: role of the imposed w change |

## Diagnostics and figures

| figure | shows |
|---|---|
| `euc_column_kpp.png` | u(z) and Δu(z) by route vs ACCESS; ν(z); control budget; Δcentroid and Δu_surf by route |
| `euc_column_pgf.png` | PGF(z) control, warm, and the change |
| `euc_column_shear_layer.png` | shear layer above the jet (jet max → KPP h): u, N² and S², ν; layer-mean ΔN², ΔS², Ri, Δν in the title |
| `euc_column_ladder.png` | Δu_surf = Δu_core − Δ(drop across shear layer) − Δ(drop across mixed layer), column runs vs ACCESS |
| `euc_column_spinup.png` | centroid depth vs time |

The script also prints the shear-layer means, the ladder table, and a
mixed-layer momentum budget (τ/ρh, base flux, PGF, advection, Rayleigh) per run.

## Running

1. In `extract_column_inputs.py`, provide `DATASETS` and `load_vars` (must return
   `u, wt, pottemp, prac_salt, pot_rho_0, sea_level, tau_x, net_sfc_heating`).
   Run it on Gadi; copy the NetCDF to `inputs/`.
2. `julia --project scripts/euc_column_kpp.jl` from the repo root (edit `INPUT`
   path if needed). Needs OceanTurb, NCDatasets, PyPlot.

## Checks on first run

- control budget panel: if −u/τ_d rivals the PGF, test τ_d = 30–180 d (sensitivity is printed)
- control core depth within ~20 m of ACCESS, speed within ~30 %
- sign of τ: `tau_x` assumed eastward-positive (easterlies negative)
- `euc_column_spinup.png`: core depth flat well before the end of the run

## Background

Inertial scaling (H ∝ (τ/g′)^½, U ∝ (τg′)^¼) applied to the ACCESS runs: under
warming τ and g′ barely change and so do U and thickness, yet the core shoals.
Shoaling is a displacement, not an inertial adjustment; this repo asks what
displaces it, and what the shear layer above the core does as it rises.
