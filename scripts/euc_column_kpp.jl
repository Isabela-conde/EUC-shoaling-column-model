# euc_column_kpp.jl
# -----------------------------------------------------------------------------
# 1-D equatorial column (OceanTurb.jl KPP + LMD94 interior shear mixing) forced
# like ACCESS-OM2 at 0°, 140°W, used to split warming-induced EUC shoaling into
#   pgf    : PGF profile SHAPE changes (warm PGF rescaled to the control integral),
#            stress and stratification fixed
#   stress : stress and PGF magnitude change (warm τ, warm ∫PGF), shape and
#            stratification fixed   [the c_d-driven stress change]
#   strat  : stratification changes (restoring targets, SST, w), PGF and τ fixed
#   both   : everything from the warm run
#
#   dU/dt = PGF(z) - w dU/dz - U/τ_d + d/dz(ν dU/dz),      top flux τ/ρ₀
#   dT/dt = (T*(z) - T)/τ_T + d/dz(κ dT/dz) + KPP nonlocal,  top flux λ_fb (SST - SST*)/(ρ₀ cp)
#
# Jet metrics: argmax core depth is unreliable for flat-topped profiles, so the
# primary metrics are the eastward-flow centroid depth and the half-max band.
#
# Inputs: inputs/euc_column_inputs_140W.nc from extract_column_inputs.py
# -----------------------------------------------------------------------------

using OceanTurb
using NCDatasets
using Printf
using PyPlot
@use_pyplot_utils

const KPP = OceanTurb.KPP

# ============================================================================ #
# 1. LMD94 interior shear-instability mixing, missing from OceanTurb's KPP.
# ============================================================================ #
import OceanTurb.KPP: K_KPP, 𝒲_U, 𝒲_T, d
using OceanTurb.PacanowskiPhilander: local_richardson

const ν₀_sh  = 5.0e-3      # m² s⁻¹ (MOM5 default)
const κ₀_sh  = 5.0e-3
const Ri₀_sh = 0.7

@inline shear_K(Ri, K₀) = Ri < 0 ? K₀ : (Ri ≥ Ri₀_sh ? 0.0 : K₀ * (1 - (Ri / Ri₀_sh)^2)^3)

KPP.KU(m::KPP.Model, i) = K_KPP(m.state.h, 𝒲_U(m, i), d(m, i)) + m.parameters.KU₀ +
                          shear_K(local_richardson(m, i), ν₀_sh)
KPP.KT(m::KPP.Model, i) = K_KPP(m.state.h, 𝒲_T(m, i), d(m, i)) + m.parameters.KT₀ +
                          shear_K(local_richardson(m, i), κ₀_sh)

# ============================================================================ #
# 2. Settings
# ============================================================================ #
const INPUT  = "inputs/euc_column_inputs_140W.nc"
const FIGDIR = "figures"
constants    = Constants(f = 0.0, α = 2.0e-4, β = 8.0e-4, ρ₀ = 1026.0, g = 9.81)
const cp     = 3991.0

N, H    = 160, 400.0            # 2.5 m cells, 400 m deep
Δt      = 10minute
tfinal  = 360day
τ_d     = 60day                 # Rayleigh drag on U; sensitivity below
τ_T     = 20day                 # restoring of T, S toward ACCESS profile
λ_fb    = 40.0                  # SST feedback, W m⁻² K⁻¹
use_w   = true
kpp_params = KPP.Parameters(KU₀ = 1e-4, KT₀ = 1e-5, KS₀ = 1e-5)   # MOM5 backgrounds

# ============================================================================ #
# 3. Load ACCESS inputs
# ============================================================================ #
function interp1(zq, z, v)                     # z ascending, linear, clamped
    out = similar(zq)
    for (k, zz) in enumerate(zq)
        if zz <= z[1];        out[k] = v[1]
        elseif zz >= z[end];  out[k] = v[end]
        else
            j = searchsortedlast(z, zz)
            out[k] = v[j] + (v[j+1] - v[j]) * (zz - z[j]) / (z[j+1] - z[j])
        end
    end
    return out
end

function load_inputs(fn, zc)
    ds   = NCDataset(fn)
    runs = String.(ds["run"][:])
    zin  = -Float64.(ds["st_ocean"][:])
    ord  = sortperm(zin)
    prof = Dict{String, Any}()
    for (ir, r) in enumerate(runs)
        function get(v)
            val = Float64.(nomissing(ds[v][:, ir], NaN))[ord]
            ok  = isfinite.(val)
            return interp1(zc, zin[ord][ok], val[ok])
        end
        prof[r] = (T = get("T"), S = get("S"), u = get("u"), w = get("w"), pgf = get("pgf"),
                   tau = Float64(ds["tau_x"][ir]), sst = Float64(ds["sst"][ir]))
    end
    close(ds)
    return prof
end

grid = UniformGrid(N, H)
zc   = grid.zc
zf   = grid.zf
Δz   = H / N
prof = load_inputs(INPUT, zc)
for r in ("ctrl", "warm")
    haskey(prof, r) || error("run \"$r\" missing from $INPUT; it has $(collect(keys(prof))).")
end
mkpath(FIGDIR)

# ============================================================================ #
# 4. Jet metrics (shared by column output and ACCESS profiles)
# ============================================================================ #
function jet_metrics(U::AbstractVector, z::AbstractVector)
    zq = collect(-300.0:1.0:-1.0)
    uq = interp1(zq, z, U)
    k  = argmax(uq);  Umax = uq[k];  z_argmax = zq[k]
    m  = uq .> 0
    centroid = any(m) ? sum(zq[m] .* uq[m]) / sum(uq[m]) : NaN
    half = Umax / 2
    above = findall(i -> i > k && uq[i] < half, eachindex(uq))      # shallower than core (z larger)
    below = findall(i -> i < k && uq[i] < half, eachindex(uq))
    z_hi  = isempty(above) ? zq[end] : zq[first(above)]
    z_lo  = isempty(below) ? zq[1]   : zq[last(below)]
    us    = uq[end]
    cross = findall(i -> i > k && uq[i] <= 0, eachindex(uq))
    z0    = isempty(cross) ? NaN : zq[first(cross)]
    return (; Umax, z_argmax, centroid, z_hi, z_lo, thickness = z_hi - z_lo, u_surf = us, z_zero = z0)
end

# ============================================================================ #
# 5. One column experiment
# ============================================================================ #
function run_column(; pgf_run = "ctrl", strat_run = "ctrl", tau_run = "ctrl",
                      pgf_scale_to = nothing, τ_d = τ_d, τ_T = τ_T, use_w = use_w,
                      tfinal = tfinal, verbose = true, label = "")

    pgf = copy(prof[pgf_run].pgf)
    if pgf_scale_to !== nothing            # keep the SHAPE of pgf_run, the INTEGRAL of pgf_scale_to
        pgf .*= sum(prof[pgf_scale_to].pgf) / sum(pgf)
    end
    wz   = use_w ? prof[strat_run].w : zeros(N)
    Ttgt = prof[strat_run].T
    Stgt = prof[strat_run].S
    SST★ = prof[strat_run].sst
    τx   = prof[tau_run].tau

    function forcing_U(m, i)
        f = pgf[i] - m.solution.U[i] / τ_d
        if 1 < i < N
            f -= wz[i] * (m.solution.U[i+1] - m.solution.U[i-1]) / (2Δz)
        end
        return f
    end
    forcing_T(m, i) = (Ttgt[i] - m.solution.T[i]) / τ_T
    forcing_S(m, i) = (Stgt[i] - m.solution.S[i]) / τ_T

    Qᵘ    = -τx / constants.ρ₀                                  # positive top flux removes U
    Qθ(m) = λ_fb * (m.solution.T[N] - SST★) / (constants.ρ₀ * cp)

    bcs = KPP.ModelBoundaryConditions(
        U = FieldBoundaryConditions(FluxBoundaryCondition(0.0), FluxBoundaryCondition(Qᵘ)),
        T = FieldBoundaryConditions(FluxBoundaryCondition(0.0), FluxBoundaryCondition(Qθ)),
    )

    model = KPP.Model(grid = grid, constants = constants, parameters = kpp_params,
                      stepper = :BackwardEuler, bcs = bcs,
                      forcing = KPP.Forcing(U = forcing_U, T = forcing_T, S = forcing_S))

    u0 = prof[strat_run].u
    model.solution.U = z -> u0[argmin(abs.(zc .- z))]
    model.solution.V = z -> 0.0
    model.solution.T = z -> Ttgt[argmin(abs.(zc .- z))]
    model.solution.S = z -> Stgt[argmin(abs.(zc .- z))]

    nsteps = Int(round(tfinal / Δt))
    t = Float64[]; cen_t = Float64[]
    for n in 1:nsteps
        run_until!(model, Δt, n * Δt)
        if n % Int(round(1day / Δt)) == 0
            U = [model.solution.U[i] for i in 1:N]
            push!(t, model.clock.time); push!(cen_t, jet_metrics(U, zc).centroid)
        end
    end

    U   = [model.solution.U[i] for i in 1:N]
    T   = [model.solution.T[i] for i in 1:N]
    νf  = [i == 1 || i == N + 1 ? NaN : KPP.KU(model, i) for i in 1:N+1]
    Rif = [i == 1 || i == N + 1 ? NaN : local_richardson(model, i) for i in 1:N+1]

    fric = fill(NaN, N); adv = fill(NaN, N)
    for i in 2:N-1
        fric[i] = (νf[i+1] * (U[i+1] - U[i]) - νf[i] * (U[i] - U[i-1])) / Δz^2
        adv[i]  = -wz[i] * (U[i+1] - U[i-1]) / (2Δz)
    end
    ray = -U ./ τ_d

    jm = jet_metrics(U, zc)
    kc = argmin(abs.(zc .- jm.centroid))
    ray_frac = abs(ray[kc]) / max(abs(pgf[kc]), 1e-12)
    verbose && @printf("%-7s | Umax %5.3f  argmax %6.1f  centroid %6.1f  half-max %4.0f–%4.0f  u_surf %+.3f  | ∫PGF %+.2e τ/ρ %+.2e  Ray/PGF@centroid %.2f\n",
                       label, jm.Umax, jm.z_argmax, jm.centroid, jm.z_hi, jm.z_lo, jm.u_surf,
                       sum(pgf) * Δz, τx / constants.ρ₀, ray_frac)
    return (; U, T, νf, Rif, pgf, fric, adv, ray, jm, t, cen_t, ray_frac)
end

# ============================================================================ #
# 6. τ_d sensitivity on the control (pick the value, then run the experiments)
# ============================================================================ #
println("\n=== τ_d sensitivity (control) ===")
A_c = jet_metrics(prof["ctrl"].u, zc)
@printf("ACCESS  | Umax %5.3f  argmax %6.1f  centroid %6.1f  half-max %4.0f–%4.0f  u_surf %+.3f\n",
        A_c.Umax, A_c.z_argmax, A_c.centroid, A_c.z_hi, A_c.z_lo, A_c.u_surf)
for τd in (30day, 60day, 90day, 180day)
    run_column(τ_d = τd, label = @sprintf("τ_d %3.0fd", τd / day))
end

# ============================================================================ #
# 7. Experiments
# ============================================================================ #
println("\n=== experiments (τ_d = $(τ_d / day) d) ===")
runs = Dict(
    "ctrl"   => run_column(label = "ctrl"),
    "pgf"    => run_column(pgf_run = "warm", pgf_scale_to = "ctrl", label = "pgf"),      # shape only
    "stress" => run_column(pgf_run = "warm", tau_run = "warm", label = "stress"),         # warm τ + warm ∫PGF (shape too)
    "strat"  => run_column(strat_run = "warm", label = "strat"),
    "both"   => run_column(pgf_run = "warm", tau_run = "warm", strat_run = "warm", label = "both"),
)
# note: "stress" − "pgf" isolates the magnitude (c_d) effect; "pgf" isolates the shape.

A_w = jet_metrics(prof["warm"].u, zc)
println("\n=== changes vs control: Δcentroid (m, + = shallower), Δhalf-max top, Δu_surf ===")
@printf("ACCESS   Δcentroid %+5.1f  Δz_hi %+5.1f  Δu_surf %+.3f\n",
        A_w.centroid - A_c.centroid, A_w.z_hi - A_c.z_hi, A_w.u_surf - A_c.u_surf)
c0 = runs["ctrl"].jm
for k in ("pgf", "stress", "strat", "both")
    j = runs[k].jm
    @printf("column %-6s Δcentroid %+5.1f  Δz_hi %+5.1f  Δu_surf %+.3f\n",
            k, j.centroid - c0.centroid, j.z_hi - c0.z_hi, j.u_surf - c0.u_surf)
end
Δ(k) = runs[k].jm.centroid - c0.centroid
@printf("additivity: pgf + (stress − pgf) + strat = %+5.1f  vs both = %+5.1f\n",
        Δ("stress") + Δ("strat"), Δ("both"))

# ============================================================================ #
# 8. Figures
# ============================================================================ #
cols = Dict("ctrl" => "k", "pgf" => "C0", "stress" => "C4", "strat" => "C3", "both" => "C2")
order = ("pgf", "stress", "strat", "both")

fig, ax = subplots(1, 5, figsize = (21, 5.5))
ax[1].set_ylabel("z (m)")
for a in ax[1:4]; a.set_ylim(-300, 0); end

# (a) u profiles
for k in ("ctrl", order...)
    ax[1].plot(runs[k].U, zc, cols[k], label = "column $k")
end
ax[1].plot(prof["ctrl"].u, zc, "k--", lw = 1, label = "ACCESS ctrl")
ax[1].plot(prof["warm"].u, zc, "C2--", lw = 1, label = "ACCESS warm")
ax[1].axvline(0, color = "0.6", lw = 0.6); ax[1].set_xlabel("u (m/s)")
ax[1].set_title("zonal velocity"); ax[1].legend(fontsize = 7)

# (b) Δu vs column control
for k in order
    ax[2].plot(runs[k].U .- runs["ctrl"].U, zc, cols[k], label = k)
end
ax[2].plot(prof["warm"].u .- prof["ctrl"].u, zc, "k--", lw = 1, label = "ACCESS warm − ctrl")
ax[2].axvline(0, color = "0.6", lw = 0.6); ax[2].set_xlabel("Δu (m/s)"); ax[2].set_title("Δu vs control")
ax[2].legend(fontsize = 7)

# (c) viscosity
for k in ("ctrl", "strat")
    ax[3].semilogx(runs[k].νf, zf, cols[k], label = k)
end
ax[3].set_xlabel("ν (m²/s)"); ax[3].set_title("KPP + interior shear ν"); ax[3].legend(fontsize = 7)
ax[3].set_xlim(1e-5, 1e-1)

# (d) budget, control
r = runs["ctrl"]
ax[4].plot(r.pgf,  zc, "C0", label = "PGF")
ax[4].plot(r.fric, zc, "C3", label = "∂z(ν ∂z u)")
ax[4].plot(r.adv,  zc, "C1", label = "−w ∂z u")
ax[4].plot(r.ray,  zc, "0.5", label = "−u/τ_d")
ax[4].plot(r.pgf .+ r.fric .+ r.adv .+ r.ray, zc, "k:", label = "sum")
ax[4].axvline(0, color = "0.6", lw = 0.6); ax[4].set_xlabel("m s⁻²"); ax[4].set_title("budget (control)")
ax[4].legend(fontsize = 7)

# (e) bars: Δcentroid and Δu_surf by route, with ACCESS
labels = [order..., "ACCESS"]
dcen = [[Δ(k) for k in order]..., A_w.centroid - A_c.centroid]
dus  = [[runs[k].jm.u_surf - c0.u_surf for k in order]..., A_w.u_surf - A_c.u_surf]
x = 1:length(labels)
ax[5].bar(x .- 0.2, dcen, 0.4, color = "C0", label = "Δ centroid (m, + = up)")
ax5b = ax[5].twinx()
ax5b.bar(x .+ 0.2, dus .* 100, 0.4, color = "C3", label = "Δ u_surf (cm/s)")
ax[5].set_xticks(x); ax[5].set_xticklabels(labels, fontsize = 8)
ax[5].axhline(0, color = "0.6", lw = 0.6)
ax[5].set_ylabel("Δ centroid depth (m)", color = "C0"); ax5b.set_ylabel("Δ u_surf (cm/s)", color = "C3")
ax[5].set_title("by route")

suptitle(@sprintf("1-D KPP column at 0°,140°W (τ_d = %.0f d) — Δcentroid: pgf %+.0f, stress %+.0f, strat %+.0f, both %+.0f m   (ACCESS %+.0f m)",
                  τ_d / day, Δ("pgf"), Δ("stress"), Δ("strat"), Δ("both"), A_w.centroid - A_c.centroid))
tight_layout(); savefig(joinpath(FIGDIR, "euc_column_kpp.png"), dpi = 150)
println("saved $(joinpath(FIGDIR, "euc_column_kpp.png"))")

fig2, ax2 = subplots(figsize = (6, 3.5))
for k in ("ctrl", order...)
    ax2.plot(runs[k].t ./ day, runs[k].cen_t, cols[k], label = k)
end
ax2.set_xlabel("day"); ax2.set_ylabel("eastward-flow centroid (m)"); ax2.legend(fontsize = 8)
tight_layout(); savefig(joinpath(FIGDIR, "euc_column_spinup.png"), dpi = 150)
