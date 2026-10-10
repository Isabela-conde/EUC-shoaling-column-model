# euc_column_kpp.jl
# -----------------------------------------------------------------------------
# 1-D equatorial column (OceanTurb.jl KPP + LMD94 interior shear mixing) forced
# like ACCESS-OM2 at 0°, 140°W, used to split warming-induced EUC shoaling into
#   PGF route    : pressure-gradient profile changes, stratification fixed
#   mixing route : stratification changes (suppresses momentum mixing), PGF fixed
#
#   dU/dt = PGF(z) - w dU/dz - U/τ_d + d/dz(ν dU/dz),      top flux τ/ρ₀
#   dT/dt = (T*(z) - T)/τ_T + d/dz(κ dT/dz) + KPP nonlocal,  top flux λ_fb (SST - SST*)/(ρ₀ cp)
#   dS/dt = (S*(z) - S)/τ_T + ...
#
# No restoring on U (that would pin the core). The Rayleigh term -U/τ_d stands in
# for horizontal advection / meridional divergence; its size is reported.
#
# Inputs: euc_column_inputs_140W.nc from extract_column_inputs.py
#         (runs "ctrl" and "warm": T, S, u, w, pgf on st_ocean; tau_x, sst scalars)
# -----------------------------------------------------------------------------

using OceanTurb
using NCDatasets
using Printf
using PyPlot
@use_pyplot_utils

const KPP = OceanTurb.KPP

# ============================================================================ #
# 1. LMD94 interior shear-instability mixing, missing from OceanTurb's KPP.
#    MOM5/ACCESS: ν = ν₀ (1 - (Ri/Ri₀)²)³ for 0 < Ri < Ri₀, ν₀ for Ri < 0, 0 above.
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
const INPUT  = "euc_column_inputs_140W.nc"
constants    = Constants(f = 0.0, α = 2.0e-4, β = 8.0e-4, ρ₀ = 1026.0, g = 9.81)
const cp     = 3991.0

N, H    = 160, 400.0            # 2.5 m cells, 400 m deep
Δt      = 10minute
tfinal  = 360day
τ_d     = 30day                 # Rayleigh drag on U  (sensitivity: 15, 60 day)
τ_T     = 20day                 # restoring of T, S toward ACCESS profile
λ_fb    = 40.0                  # SST feedback, W m⁻² K⁻¹
use_w   = true                  # include -w dU/dz
kpp_params = KPP.Parameters(KU₀ = 1e-4, KT₀ = 1e-5, KS₀ = 1e-5)   # MOM5 backgrounds

# ============================================================================ #
# 3. Load ACCESS inputs and interpolate onto the column grid
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
    zin  = -Float64.(ds["st_ocean"][:])                 # negative, descending
    ord  = sortperm(zin)                                # ascending for interp
    prof = Dict{String, Any}()
    for (ir, r) in enumerate(runs)
        get(v) = interp1(zc, zin[ord], Float64.(nomissing(ds[v][:, ir], NaN))[ord])
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

# ============================================================================ #
# 4. One column experiment
# ============================================================================ #
function run_column(; pgf_run = "ctrl", strat_run = "ctrl", tau_run = "ctrl",
                      τ_d = τ_d, τ_T = τ_T, use_w = use_w, tfinal = tfinal, verbose = true)

    pgf  = prof[pgf_run].pgf
    wz   = use_w ? prof[strat_run].w : zeros(N)        # w belongs to the dynamical state; keep with strat
    Ttgt = prof[strat_run].T
    Stgt = prof[strat_run].S
    SST★ = prof[strat_run].sst
    τx   = prof[tau_run].tau                            # negative = easterly in ACCESS

    function forcing_U(m, i)
        f = pgf[i] - m.solution.U[i] / τ_d
        if 1 < i < N
            f -= wz[i] * (m.solution.U[i+1] - m.solution.U[i-1]) / (2Δz)
        end
        return f
    end
    forcing_T(m, i) = (Ttgt[i] - m.solution.T[i]) / τ_T
    forcing_S(m, i) = (Stgt[i] - m.solution.S[i]) / τ_T

    # OceanTurb: positive top flux removes the quantity. Easterly stress (τx < 0 in
    # ACCESS, eastward-positive) must remove eastward momentum → Qᵘ = -τx/ρ₀ > 0.
    Qᵘ    = -τx / constants.ρ₀
    Qθ(m) = λ_fb * (m.solution.T[N] - SST★) / (constants.ρ₀ * cp)

    bcs = KPP.ModelBoundaryConditions(
        U = FieldBoundaryConditions(FluxBoundaryCondition(0.0), FluxBoundaryCondition(Qᵘ)),
        T = FieldBoundaryConditions(FluxBoundaryCondition(0.0), FluxBoundaryCondition(Qθ)),
    )

    model = KPP.Model(grid = grid, constants = constants, parameters = kpp_params,
                      stepper = :BackwardEuler, bcs = bcs,
                      forcing = KPP.Forcing(U = forcing_U, T = forcing_T, S = forcing_S))

    # start from the ACCESS state of the stratification run
    u0 = prof[strat_run].u
    model.solution.U = z -> u0[argmin(abs.(zc .- z))]
    model.solution.V = z -> 0.0
    model.solution.T = z -> Ttgt[argmin(abs.(zc .- z))]
    model.solution.S = z -> Stgt[argmin(abs.(zc .- z))]

    nsteps = Int(round(tfinal / Δt))
    t = Float64[]; zcore_t = Float64[]
    for n in 1:nsteps
        run_until!(model, Δt, n * Δt)
        if n % Int(round(1day / Δt)) == 0
            push!(t, model.clock.time); push!(zcore_t, core_depth(model))
        end
    end

    U   = [model.solution.U[i] for i in 1:N]
    T   = [model.solution.T[i] for i in 1:N]
    νf  = [i == 1 || i == N + 1 ? NaN : KPP.KU(model, i) for i in 1:N+1]
    Rif = [i == 1 || i == N + 1 ? NaN : local_richardson(model, i) for i in 1:N+1]

    # momentum budget at cell centres (steady: should sum ≈ 0)
    fric = fill(NaN, N); adv = fill(NaN, N)
    for i in 2:N-1
        fric[i] = (νf[i+1] * (U[i+1] - U[i]) - νf[i] * (U[i] - U[i-1])) / Δz^2
        adv[i]  = -wz[i] * (U[i+1] - U[i-1]) / (2Δz)
    end
    ray = -U ./ τ_d

    zcr, Ucr = core_depth(model; return_speed = true)
    verbose && @printf("pgf=%-5s strat=%-5s | core %6.1f m  U %5.3f m/s  | ∫PGF %+.2e  ∫Ray %+.2e  τ/ρ %+.2e  (m²/s²)\n",
                       pgf_run, strat_run, zcr, Ucr, sum(pgf) * Δz, sum(ray) * Δz, τx / constants.ρ₀)
    return (; U, T, νf, Rif, pgf, fric, adv, ray, zcore = zcr, Ucore = Ucr, t, zcore_t, zc, zf)
end

# core depth from a 1 m-interpolated profile, searched between 20 and 300 m
function core_depth(model; return_speed = false)
    U  = [model.solution.U[i] for i in 1:N]
    zq = collect(-300.0:1.0:-20.0)
    uq = interp1(zq, zc, U)
    k  = argmax(uq)
    return return_speed ? (zq[k], uq[k]) : zq[k]
end

# ============================================================================ #
# 5. Experiments
# ============================================================================ #
runs = Dict(
    "ctrl"  => run_column(pgf_run = "ctrl", strat_run = "ctrl"),
    "pgf"   => run_column(pgf_run = "warm", strat_run = "ctrl"),
    "strat" => run_column(pgf_run = "ctrl", strat_run = "warm"),
    "both"  => run_column(pgf_run = "warm", strat_run = "warm"),
)

# ACCESS core depths for comparison (same algorithm on the ACCESS u profiles)
access_core(r) = (zq = collect(-300.0:1.0:-20.0); uq = interp1(zq, zc, prof[r].u); zq[argmax(uq)])
zA_c, zA_w = access_core("ctrl"), access_core("warm")

println("\n=== EUC core depth (m) ===")
@printf("ACCESS   ctrl %6.1f   warm %6.1f   Δ %+5.1f\n", zA_c, zA_w, zA_w - zA_c)
for k in ("ctrl", "pgf", "strat", "both")
    @printf("column   %-5s %6.1f   Δ vs ctrl %+5.1f\n", k, runs[k].zcore, runs[k].zcore - runs["ctrl"].zcore)
end
Δp, Δs, Δb = (runs[k].zcore - runs["ctrl"].zcore for k in ("pgf", "strat", "both"))
@printf("additivity: pgf + strat = %+5.1f  vs  both = %+5.1f\n", Δp + Δs, Δb)

# ============================================================================ #
# 6. Figure
# ============================================================================ #
fig, ax = subplots(1, 4, figsize = (17, 5.5), sharey = true)
cols = Dict("ctrl" => "k", "pgf" => "C0", "strat" => "C3", "both" => "C2")

# (a) u profiles
for k in ("ctrl", "pgf", "strat", "both")
    ax[1].plot(runs[k].U, zc, cols[k], label = "column $k")
end
ax[1].plot(prof["ctrl"].u, zc, "k--", lw = 1, label = "ACCESS ctrl")
ax[1].plot(prof["warm"].u, zc, "C2--", lw = 1, label = "ACCESS warm")
ax[1].axvline(0, color = "0.6", lw = 0.6); ax[1].set_xlabel("u (m/s)"); ax[1].set_ylabel("z (m)")
ax[1].set_title("zonal velocity"); ax[1].legend(fontsize = 7); ax[1].set_ylim(-300, 0)

# (b) Δu relative to column control
for k in ("pgf", "strat", "both")
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

# (d) momentum budget, control
r = runs["ctrl"]
ax[4].plot(r.pgf,  zc, "C0", label = "PGF")
ax[4].plot(r.fric, zc, "C3", label = "∂z(ν ∂z u)")
ax[4].plot(r.adv,  zc, "C1", label = "−w ∂z u")
ax[4].plot(r.ray,  zc, "0.5", label = "−u/τ_d")
ax[4].plot(r.pgf .+ r.fric .+ r.adv .+ r.ray, zc, "k:", label = "sum")
ax[4].axvline(0, color = "0.6", lw = 0.6); ax[4].set_xlabel("m s⁻²"); ax[4].set_title("budget (control)")
ax[4].legend(fontsize = 7)

suptitle(@sprintf("1-D KPP column at 0°,140°W — core: ctrl %.0f m, pgf %+.0f, strat %+.0f, both %+.0f  (ACCESS %+.0f m)",
                  runs["ctrl"].zcore, Δp, Δs, Δb, zA_w - zA_c))
tight_layout(); savefig("euc_column_kpp.png", dpi = 150); println("saved euc_column_kpp.png")

# steady-state check
fig2, ax2 = subplots(figsize = (6, 3.5))
for k in ("ctrl", "pgf", "strat", "both")
    ax2.plot(runs[k].t ./ day, runs[k].zcore_t, cols[k], label = k)
end
ax2.set_xlabel("day"); ax2.set_ylabel("core depth (m)"); ax2.legend(fontsize = 8)
tight_layout(); savefig("euc_column_spinup.png", dpi = 150)
