# euc_column_kpp.jl
# -----------------------------------------------------------------------------
# 1-D equatorial column (OceanTurb.jl KPP + LMD94 interior shear mixing) forced
# like ACCESS-OM2 at 0°, 140°W, used to split warming-induced EUC shoaling into
#   peng   : PGF change caused by the pycnocline shoaling, at FIXED stress.
#            = warm ΔPGF minus the part attributable to the stress drop, where the
#            latter is the wind-only run's ΔPGF scaled by Δτ_warm/Δτ_wind.
#   cd     : the drag-coefficient route: warm τ with the stress-attributable ΔPGF only.
#   strat  : stratification route — warm T, S, SST, w targets; PGF and τ fixed
#   pgfstr : warm PGF + warm τ as-is (= peng + cd by construction; consistency check)
#   both   : everything from the warm run
#   Requires runs "ctrl", "warm", "wind" (uas_015) in the input file.
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
tfinal  = 540day
τ_d     = 60day                 # Rayleigh drag on U; sensitivity below
τ_T     = 20day                 # restoring of T, S toward ACCESS profile
λ_fb    = 40.0                  # SST feedback, W m⁻² K⁻¹
use_w   = true
RUN_SHAPE_TEST = false        # optional: warm PGF shape rescaled to the control integral
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
for r in ("ctrl", "warm", "wind")
    haskey(prof, r) || error("run \"$r\" missing from $INPUT; it has $(collect(keys(prof))). Re-run the extractor with RUNS incl. \"wind\": \"uas_015\".")
end
mkpath(FIGDIR)

# ---- split the warm ΔPGF into a stress-attributable part and a pycnocline part ----
Δτ_ratio = (prof["warm"].tau - prof["ctrl"].tau) / (prof["wind"].tau - prof["ctrl"].tau)
ΔPGF_warm = prof["warm"].pgf .- prof["ctrl"].pgf
ΔPGF_wind = prof["wind"].pgf .- prof["ctrl"].pgf
ΔPGF_cd   = Δτ_ratio .* ΔPGF_wind              # PGF response to the stress drop (linear in Δτ)
ΔPGF_peng = ΔPGF_warm .- ΔPGF_cd               # PGF response to the pycnocline shoaling
pgf_peng  = prof["ctrl"].pgf .+ ΔPGF_peng
pgf_cd    = prof["ctrl"].pgf .+ ΔPGF_cd
@printf("\nstress: ctrl %+.4f  warm %+.4f  wind %+.4f N/m²  → Δτ_warm/Δτ_wind = %.2f\n",
        prof["ctrl"].tau, prof["warm"].tau, prof["wind"].tau, Δτ_ratio)
@printf("∫ΔPGF dz: warm %+.2e  stress-attributable %+.2e  pycnocline %+.2e  (m²/s²)\n",
        sum(ΔPGF_warm) * Δz, sum(ΔPGF_cd) * Δz, sum(ΔPGF_peng) * Δz)

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

# Mixed-layer depth from a T,S profile: first depth where the buoyancy deficit
# relative to the top cell exceeds Δb_crit (linear EOS, same α, β as the model).
function mld_from_profile(T, S, z; Δb_crit = 3e-4)
    g, α, β = constants.g, constants.α, constants.β
    b = g .* (α .* T .- β .* S)
    k = findlast(i -> b[end] - b[i] < Δb_crit, eachindex(z))   # z ascending, surface = end
    return k === nothing ? NaN : (k == 1 ? z[1] : z[k])
end

# ============================================================================ #
# 5. One column experiment
# ============================================================================ #
# Shear S² and buoyancy frequency N² on cell faces (face i sits between cells i-1 and i).
function shear_strat(U, T, S)
    S2 = fill(NaN, N + 1); N2 = fill(NaN, N + 1)
    for i in 2:N
        S2[i] = ((U[i] - U[i-1]) / Δz)^2
        N2[i] = constants.g * (constants.α * (T[i] - T[i-1]) - constants.β * (S[i] - S[i-1])) / Δz
    end
    return S2, N2
end

# Means over the shear layer above the jet: from the jet maximum up to the KPP boundary-layer base.
function shear_layer_means(U, S2f, N2f, Rif, νf, h_kpp)
    k_core = argmax(U)
    sel = [i for i in 2:N if zf[i] > zc[k_core] && zf[i] < h_kpp]
    isempty(sel) && return (; S2 = NaN, N2 = NaN, Ri = NaN, ν = NaN, z_top = h_kpp, z_bot = zc[k_core])
    m(x) = sum(x[sel]) / length(sel)
    return (; S2 = m(S2f), N2 = m(N2f), Ri = m(N2f) / m(S2f), ν = m(νf), z_top = h_kpp, z_bot = zc[k_core])
end

function run_column(; pgf_run = "ctrl", strat_run = "ctrl", tau_run = "ctrl",
                      pgf_scale_to = nothing, pgf_override = nothing, tau_override = nothing,
                      τ_d = τ_d, τ_T = τ_T, use_w = use_w, w_run = nothing,
                      tfinal = tfinal, verbose = true, label = "")

    pgf = pgf_override === nothing ? copy(prof[pgf_run].pgf) : copy(pgf_override)
    if pgf_scale_to !== nothing            # keep the SHAPE of pgf_run, the INTEGRAL of pgf_scale_to
        pgf .*= sum(prof[pgf_scale_to].pgf) / sum(pgf)
    end
    wz   = use_w ? prof[w_run === nothing ? strat_run : w_run].w : zeros(N)   # upwelling from w_run (default: strat_run)
    Ttgt = prof[strat_run].T
    Stgt = prof[strat_run].S
    SST★ = prof[strat_run].sst
    τx   = tau_override === nothing ? prof[tau_run].tau : tau_override

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
    S   = [model.solution.S[i] for i in 1:N]
    h_kpp = -model.state.h                                      # KPP boundary-layer depth (m, negative)
    mld   = mld_from_profile(T, S, zc)
    νf  = [i == 1 || i == N + 1 ? NaN : KPP.KU(model, i) for i in 1:N+1]
    Rif = [i == 1 || i == N + 1 ? NaN : local_richardson(model, i) for i in 1:N+1]
    S2f, N2f = shear_strat(U, T, S)                              # on faces, like νf and Rif
    sl  = shear_layer_means(U, S2f, N2f, Rif, νf, h_kpp)         # between jet max and KPP h

    fric = fill(NaN, N); adv = fill(NaN, N)
    for i in 2:N-1
        fric[i] = (νf[i+1] * (U[i+1] - U[i]) - νf[i] * (U[i] - U[i-1])) / Δz^2
        adv[i]  = -wz[i] * (U[i+1] - U[i-1]) / (2Δz)
    end
    ray = -U ./ τ_d

    jm = jet_metrics(U, zc)
    kc = argmin(abs.(zc .- jm.centroid))
    ray_frac = abs(ray[kc]) / max(abs(pgf[kc]), 1e-12)
    verbose && @printf("%-7s | Umax %5.3f  argmax %6.1f  centroid %6.1f  half-max %4.0f–%4.0f  u_surf %+.3f  | KPP h %5.1f  mld %5.1f  | ∫PGF %+.2e τ/ρ %+.2e  Ray/PGF@centroid %.2f\n",
                       label, jm.Umax, jm.z_argmax, jm.centroid, jm.z_hi, jm.z_lo, jm.u_surf,
                       h_kpp, mld, sum(pgf) * Δz, τx / constants.ρ₀, ray_frac)
    return (; U, T, S, νf, Rif, S2f, N2f, sl, pgf, fric, adv, ray, jm, t, cen_t, ray_frac, h_kpp, mld, τx, wz)
end

# ============================================================================ #
# 6. τ_d sensitivity on the control (pick the value, then run the experiments)
# ============================================================================ #
println("\n=== τ_d sensitivity (control) ===")
A_c = jet_metrics(prof["ctrl"].u, zc)
mld_A = Dict(r => mld_from_profile(prof[r].T, prof[r].S, zc) for r in ("ctrl", "warm"))
@printf("ACCESS  | Umax %5.3f  argmax %6.1f  centroid %6.1f  half-max %4.0f–%4.0f  u_surf %+.3f  | mld(Δb=3e-4) ctrl %5.1f  warm %5.1f\n",
        A_c.Umax, A_c.z_argmax, A_c.centroid, A_c.z_hi, A_c.z_lo, A_c.u_surf, mld_A["ctrl"], mld_A["warm"])
for τd in (30day, 60day, 90day, 180day)
    run_column(τ_d = τd, label = @sprintf("τ_d %3.0fd", τd / day))
end

# ============================================================================ #
# 7. Experiments
# ============================================================================ #
println("\n=== experiments (τ_d = $(τ_d / day) d) ===")
runs = Dict(
    "ctrl"   => run_column(label = "ctrl"),
    "peng"   => run_column(pgf_override = pgf_peng, label = "peng"),                            # pycnocline PGF shift, fixed τ
    "cd"     => run_column(pgf_override = pgf_cd, tau_run = "warm", label = "cd"),              # stress drop + its PGF
    "strat"  => run_column(strat_run = "warm", label = "strat"),                                # stratification
    "pgfstr" => run_column(pgf_run = "warm", tau_run = "warm", label = "pgfstr"),               # = peng + cd (check)
    "both"   => run_column(pgf_run = "warm", tau_run = "warm", strat_run = "warm", label = "both"),
    "stratwc" => run_column(strat_run = "warm", w_run = "ctrl", label = "stratwc"),              # stratification, control w
)
if RUN_SHAPE_TEST
    runs["pgf_shape"] = run_column(pgf_run = "warm", pgf_scale_to = "ctrl", label = "pgfshp")
end

A_w = jet_metrics(prof["warm"].u, zc)
println("\n=== changes vs control: Δcentroid (m, + = shallower), Δhalf-max top, Δu_surf, boundary layer ===")
@printf("ACCESS   Δcentroid %+5.1f  Δz_hi %+5.1f  Δu_surf %+.3f   mld ctrl %5.1f → warm %5.1f\n",
        A_w.centroid - A_c.centroid, A_w.z_hi - A_c.z_hi, A_w.u_surf - A_c.u_surf, mld_A["ctrl"], mld_A["warm"])
c0 = runs["ctrl"].jm
for k in ("peng", "cd", "strat", "pgfstr", "both")
    j = runs[k].jm
    @printf("column %-6s Δcentroid %+5.1f  Δz_hi %+5.1f  Δu_surf %+.3f   KPP h %5.1f  mld %5.1f\n",
            k, j.centroid - c0.centroid, j.z_hi - c0.z_hi, j.u_surf - c0.u_surf, runs[k].h_kpp, runs[k].mld)
end
Δ(k) = runs[k].jm.centroid - c0.centroid
@printf("additivity: peng + cd = %+5.1f vs pgfstr = %+5.1f  |  peng + cd + strat = %+5.1f vs both = %+5.1f\n",
        Δ("peng") + Δ("cd"), Δ("pgfstr"), Δ("peng") + Δ("cd") + Δ("strat"), Δ("both"))
@printf("fixed-stress warming response (peng + strat) = %+5.1f m centroid, %+.3f m/s u_surf\n",
        Δ("peng") + Δ("strat"),
        (runs["peng"].jm.u_surf - c0.u_surf) + (runs["strat"].jm.u_surf - c0.u_surf))
@printf("control: KPP h %5.1f m, column mld %5.1f m, ACCESS mld %5.1f m (same Δb criterion)\n",
        runs["ctrl"].h_kpp, runs["ctrl"].mld, mld_A["ctrl"])

# ============================================================================ #
# 8. Figures
# ============================================================================ #
cols = Dict("ctrl" => "k", "peng" => "C0", "cd" => "C4", "strat" => "C3", "both" => "C2", "pgfstr" => "0.6")
order = ("peng", "cd", "strat", "both")
route_name = Dict("peng" => "pycnocline PGF (warm PGF − stress part; ctrl τ, ctrl strat.)",
                  "cd"   => "drag-coef. stress (warm τ + its PGF; ctrl strat.)",
                  "strat" => "stratification (warm T, S, w restoring; ctrl PGF, ctrl τ)",
                  "both" => "all warm forcing (warm PGF + warm τ + warm strat.)",
                  "stratwc" => "stratification with ctrl w")

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
    ax[2].plot(runs[k].U .- runs["ctrl"].U, zc, cols[k], label = route_name[k])
end
ax[2].plot(prof["warm"].u .- prof["ctrl"].u, zc, "k--", lw = 1, label = "ACCESS warm − ctrl")
ax[2].axvline(0, color = "0.6", lw = 0.6); ax[2].set_xlabel("Δu (m/s)"); ax[2].set_title("Δu vs control")
ax[2].legend(fontsize = 7)

# (c) viscosity
for k in ("ctrl", "strat")
    ax[3].semilogx(runs[k].νf, zf, cols[k], label = k)
    ax[3].axhline(runs[k].h_kpp, color = cols[k], ls = ":", lw = 0.8)
end
ax[3].axhline(mld_A["ctrl"], color = "k", ls = "--", lw = 0.8, label = "ACCESS mld (ctrl)")
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
route_short = Dict("peng" => "pycnocline\nPGF", "cd" => "c_d stress\n(+ its PGF)", "strat" => "stratification", "both" => "all warm\nforcing")
labels = [[route_short[k] for k in order]..., "ACCESS\nwarm − ctrl"]
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

suptitle(@sprintf("1-D KPP column at 0°,140°W (τ_d = %.0f d, %.0f d) — Δcentroid: pycnocline PGF %+.0f, drag-coef. stress %+.0f, stratification %+.0f, both %+.0f m   (ACCESS %+.0f m)",
                  τ_d / day, tfinal / day, Δ("peng"), Δ("cd"), Δ("strat"), Δ("both"), A_w.centroid - A_c.centroid))
tight_layout(); savefig(joinpath(FIGDIR, "euc_column_kpp.png"), dpi = 150)
println("saved $(joinpath(FIGDIR, "euc_column_kpp.png"))")

fig2, ax2 = subplots(figsize = (6, 3.5))
for k in ("ctrl", order...)
    ax2.plot(runs[k].t ./ day, runs[k].cen_t, cols[k], label = k == "ctrl" ? "ctrl" : route_name[k])
end
ax2.set_xlabel("day"); ax2.set_ylabel("eastward-flow centroid (m)"); ax2.legend(fontsize = 8)
tight_layout(); savefig(joinpath(FIGDIR, "euc_column_spinup.png"), dpi = 150)

# ---- PGF decomposition figure ----
fig3, ax3 = subplots(1, 2, figsize = (9, 4.5), sharey = true)
ax3[1].plot(prof["ctrl"].pgf, zc, "k", label = "ctrl")
ax3[1].plot(prof["warm"].pgf, zc, "C2", label = "warm")
ax3[1].plot(prof["wind"].pgf, zc, "C1", label = "wind-only")
ax3[1].set_xlabel("PGF (m s⁻²)"); ax3[1].set_ylabel("z (m)"); ax3[1].set_ylim(-300, 0); ax3[1].legend(fontsize = 8)
ax3[2].plot(ΔPGF_warm, zc, "C2", label = "warm − ctrl")
ax3[2].plot(ΔPGF_cd,   zc, "C4", label = "stress-attributable (wind run × Δτ ratio)")
ax3[2].plot(ΔPGF_peng, zc, "C0", label = "pycnocline (residual)")
ax3[2].axvline(0, color = "0.6", lw = 0.6); ax3[2].set_xlabel("ΔPGF (m s⁻²)"); ax3[2].legend(fontsize = 8)
suptitle("Zonal PGF at 0°,140°W and its decomposition")
tight_layout(); savefig(joinpath(FIGDIR, "euc_column_pgf_split.png"), dpi = 150)

# ============================================================================ #
# 9. Shear layer above the jet: does the shear rise with the stratification?
#    Expectation: S² and N² both rise, Ri stays near critical, ν is maintained
#    (shifted up with the core) rather than switched off.
# ============================================================================ #
S2_A = Dict(); N2_A = Dict()
for r in ("ctrl", "warm")
    S2_A[r], N2_A[r] = shear_strat(prof[r].u, prof[r].T, prof[r].S)
end

println("\n=== shear layer above the jet (jet max → KPP h): layer means ===")
@printf("%-8s %14s %10s %10s %10s %10s\n", "run", "layer (m)", "S² ×1e-4", "N² ×1e-4", "Ri", "ν ×1e-3")
for k in ("ctrl", order...)
    sl = runs[k].sl
    @printf("%-8s %6.0f → %5.0f %10.2f %10.2f %10.2f %10.2f\n", k, sl.z_bot, sl.z_top,
            sl.S2 * 1e4, sl.N2 * 1e4, sl.Ri, sl.ν * 1e3)
end
c_sl = runs["ctrl"].sl
println("fractional change vs ctrl:")
for k in order
    sl = runs[k].sl
    @printf("%-8s ΔS² %+5.0f%%  ΔN² %+5.0f%%  ΔRi %+5.0f%%  Δν %+5.0f%%\n", k,
            100(sl.S2 / c_sl.S2 - 1), 100(sl.N2 / c_sl.N2 - 1), 100(sl.Ri / c_sl.Ri - 1), 100(sl.ν / c_sl.ν - 1))
end

fig4, ax4 = subplots(1, 3, figsize = (13, 5.5), sharey = true)
ax4[1].set_ylabel("z (m)"); ax4[1].set_ylim(-200, 0)
for k in ("ctrl", "strat", "both")
    lab = k == "ctrl" ? "ctrl" : route_name[k]
    ax4[1].plot(runs[k].U, zc, cols[k], label = lab)
    ax4[1].axhline(runs[k].h_kpp, color = cols[k], lw = 0.6, ls = "--")
    ax4[2].plot(runs[k].N2f .* 1e4, zf, cols[k], label = lab)
    ax4[2].plot(runs[k].S2f .* 1e4, zf, cols[k], ls = ":", lw = 1)
    ax4[3].semilogx(runs[k].νf, zf, cols[k], label = lab)
end
ax4[1].plot(prof["ctrl"].u, zc, "k--", lw = 1, label = "ACCESS ctrl")
ax4[1].plot(prof["warm"].u, zc, "C2--", lw = 1, label = "ACCESS warm")
ax4[2].plot(N2_A["ctrl"] .* 1e4, zf, "k--", lw = 1); ax4[2].plot(N2_A["warm"] .* 1e4, zf, "C2--", lw = 1)
ax4[2].plot(S2_A["ctrl"] .* 1e4, zf, "k:", lw = 1);  ax4[2].plot(S2_A["warm"] .* 1e4, zf, "C2:", lw = 1)
ax4[1].axvline(0, color = "0.6", lw = 0.6); ax4[1].set_xlabel("u (m/s)"); ax4[1].set_title("zonal velocity (dashed: KPP h)")
ax4[2].set_xlabel("×1e-4 s⁻²"); ax4[2].set_title("N² (solid) and S² (dotted); ACCESS dashed/dotted")
ax4[3].set_xlim(1e-5, 1e-1); ax4[3].set_xlabel("ν (m²/s)"); ax4[3].set_title("KPP + interior shear ν")
ax4[1].legend(fontsize = 7)
suptitle(@sprintf("Shear layer above the jet (jet max → KPP h) — strat: ΔN² %+.0f%%, ΔS² %+.0f%%, Ri %.2f → %.2f, Δν %+.0f%%;  both: ΔN² %+.0f%%, ΔS² %+.0f%%, Ri %.2f → %.2f, Δν %+.0f%%",
                  100(runs["strat"].sl.N2 / c_sl.N2 - 1), 100(runs["strat"].sl.S2 / c_sl.S2 - 1), c_sl.Ri, runs["strat"].sl.Ri, 100(runs["strat"].sl.ν / c_sl.ν - 1),
                  100(runs["both"].sl.N2 / c_sl.N2 - 1),  100(runs["both"].sl.S2 / c_sl.S2 - 1),  c_sl.Ri, runs["both"].sl.Ri,  100(runs["both"].sl.ν / c_sl.ν - 1)))
tight_layout(); savefig(joinpath(FIGDIR, "euc_column_shear_layer.png"), dpi = 150)
println("saved $(joinpath(FIGDIR, "euc_column_shear_layer.png"))")

# ============================================================================ #
# 10. Why does the surface current (not) weaken?  Two diagnostics.
#
# (a) Shear-layer ladder.  Between the mixed-layer base h and the jet maximum the
#     velocity climbs from u_h to u_core over a thickness L = h − z_core with mean
#     shear S̄ = (u_core − u_h)/L; inside the mixed layer it drops by D = u_h − u_surf:
#         u_surf = u_core − S̄·L − D
#     so   Δu_surf = Δu_core − L·ΔS̄ − S̄·ΔL − ΔS̄·ΔL − ΔD.
#     "EUC intrusion" is the −S̄·ΔL term (shoaling thins the layer, ΔL < 0 → eastward);
#     a sharper pycnocline raises S̄ (Ri held at threshold) → −L·ΔS̄ westward.
#     Done for the column runs and for the ACCESS profiles (h = density MLD).
#
# (b) Mixed-layer momentum budget per run (0 → h):  τ/(ρ₀h) + F_h/h + PGF + adv + Rayleigh,
#     with F_h = −ν ∂u/∂z at h (eastward when the EUC is below), and its change vs ctrl.
# ============================================================================ #
function ladder(U, T, S, h)
    k_core = argmax(U);  u_core = U[k_core];  z_core = zc[k_core]
    u_h    = interp1([h], zc, U)[1]
    L      = h - z_core
    S̄      = (u_core - u_h) / L
    N2     = shear_strat(U, T, S)[2]
    sel    = [i for i in 2:N if zf[i] > z_core && zf[i] < h]
    N̄2     = isempty(sel) ? NaN : sum(N2[sel]) / length(sel)
    return (; u_core, z_core, h, u_h, L, S̄, D = u_h - U[end], u_surf = U[end], Ri_eff = N̄2 / S̄^2)
end
function ladder_change(a, c)       # a = run, c = control
    (; du_core = a.u_core - c.u_core, dS = -c.L * (a.S̄ - c.S̄), dL = -c.S̄ * (a.L - c.L),
       cross = -(a.S̄ - c.S̄) * (a.L - c.L), dD = -(a.D - c.D), du_surf = a.u_surf - c.u_surf,
       dDrop = -((a.u_core - a.u_h) - (c.u_core - c.u_h)),      # −Δ(drop across shear layer) = −Δ(S̄L); robust to the S̄/L split
       du_h = a.u_h - c.u_h)
end

lad = Dict(k => ladder(runs[k].U, runs[k].T, runs[k].S, runs[k].h_kpp) for k in keys(runs))
lad_A = Dict(r => ladder(prof[r].u, prof[r].T, prof[r].S, mld_from_profile(prof[r].T, prof[r].S, zc)) for r in ("ctrl", "warm"))

println("\n=== shear-layer ladder: u_surf = u_core − S̄·L − D ===")
@printf("%-14s %7s %7s %7s %7s %6s %9s %7s %7s %6s\n", "run", "u_core", "z_core", "h", "u_h", "L", "S̄ (1/s)", "D", "u_surf", "Ri_eff")
for (name, l) in [("ACCESS ctrl", lad_A["ctrl"]), ("ACCESS warm", lad_A["warm"]),
                  [("column $k", lad[k]) for k in ("ctrl", "peng", "cd", "strat", "stratwc", "both")]...]
    @printf("%-14s %7.3f %7.1f %7.1f %7.3f %6.1f %9.2e %7.3f %7.3f %6.2f\n",
            name, l.u_core, l.z_core, l.h, l.u_h, l.L, l.S̄, l.D, l.u_surf, l.Ri_eff)
end
println("\nΔu_surf decomposition (cm/s):  Δu_core  −L·ΔS̄ (sharper shear)  −S̄·ΔL (thinner layer = intrusion)  cross  −ΔD (ML shear)  | Δu_surf")
chg = Dict("ACCESS" => ladder_change(lad_A["warm"], lad_A["ctrl"]))
for k in ("peng", "cd", "strat", "stratwc", "both"); chg[k] = ladder_change(lad[k], lad["ctrl"]); end
for name in ("ACCESS", "peng", "cd", "strat", "stratwc", "both")
    x = chg[name]
    @printf("%-8s %+7.2f %+7.2f %+7.2f %+7.2f %+7.2f  | %+7.2f   (sum %+7.2f)\n", name,
            100 * x.du_core, 100 * x.dS, 100 * x.dL, 100 * x.cross, 100 * x.dD, 100 * x.du_surf,
            100(x.du_core + x.dS + x.dL + x.cross + x.dD))
end

function ml_budget(r)
    k_h   = argmin(abs.(zf .- r.h_kpp))                     # face nearest the KPP base
    cells = k_h:N                                            # cells above that face
    h     = -zf[k_h]
    F_h   = -r.νf[k_h] * (r.U[k_h] - r.U[k_h-1]) / Δz        # upward flux of u through h
    nm(x) = (v = filter(isfinite, x[cells]); sum(v) / length(v))
    return (; h, stress = r.τx / (constants.ρ₀ * h), base = F_h / h, pgf = nm(r.pgf),
              adv = nm(r.adv), ray = nm(r.ray), u_ml = nm(r.U))
end
mlb = Dict(k => ml_budget(runs[k]) for k in keys(runs))
println("\n=== column mixed-layer momentum budget (×1e-8 m s⁻², 0 → KPP h) ===")
@printf("%-8s %6s %8s %8s %8s %8s %8s %8s %8s\n", "run", "h", "τ/ρh", "base F_h/h", "PGF", "adv", "Rayleigh", "sum", "u_ml")
for k in ("ctrl", "peng", "cd", "strat", "stratwc", "both")
    b = mlb[k]
    @printf("%-8s %6.1f %8.1f %10.1f %8.1f %8.1f %8.1f %8.1f %8.3f\n", k, b.h, 1e8 * b.stress, 1e8 * b.base, 1e8 * b.pgf,
            1e8 * b.adv, 1e8 * b.ray, 1e8 * (b.stress + b.base + b.pgf + b.adv + b.ray), b.u_ml)
end
println("changes vs ctrl (×1e-8 m s⁻²); stress term split into Δτ at ctrl h and Δh at ctrl τ:")
bc = mlb["ctrl"]
for k in ("peng", "cd", "strat", "stratwc", "both")
    b = mlb[k]
    dτ = (runs[k].τx - runs["ctrl"].τx) / (constants.ρ₀ * bc.h)
    dh = runs["ctrl"].τx / constants.ρ₀ * (1 / b.h - 1 / bc.h)
    @printf("%-8s Δτ-part %+6.1f  Δh-part %+6.1f  Δbase %+6.1f  ΔPGF %+6.1f  Δadv %+6.1f  ΔRay %+6.1f  | Δu_ml %+.2f cm/s  Δu_surf %+.2f cm/s\n",
            k, 1e8 * dτ, 1e8 * dh, 1e8 * (b.base - bc.base), 1e8 * (b.pgf - bc.pgf), 1e8 * (b.adv - bc.adv), 1e8 * (b.ray - bc.ray),
            100(b.u_ml - bc.u_ml), 100(runs[k].jm.u_surf - runs["ctrl"].jm.u_surf))
end

# figure: ladder, robust form.  Δu_surf = Δu_core − Δ(drop across shear layer, h → core) − Δ(drop across mixed layer)
# (the split of the shear-layer drop into "sharper" (ΔS̄) and "thinner" (ΔL) is argmax-sensitive on a flat-topped jet; table only)
fig5, ax5 = subplots(figsize = (11, 4.5))
names5 = ("ACCESS", "peng", "cd", "strat", "stratwc", "both")
xlab5  = ("ACCESS\nwarm − ctrl", "pycnocline PGF\n(ctrl τ, strat.)", "c_d stress\n(+ its PGF)", "stratification\n(warm T,S,w)",
          "stratification\n(ctrl w)", "all warm forcing\n(PGF + τ + strat.)")
terms5 = [("Δu_core (jet maximum)", :du_core, "C0"),
          ("−Δ(u_core − u_h): drop across shear layer, h → core", :dDrop, "C3"),
          ("−Δ(u_h − u_surf): drop across mixed layer", :dD, "C1")]
w5 = 0.25
for (j, (lab, key, col)) in enumerate(terms5)
    ax5.bar((1:length(names5)) .+ (j - 2) * w5, [100 * getfield(chg[n], key) for n in names5], w5, color = col, label = lab)
end
ax5.plot((1:length(names5)) .- 0.125, [100 * chg[n].du_h for n in names5], "k", ls = "none", marker = "o", mfc = "none", ms = 8, label = "Δu_h (mixed-layer base)")
ax5.plot((1:length(names5)) .+ 0.125, [100 * chg[n].du_surf for n in names5], "k_", ms = 18, mew = 2, label = "Δu_surf")
ax5.set_xticks(1:length(names5)); ax5.set_xticklabels(xlab5, fontsize = 8); ax5.axhline(0, color = "0.6", lw = 0.6)
ax5.set_ylabel("cm/s (+ = eastward)")
ax5.set_title("Δu_surf = Δu_core − Δ(drop h→core) − Δ(drop in mixed layer);  column runs vs ACCESS at 0°,140°W")
ax5.legend(fontsize = 7, loc = "lower left")
tight_layout(); savefig(joinpath(FIGDIR, "euc_column_ladder.png"), dpi = 150)
println("saved $(joinpath(FIGDIR, "euc_column_ladder.png"))")
