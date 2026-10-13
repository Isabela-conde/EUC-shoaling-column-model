"""
Mixing above the EUC core: fixed-depth vs core-relative view (ACCESS-OM2, control vs 4° warming).

Answers: does warming suppress the mixing above the core, or just move it up with the core?
  panel (a) κ(z) at fixed depth, ctrl vs warm           -> looks like suppression
  panel (b) κ, N², S² vs height above the core, ctrl vs warm -> displacement: all three rise
  panel (c) % change of the 20–60 m-above-core layer, fixed-depth vs core-relative

Core depth is the eastward-flow centroid on a 1 m interpolated profile (smooth in longitude;
the argmax of a flat-topped jet jumps between levels and produces a sawtooth).

Needs, per run, last-decade means over 2S–2N on st_ocean levels:
  u (yu, xu), diff_cbt_t (yt, xt), pot_rho_0 (yt, xt), mld (yt, xt)
Fill in load_run() with your own loader (same files as ml_momentum_budget.ipynb).
"""
import numpy as np, xarray as xr, matplotlib.pyplot as plt

RUNS   = {"ctrl": "era5", "warm": "tas_053"}
YEARS  = slice("2008", "2017")
LAT    = slice(-2, 2)
LON    = (180.0, 230.0)          # 0-360
ZMAX   = 300.0
LAYER  = (20.0, 60.0)            # metres above the core for the layer mean
RHO0, G = 1026.0, 9.81
nat = lambda x: x - 360 if x > 80 else x

# ---------------------------------------------------------------------------- loading
def load_run(base):
    """Return dict with band-mean, time-mean DataArrays: u, kappa, rho (st_ocean, lon), mld (lon).
    Replace the body with your loader; keep the names and dims."""
    raise NotImplementedError("fill load_run() with your mf()/DATASETS loader")
    # example skeleton:
    # u   = mf(base, 'output*/ocean/ocean-3d-u-1-monthly-mean-*.nc').u.sel(time=YEARS, yu_ocean=LAT,
    #           xu_ocean=slice(nat(LON[0]), nat(LON[1])), st_ocean=slice(0, ZMAX)).mean(["time", "yu_ocean"])
    # kap = ... diff_cbt_t   (yt_ocean, xt_ocean, st_ocean)
    # rho = ... pot_rho_0    (yt_ocean, xt_ocean, st_ocean); +1000 if sigma
    # mld = ... mld          (yt_ocean, xt_ocean)
    # return dict(u=u.rename({"xu_ocean": "lon"}), kappa=kap.rename({"xt_ocean": "lon"}),
    #             rho=rho.rename({"xt_ocean": "lon"}), mld=mld.rename({"xt_ocean": "lon"}))

# ---------------------------------------------------------------------------- core depth
def centroid_depth(u_prof, z):
    """Eastward-flow centroid (m, positive down) of one profile on a 1 m grid above ZMAX."""
    zf = np.arange(z.min(), ZMAX, 1.0)
    uf = np.interp(zf, z, u_prof)
    m = uf > 0
    return np.sum(zf[m] * uf[m]) / np.sum(uf[m]) if m.any() else np.nan

def core_depths(u):
    z = u["st_ocean"].values
    return xr.DataArray([centroid_depth(u.sel(lon=x).values, z) for x in u["lon"].values],
                        coords={"lon": u["lon"]}, dims="lon")

# ---------------------------------------------------------------------------- fields on st_ocean
def n2_from_rho(rho):
    """N² = -(g/ρ0) dρ/dz on st_ocean (centred differences; z positive down)."""
    z = rho["st_ocean"]
    return (G / RHO0) * rho.differentiate("st_ocean")          # dρ/dz with z downward → positive when stable

def s2_from_u(u):
    return u.differentiate("st_ocean") ** 2

def to_core_relative(field, zc, zeta):
    """Interpolate field(st_ocean, lon) onto height-above-core zeta (m, positive upward) per longitude."""
    out = []
    for x in field["lon"].values:
        prof = field.sel(lon=x)
        zq = zc.sel(lon=x).item() - zeta                         # depth = core depth − height above core
        out.append(np.interp(zq, prof["st_ocean"].values, prof.values, left=np.nan, right=np.nan))
    return xr.DataArray(np.array(out).T, coords={"zeta": zeta, "lon": field["lon"]}, dims=("zeta", "lon"))

# ---------------------------------------------------------------------------- main
def analyse(fields):
    zeta = np.arange(0.0, 120.0, 2.0)
    res = {}
    for k, F in fields.items():
        u, kap, rho = F["u"], F["kappa"], F["rho"]
        # put kappa/rho on the u longitudes (nearest)
        kap = kap.reindex(lon=u["lon"].values, method="nearest")
        rho = rho.reindex(lon=u["lon"].values, method="nearest")
        zc  = core_depths(u)
        n2  = n2_from_rho(rho)
        s2  = s2_from_u(u)
        res[k] = dict(zc=zc, mld=F["mld"].reindex(lon=u["lon"].values, method="nearest"),
                      kap_z=kap.mean("lon"), n2_z=n2.mean("lon"), s2_z=s2.mean("lon"),
                      kap_c=to_core_relative(kap, zc, zeta).mean("lon"),
                      n2_c=to_core_relative(n2, zc, zeta).mean("lon"),
                      s2_c=to_core_relative(s2, zc, zeta).mean("lon"))
    return res, zeta

def layer_change(res, zeta):
    """% change warm vs ctrl of the layer mean, fixed depth (same absolute depths as ctrl's layer) and core-relative."""
    c, w = res["ctrl"], res["warm"]
    zc_c = float(c["zc"].mean())
    out = {}
    for name in ("kap", "n2", "s2"):
        # core-relative: LAYER above each run's own core
        sel = (zeta >= LAYER[0]) & (zeta <= LAYER[1])
        cr = float(w[f"{name}_c"][sel].mean() / c[f"{name}_c"][sel].mean() - 1) * 100
        # fixed depth: the control layer's absolute depths, for both runs
        zsel = slice(zc_c - LAYER[1], zc_c - LAYER[0])
        fd = float(w[f"{name}_z"].sel(st_ocean=zsel).mean() / c[f"{name}_z"].sel(st_ocean=zsel).mean() - 1) * 100
        out[name] = (fd, cr)
    return out, zc_c

def plot(res, zeta, chg, zc_c, fn=None):
    c, w = res["ctrl"], res["warm"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    # (a) fixed depth
    ax[0].semilogx(c["kap_z"], -c["kap_z"]["st_ocean"], "k", label="ctrl")
    ax[0].semilogx(w["kap_z"], -w["kap_z"]["st_ocean"], "C3", label="warm")
    ax[0].axhline(-float(c["zc"].mean()), color="k", ls="--", lw=0.8, label="core (ctrl)")
    ax[0].axhline(-float(w["zc"].mean()), color="C3", ls="--", lw=0.8, label="core (warm)")
    ax[0].axhline(-float(c["mld"].mean()), color="0.5", ls=":", lw=0.8, label="MLD (ctrl)")
    ax[0].axhspan(-(zc_c - LAYER[0]), -(zc_c - LAYER[1]), color="0.9", zorder=0)
    ax[0].set_ylim(-250, 0); ax[0].set_xlabel("κ (m² s⁻¹)"); ax[0].set_ylabel("z (m)")
    ax[0].set_title("(a) fixed depth: κ falls above the old core"); ax[0].legend(fontsize=8)
    # (b) core-relative
    for name, col, lab in (("kap", "C0", "κ"), ("n2", "C2", "N²"), ("s2", "C1", "S²")):
        ax[1].plot(c[f"{name}_c"] / c[f"{name}_c"].max(), zeta, col, label=f"{lab} ctrl")
        ax[1].plot(w[f"{name}_c"] / c[f"{name}_c"].max(), zeta, col, ls="--", label=f"{lab} warm")
    ax[1].axhspan(LAYER[0], LAYER[1], color="0.9", zorder=0)
    ax[1].set_xlabel("normalised by control maximum"); ax[1].set_ylabel("height above core (m)")
    ax[1].set_title("(b) core-relative: the layer rides up intact"); ax[1].legend(fontsize=8, ncol=3)
    # (c) bars
    names = ["κ", "N²", "S²"]; fd = [chg[k][0] for k in ("kap", "n2", "s2")]; cr = [chg[k][1] for k in ("kap", "n2", "s2")]
    x = np.arange(3)
    ax[2].bar(x - 0.2, fd, 0.4, color="0.6", label=f"fixed depth ({zc_c-LAYER[1]:.0f}–{zc_c-LAYER[0]:.0f} m)")
    ax[2].bar(x + 0.2, cr, 0.4, color="C0", label=f"core-relative ({LAYER[0]:.0f}–{LAYER[1]:.0f} m above core)")
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xticks(x); ax[2].set_xticklabels(names)
    ax[2].set_ylabel("warm − ctrl (%)"); ax[2].set_title("(c) layer-mean change"); ax[2].legend(fontsize=8)
    fig.suptitle(f"Mixing above the EUC core, {LON[0]:.0f}–{LON[1]:.0f}°E, 2S–2N, {YEARS.start}–{YEARS.stop}: "
                 f"core shift {float(c['zc'].mean() - w['zc'].mean()):+.0f} m (up)")
    fig.tight_layout()
    if fn: fig.savefig(fn, dpi=150)
    return fig

if __name__ == "__main__":
    fields = {k: load_run(DATASETS[v]) for k, v in RUNS.items()}
    res, zeta = analyse(fields)
    chg, zc_c = layer_change(res, zeta)
    for k, (fd, cr) in chg.items():
        print(f"{k:4s}  fixed depth {fd:+6.0f}%   core-relative {cr:+6.0f}%")
    plot(res, zeta, chg, zc_c, fn="kappa_core_relative.png")
    plt.show()
