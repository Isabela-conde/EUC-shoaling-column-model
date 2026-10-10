"""
EUC inertial-scaling analysis (Pedlosky 1987; Saenko et al. 2011) for ACCESS-OM2 runs.

Step 1  (euc_timeseries)   per run: monthly U, z_core at LON_EVAL; tau and g' averaged
                            over the same equatorial band; saved to <OUTDIR>/<run>_euc.nc
Step 2  (analyse)           annual means, fractional trends vs scaling prediction,
                            exponent fit  log U = a + p log tau + q log g'
Step 3  (plot)              one figure: actual vs predicted fractional trends, per run

You supply DATASETS = {name: base_path} and load_vars(base) -> xarray.Dataset with
  u (time, st_ocean, yu_ocean, xu_ocean), pottemp / prac_salt (time, st_ocean, yt_ocean, xt_ocean),
  tau_x (time, yu_ocean, xu_ocean).   Longitudes in ACCESS-OM2 native (-280..80).
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- settings
OUTDIR   = "/g/data/e14/ic0706/energetics_winds_surf_warming/EUC"
CONTROL  = "era5"
LON_EVAL = 220.0                 # 140W, degrees east 0-360 (jet metrics U, z_core)
BAND_LON = (160.0, 250.0)        # 160E-110W, degrees east 0-360 (tau and g')
LAT_BAND = slice(-2, 2)
Z_DEEP   = 500.0                 # "deeper ocean" for g'
Z_MAX    = 400.0                 # search core above this depth
L, RHO0, G = 9.0e6, 1026.0, 9.81

nat = lambda x: x - 360 if x > 80 else x      # 0-360 -> ACCESS-OM2 native (-280..80)

# DATASETS = {"era5": ..., "tas_053": ..., "uas_015": ..., "tas_053_uas_015": ...}
# def load_vars(base): ...


# ----------------------------------------------------------------------------- step 1
def _band_mean_u(ds):
    """u averaged 2S-2N, restricted to the tau/g' longitude band and upper Z_MAX m."""
    return (ds["u"].sel(yu_ocean=LAT_BAND).mean("yu_ocean")
                   .sel(xu_ocean=slice(nat(BAND_LON[0]), nat(BAND_LON[1])))
                   .sel(st_ocean=slice(0, Z_MAX))
                   .transpose("time", "st_ocean", "xu_ocean"))


def _core_depth(u_prof_da, dz=2.0):
    """Core depth from a profile interpolated to dz m spacing (removes level-jumping).
    u_prof_da: DataArray with st_ocean as a dim (other dims kept)."""
    zf = np.arange(float(u_prof_da.st_ocean.min()), Z_MAX, dz)
    uf = u_prof_da.interp(st_ocean=zf)
    return uf.idxmax("st_ocean"), uf.max("st_ocean")


def _sigma0(ds, lon_deg_east):
    """sigma0 (time, st_ocean, xt_ocean) over 2S-2N band, from pottemp / prac_salt."""
    SP = (ds["prac_salt"].sel(yt_ocean=LAT_BAND).mean("yt_ocean")
                         .sel(xt_ocean=slice(nat(BAND_LON[0]), nat(BAND_LON[1])))
                         .sel(st_ocean=slice(0, Z_DEEP + 100))
                         .transpose("time", "st_ocean", "xt_ocean").load())
    pt = (ds["pottemp"].sel(yt_ocean=LAT_BAND).mean("yt_ocean")
                       .sel(xt_ocean=slice(nat(BAND_LON[0]), nat(BAND_LON[1])))
                       .sel(st_ocean=slice(0, Z_DEEP + 100))
                       .transpose("time", "st_ocean", "xt_ocean").load())
    if float(pt.max()) > 100:                       # Kelvin -> Celsius
        pt = pt - 273.15
    z   = SP["st_ocean"].values
    p   = gsw.p_from_z(-z, 0.0)[None, :, None]
    lon = (SP["xt_ocean"].values % 360)[None, None, :]
    SA  = gsw.SA_from_SP(SP.values, p, lon, 0.0)
    CT  = gsw.CT_from_pt(SA, pt.values)
    return xr.DataArray(gsw.sigma0(SA, CT), coords=SP.coords, dims=SP.dims)


def euc_timeseries(ds):
    # --- jet metrics at LON_EVAL ---------------------------------------------------
    u_band = _band_mean_u(ds).load()
    u_eval = u_band.sel(xu_ocean=nat(LON_EVAL), method="nearest")
    z_core_eval, U_eval = _core_depth(u_eval)

    # --- g' over the band: sigma0 at each longitude's own core depth vs Z_DEEP ------
    z_core_x, _ = _core_depth(u_band)                                  # (time, xu_ocean)
    sig = _sigma0(ds, LON_EVAL)
    sig = sig.reindex(xt_ocean=u_band["xu_ocean"].values, method="nearest") \
             .rename({"xt_ocean": "xu_ocean"})                          # t -> u columns
    rho_core = sig.interp(st_ocean=z_core_x)                            # vectorised
    rho_deep = sig.sel(st_ocean=Z_DEEP, method="nearest")
    g_prime  = (G * (rho_deep - rho_core) / RHO0).mean("xu_ocean")

    # --- tau over the band --------------------------------------------------------
    tau_s = (ds["tau_x"].sel(yu_ocean=LAT_BAND,
                             xu_ocean=slice(nat(BAND_LON[0]), nat(BAND_LON[1])))
                        .mean(["yu_ocean", "xu_ocean"]).load())
    tau = np.abs(tau_s)

    out = xr.Dataset(dict(U=U_eval, z_core=z_core_eval, g_prime=g_prime,
                          tau_x=tau_s, tau_mag=tau,
                          H_pred=np.sqrt(tau * L / (RHO0 * g_prime)),
                          U_pred=(tau * g_prime * L / RHO0) ** 0.25))
    out.attrs.update(lon_eval=LON_EVAL, band_lon=str(BAND_LON), lat_band="2S-2N",
                     z_deep=Z_DEEP, L=L, rho0=RHO0)
    return out


def build_all(DATASETS, load_vars):
    os.makedirs(OUTDIR, exist_ok=True)
    for name, base in DATASETS.items():
        print(f"=== {name} ===", flush=True)
        out = euc_timeseries(load_vars(base))
        fn = os.path.join(OUTDIR, f"{name}_euc.nc")
        out.to_netcdf(fn + ".tmp"); os.replace(fn + ".tmp", fn)
        print(f"  saved {fn}: mean U={float(out.U.mean()):.2f} m/s, "
              f"z_core={float(out.z_core.mean()):.0f} m, g'={float(out.g_prime.mean()):.4f}")


# ----------------------------------------------------------------------------- step 2
def _trend_per_dec(da):
    """Linear trend of an annual series, per decade."""
    t = (da["time"].dt.year + 0.5).values
    return np.polyfit(t, da.values, 1)[0] * 10.0


def analyse(runs):
    """runs: list of run names (CONTROL must be included)."""
    ann = {r: xr.open_dataset(f"{OUTDIR}/{r}_euc.nc").resample(time="YE").mean()
           for r in runs}
    ctrl = ann[CONTROL].mean("time")

    # ---- fractional trends (per decade), actual vs scaling prediction -------------
    rows = []
    for r, ds in ann.items():
        dtau = _trend_per_dec(ds.tau_mag) / float(ctrl.tau_mag)
        dg   = _trend_per_dec(ds.g_prime) / float(ctrl.g_prime)
        rows.append(dict(
            run=r,
            dU_actual   = _trend_per_dec(ds.U)      / float(ctrl.U),
            dU_pred     = 0.25 * dtau + 0.25 * dg,
            dH_actual   = _trend_per_dec(ds.z_core) / float(ctrl.z_core),
            dH_pred     = 0.5 * dtau - 0.5 * dg,
            dtau=dtau, dg=dg,
            z_core_trend_m = _trend_per_dec(ds.z_core),
        ))
    table = pd.DataFrame(rows).set_index("run")

    # ---- exponent fit: log U = a + p log tau + q log g' ---------------------------
    df = pd.concat([ds[["U", "z_core", "tau_mag", "g_prime"]].to_dataframe().assign(run=r)
                    for r, ds in ann.items()]).dropna()
    X = sm.add_constant(np.log(df[["tau_mag", "g_prime"]]))
    hac = dict(cov_type="HAC", cov_kwds={"maxlags": 2})
    fitU = sm.OLS(np.log(df["U"]), X).fit(**hac)
    fitH = sm.OLS(np.log(df["z_core"]), X).fit(**hac)

    def _fmt(fit, name, tau_th, g_th):
        ci = fit.conf_int()
        return (f"{name}: tau exponent = {fit.params['tau_mag']:.2f} "
                f"[{ci.loc['tau_mag',0]:.2f}, {ci.loc['tau_mag',1]:.2f}]  (inertial {tau_th}) ; "
                f"g' exponent = {fit.params['g_prime']:.2f} "
                f"[{ci.loc['g_prime',0]:.2f}, {ci.loc['g_prime',1]:.2f}]  (inertial {g_th})")

    print(table.round(4).to_string())
    print(_fmt(fitU, "U     ", 0.25, 0.25))
    print(_fmt(fitH, "z_core", 0.50, -0.50))
    return table, fitU, fitH


# ----------------------------------------------------------------------------- step 3
def plot(table, fn=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, lab in zip(axes, ["dU", "dH"],
                            ["core speed U", "core depth z_core"]):
        a, p = table[f"{key}_actual"] * 100, table[f"{key}_pred"] * 100
        lim = 1.1 * max(abs(a).max(), abs(p).max(), 1e-3)
        ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
        ax.axhline(0, color="0.7", lw=0.6); ax.axvline(0, color="0.7", lw=0.6)
        for r in table.index:
            ax.scatter(p[r], a[r], s=60, label=r)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("scaling prediction (% of control per decade)")
        ax.set_ylabel("actual (% of control per decade)")
        ax.set_title(lab)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    if fn:
        fig.savefig(fn, dpi=150)
    return fig


# ----------------------------------------------------------------------------- run
if __name__ == "__main__":
    # build_all(DATASETS, load_vars)          # step 1, once
    runs = ["era5", "tas_053", "uas_015", "tas_053_uas_015"]   # add 2-degree runs if saved
    table, fitU, fitH = analyse(runs)
    plot(table, fn=os.path.join(OUTDIR, "euc_scaling_trends.png"))
    plt.show()
