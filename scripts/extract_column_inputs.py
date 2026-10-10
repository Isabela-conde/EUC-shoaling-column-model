"""
Extract 1-D column-model inputs at 0°, 140°W from ACCESS-OM2 control and warming runs.

Writes euc_column_inputs_140W.nc with, for each run in RUNS:
  T(z), S(z), u(z), w(z)   : 2S-2N band means at 140W            [degC, psu, m/s, m/s]
  pgf(z)                   : -(1/rho0) dp/dx, from pot_rho_0 + sea_level, slope over PGF_LON [m/s2]
  tau_x, q_net, sst        : scalars at 140W band mean             [N/m2, W/m2, degC]
averaged over YEARS[run].

load_vars(base) must return a Dataset with: u, wt, pottemp (or pot_temp), prac_salt (or salt),
pot_rho_0, sea_level, tau_x, net_sfc_heating.   Longitudes in ACCESS native (-280..80).
"""
import numpy as np, xarray as xr

OUT      = "euc_column_inputs_140W.nc"
LON_EVAL = 220.0                      # 140W (0-360)
PGF_LON  = (210.0, 230.0)             # 150W-130W: fit dp/dx over this span
LAT      = slice(-2, 2)
ZMAX     = 500.0
RHO0, G  = 1026.0, 9.81
RUNS     = {"ctrl": "era5", "warm": "tas_053"}
YEARS    = {"ctrl": slice("2008", "2017"), "warm": slice("2008", "2017")}   # last decade of each

nat = lambda x: x - 360 if x > 80 else x

# DATASETS = {...}; def load_vars(base): ...

def _band(da, latname, lonname, lon=None, lonslice=None):
    da = da.sel({latname: LAT}).mean(latname)
    if lon is not None:
        da = da.sel({lonname: nat(lon)}, method="nearest")
    if lonslice is not None:
        da = da.sel({lonname: slice(nat(lonslice[0]), nat(lonslice[1]))})
    return da

def column_inputs(ds, years):
    ds = ds.sel(time=years)
    pt = ds["pottemp"] if "pottemp" in ds else ds["pot_temp"]
    sp = ds["prac_salt"] if "prac_salt" in ds else ds["salt"]
    if float(pt.isel(time=0).max()) > 100: pt = pt - 273.15

    zt = ds["st_ocean"].sel(st_ocean=slice(0, ZMAX))
    T  = _band(pt, "yt_ocean", "xt_ocean", lon=LON_EVAL).sel(st_ocean=slice(0, ZMAX)).mean("time")
    S  = _band(sp, "yt_ocean", "xt_ocean", lon=LON_EVAL).sel(st_ocean=slice(0, ZMAX)).mean("time")
    u  = _band(ds["u"], "yu_ocean", "xu_ocean", lon=LON_EVAL).sel(st_ocean=slice(0, ZMAX)).mean("time")
    w  = _band(ds["wt"], "yt_ocean", "xt_ocean", lon=LON_EVAL).mean("time")        # on sw_ocean
    w  = w.interp(sw_ocean=zt.values).rename({"sw_ocean": "st_ocean"}) if "sw_ocean" in w.dims else w.sel(st_ocean=slice(0, ZMAX))

    # ---- pressure gradient force: p/rho0 = g*eta + (g/rho0) * int_z^0 (rho - rho0) dz' --------
    rho = _band(ds["pot_rho_0"], "yt_ocean", "xt_ocean", lonslice=PGF_LON).sel(st_ocean=slice(0, ZMAX)).mean("time")
    eta = _band(ds["sea_level"], "yt_ocean", "xt_ocean", lonslice=PGF_LON).mean("time")
    if float(rho.max()) < 500: rho = rho + 1000.0                                  # sigma -> rho
    dz  = xr.ones_like(rho["st_ocean"]) * np.gradient(rho["st_ocean"].values)      # layer thickness (approx)
    phyd = (G / RHO0) * ((rho - RHO0) * dz).cumsum("st_ocean") - (G / RHO0) * 0.5 * (rho - RHO0) * dz
    p_over_rho = G * eta + phyd                                                    # (st_ocean, xt_ocean)
    xm = np.deg2rad(p_over_rho["xt_ocean"].values) * 6.371e6                        # metres along equator
    slope = np.polyfit(xm, p_over_rho.transpose("st_ocean", "xt_ocean").values.T, 1)[0]   # d(p/rho)/dx per level
    pgf = xr.DataArray(-slope, coords={"st_ocean": p_over_rho["st_ocean"]}, dims="st_ocean")

    tau  = float(_band(ds["tau_x"], "yu_ocean", "xu_ocean", lon=LON_EVAL).mean("time"))
    qnet = float(_band(ds["net_sfc_heating"], "yt_ocean", "xt_ocean", lon=LON_EVAL).mean("time"))
    sst  = float(T.isel(st_ocean=0))

    out = xr.Dataset(dict(T=T, S=S, u=u, w=w, pgf=pgf))
    out = out.assign(tau_x=tau, q_net=qnet, sst=sst)
    return out

if __name__ == "__main__":
    pieces = []
    for key, name in RUNS.items():
        ds = load_vars(DATASETS[name])
        pieces.append(column_inputs(ds, YEARS[key]).expand_dims(run=[key]))
        print(key, "done")
    xr.concat(pieces, "run").to_netcdf(OUT)
    print("wrote", OUT)
