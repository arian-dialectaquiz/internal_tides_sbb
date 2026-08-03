"""
============================================================================
 hurricane_blend.py
----------------------------------------------------------------------------
 Blends a Holland parametric vortex into the ERA5 fields used to force ROMS,
 following the manuscript methods (Eqs. 7-8, Table S3).

 WHAT IS MODIFIED, and why
   u10, v10      the blend is DEFINED on the 10 m wind (Eq. 8), so this is
                 the primary correction
   metss, mntss  surface stress, recomputed from the blended wind so that the
                 stress and the wind in the forcing file remain consistent
   sp            mean sea level pressure, deepened by the Holland pressure
                 profile, keeping the pressure and wind fields dynamically
                 consistent (sp feeds air density and dQdSST downstream)

 WHERE TO CALL IT
   In make_forcing.py, immediately after `tref` is built and BEFORE the mask,
   heat flux, wspd, and dQdSST blocks, so every derived quantity is computed
   from the corrected wind.

 SOUTHERN HEMISPHERE
   Cyclonic rotation is CLOCKWISE. The tangential unit vector and the inflow
   rotation below encode that; do not reuse this on a NH storm unchanged.
============================================================================
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters (Table S3). Override via the `params` dict in blend_era5_vortex.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    rho_air             = 1.15,     # kg m-3
    pn_hpa              = 1012.0,   # ambient pressure
    bshape              = 1.45,     # Holland B, 1.2 - 1.7
    rmax_km             = 30.0,     # radius of maximum winds, 25 - 40 km
    rb_factor           = 3.5,      # R_b  ~ 3-4 Rmax
    wb_factor           = 2.5,      # W_b  ~ 2-3 Rmax
    gradient_to_surface = 0.80,     # gradient -> 10 m reduction
    inflow_angle_deg    = 22.0,     # inflow across isobars, toward centre
    alpha_trans         = 0.55,     # fraction of translation speed added
    cap_drag            = False,    # saturate Cd above u_sat
    u_sat               = 33.0,     # m s-1
    blend_pressure      = True,     # also deepen sp
    min_vmax_ms         = 0.0,      # skip times weaker than this
)


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def holland_gradient_wind(r_km, pc_hpa, lat_c, p):
    """Holland gradient wind, manuscript Eq. 7."""
    r = np.maximum(r_km, 1.0) * 1e3
    rmax = p["rmax_km"] * 1e3
    dp = max((p["pn_hpa"] - pc_hpa), 0.0) * 100.0            # Pa
    f = 2 * 7.2921e-5 * np.sin(np.radians(abs(lat_c)))       # |f| at centre
    x = (rmax / r) ** p["bshape"]
    term = (p["bshape"] / p["rho_air"]) * x * dp * np.exp(-x)
    return np.sqrt(term + (r * f / 2) ** 2) - r * f / 2


def holland_pressure(r_km, pc_hpa, p):
    """Holland surface pressure profile, hPa."""
    r = np.maximum(r_km, 1.0) * 1e3
    rmax = p["rmax_km"] * 1e3
    x = (rmax / r) ** p["bshape"]
    return pc_hpa + (p["pn_hpa"] - pc_hpa) * np.exp(-x)


def blend_weight(r_km, p):
    """w(r): 1 inside R_b, cosine taper to 0 at R_b + W_b (Eq. 8)."""
    rb = p["rb_factor"] * p["rmax_km"]
    wb = p["wb_factor"] * p["rmax_km"]
    x = np.clip((r_km - rb) / wb, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def cd_bulk(u10, p):
    """Large & Pond style drag coefficient, optionally saturated."""
    u = np.asarray(u10, dtype=float)
    ueff = np.minimum(u, p["u_sat"]) if p["cap_drag"] else u
    return np.where(ueff < 11.0, 1.2e-3, (0.49 + 0.065 * ueff) * 1e-3)


def vortex_wind_field(lon2d, lat2d, lon_c, lat_c, pc, u_tr, v_tr, p):
    """
    Southern-Hemisphere surface wind vectors of the parametric vortex.
    Returns (u, v, r_km).
    """
    r = haversine_km(lon_c, lat_c, lon2d, lat2d)

    # local east/north offsets in km, signed
    dx = haversine_km(lon_c, lat2d, lon2d, lat2d) * np.sign(lon2d - lon_c)
    dy = haversine_km(lon2d, lat_c, lon2d, lat2d) * np.sign(lat2d - lat_c)
    rr = np.maximum(np.hypot(dx, dy), 1e-6)

    # radial outward unit vector, and clockwise tangential (SH cyclonic)
    e_r = np.stack([dx / rr, dy / rr])
    e_t = np.stack([dy / rr, -dx / rr])

    vg = holland_gradient_wind(r, pc, lat_c, p) * p["gradient_to_surface"]

    # rotate tangential toward the centre by the inflow angle
    b = np.radians(p["inflow_angle_deg"])
    uu = vg * (np.cos(b) * e_t[0] - np.sin(b) * e_r[0])
    vv = vg * (np.cos(b) * e_t[1] - np.sin(b) * e_r[1])

    # translation asymmetry, tapered with the vortex strength
    vmax = np.nanmax(vg)
    taper = np.clip(vg / vmax, 0, 1) if vmax > 0 else np.zeros_like(vg)
    uu = uu + p["alpha_trans"] * u_tr * taper
    vv = vv + p["alpha_trans"] * v_tr * taper
    return uu, vv, r


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_track(csv_path):
    """
    Simple track CSV with columns: time, lon, lat, pc   (pc in hPa, lon deg E).
    """
    trk = pd.read_csv(csv_path, parse_dates=["time"])
    need = {"time", "lon", "lat", "pc"}
    missing = need - set(trk.columns)
    if missing:
        raise ValueError(f"track file missing columns: {missing}")
    return trk.sort_values("time").reset_index(drop=True)


KT_TO_MS = 0.514444


def load_track_ibtracs(csv_path, tmin=None, tmax=None, pn_hpa=None,
                       fill_mslp=True, drop_no_mslp=False, verbose=True):
    """
    Load an IBTrACS / tropycal export (as produced by
    `storm.to_dataframe().to_csv(...)`) and return the columns the blend needs.

    Expected columns: time, lat, lon, vmax (knots), mslp (hPa), type.

    Missing mslp is common at the start and end of a track. With fill_mslp the
    gaps are filled from vmax through the inverted Holland relation
        dp = rho_a * e * V_g^2 / B
    which keeps the pressure and the wind mutually consistent; otherwise those
    rows are interpolated, or dropped with drop_no_mslp.
    """
    p = dict(DEFAULTS)
    if pn_hpa is not None:
        p["pn_hpa"] = pn_hpa

    df = pd.read_csv(csv_path)
    tcol = "time" if "time" in df.columns else "date"
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.rename(columns={tcol: "time"}).sort_values("time")

    df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])].copy()
    df["vmax_ms"] = pd.to_numeric(df.get("vmax"), errors="coerce") * KT_TO_MS

    pc = pd.to_numeric(df.get("mslp"), errors="coerce")
    pc = pc.where((pc > 850) & (pc < 1030))               # reject sentinels
    n_missing = int(pc.isna().sum())

    if fill_mslp and n_missing:
        vg = df["vmax_ms"] / p["gradient_to_surface"]
        dp_hpa = (p["rho_air"] * np.e * vg ** 2 / p["bshape"]) / 100.0
        pc = pc.fillna(p["pn_hpa"] - dp_hpa)
    pc = pc.interpolate(limit_direction="both")
    df["pc"] = pc

    if drop_no_mslp:
        df = df[np.isfinite(df["pc"])]
    if tmin is not None:
        df = df[df["time"] >= pd.Timestamp(tmin)]
    if tmax is not None:
        df = df[df["time"] <= pd.Timestamp(tmax)]

    out = df[["time", "lon", "lat", "pc", "vmax_ms"]].reset_index(drop=True)
    if "type" in df.columns:
        out["type"] = df["type"].values

    if verbose:
        print("\n--- IBTrACS track loaded ---")
        print(f"  {len(out)} points, {out.time.min()} to {out.time.max()}")
        print(f"  min pc {out.pc.min():.1f} hPa, max vmax {out.vmax_ms.max():.1f} m/s")
        print(f"  mslp filled at {n_missing} points" if n_missing else "  mslp complete")
        # consistency of the Holland parameters with the observed intensity
        i = int(np.nanargmax(out["vmax_ms"].values))
        dp = max(p["pn_hpa"] - out["pc"].values[i], 1.0) * 100
        vg_model = np.sqrt(p["bshape"] * dp / (p["rho_air"] * np.e))
        v_surf = vg_model * p["gradient_to_surface"]
        print(f"  at peak: observed {out.vmax_ms.values[i]:.1f} m/s, "
              f"model surface {v_surf:.1f} m/s with B={p['bshape']}, "
              f"reduction={p['gradient_to_surface']}")
        if abs(v_surf - out.vmax_ms.values[i]) > 3.0:
            need = out.vmax_ms.values[i] / vg_model
            print(f"  MISMATCH: set gradient_to_surface={need:.2f}, "
                  f"or bshape='auto' to calibrate B at each time")
    return out


def bshape_from_vmax(vmax_ms, pc_hpa, p, b_min=1.0, b_max=2.5):
    """
    Holland B implied by the observed intensity,
        B = rho_a * e * V_g^2 / dp,   V_g = vmax_surface / reduction
    so that the vortex reproduces the reported maximum wind.
    """
    dp = max((p["pn_hpa"] - pc_hpa), 1.0) * 100.0
    vg = vmax_ms / p["gradient_to_surface"]
    return float(np.clip(p["rho_air"] * np.e * vg ** 2 / dp, b_min, b_max))


def blend_era5_vortex(nc, tref, track, params=None, verbose=True):
    """
    Blend the vortex into an ERA5 dataset in place.

    nc    : xarray.Dataset with u10, v10, metss, mntss, sp on (time, latitude, longitude)
    tref  : DatetimeIndex of the same length as nc.time
    track : DataFrame from load_track
    """
    p = dict(DEFAULTS)
    if params:
        p.update(params)

    lon1d = nc["longitude"].values.astype(float)
    lat1d = nc["latitude"].values.astype(float)
    lon1d = np.where(lon1d > 180, lon1d - 360, lon1d)          # to -180..180
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)

    # interpolate the track onto the forcing times, NaN outside its span
    t = pd.DatetimeIndex(tref)
    ti = track.set_index("time")
    ti = ti.select_dtypes(include=[np.number])       # drop 'type' and other strings
    idx = ti.index.union(t)
    trk = ti.reindex(idx).interpolate("index", limit_area="inside").reindex(t)
    active = trk["lon"].notna().values

    # translation velocity, m s-1, computed ONLY over the valid track span so
    # that the NaN boundary does not create a spurious jump in the gradient
    u_tr = np.zeros(len(t))
    v_tr = np.zeros(len(t))
    if active.any():
        k = np.where(active)[0]
        ts = t[k].astype("int64").values / 1e9
        lo, la = trk["lon"].values[k], trk["lat"].values[k]
        if len(k) > 1:
            dts = np.gradient(ts)
            u_tr[k] = np.gradient(lo) * 111e3 * np.cos(np.radians(la)) / dts
            v_tr[k] = np.gradient(la) * 111e3 / dts
    # guard against pathological values from a coarse or noisy track
    spd_tr = np.hypot(u_tr, v_tr)
    too_fast = spd_tr > 30.0
    if too_fast.any():
        u_tr[too_fast] = 0.0
        v_tr[too_fast] = 0.0

    # pull to memory once
    u10 = np.asarray(nc["u10"].values, dtype=float)
    v10 = np.asarray(nc["v10"].values, dtype=float)
    sp = np.asarray(nc["sp"].values, dtype=float)
    has_stress = ("metss" in nc) and ("mntss" in nc)
    if has_stress:
        metss = np.asarray(nc["metss"].values, dtype=float)
        mntss = np.asarray(nc["mntss"].values, dtype=float)

    sp_is_pa = np.nanmedian(sp) > 2000.0                        # Pa vs hPa

    # optional per-time Holland B calibrated to the reported intensity
    auto_b = str(p.get("bshape")).lower() == "auto"
    has_vmax = "vmax_ms" in trk.columns
    if auto_b and not has_vmax:
        raise ValueError("bshape='auto' needs a vmax_ms column in the track")
    b_used = []

    n_applied = 0
    peak_before, peak_after = 0.0, 0.0

    for i in range(len(t)):
        if not active[i]:
            continue

        pi = dict(p)
        if auto_b:
            pi["bshape"] = bshape_from_vmax(trk["vmax_ms"].values[i],
                                            trk["pc"].values[i], p)
            b_used.append(pi["bshape"])

        # skip times when the system is too weak to justify a vortex
        if has_vmax and np.isfinite(trk["vmax_ms"].values[i]) \
           and trk["vmax_ms"].values[i] < p.get("min_vmax_ms", 0.0):
            continue

        hu, hv, r = vortex_wind_field(lon2d, lat2d,
                                      trk["lon"].values[i], trk["lat"].values[i],
                                      trk["pc"].values[i], u_tr[i], v_tr[i], pi)
        w = blend_weight(r, pi)

        peak_before = max(peak_before, np.nanmax(np.hypot(u10[i], v10[i])))

        # ---- Eq. 8, the blend itself ----
        u10[i] = w * hu + (1 - w) * u10[i]
        v10[i] = w * hv + (1 - w) * v10[i]

        peak_after = max(peak_after, np.nanmax(np.hypot(u10[i], v10[i])))

        # ---- stress consistent with the blended wind ----
        if has_stress:
            spd = np.hypot(u10[i], v10[i])
            tau = p["rho_air"] * cd_bulk(spd, p) * spd
            metss[i] = tau * u10[i]
            mntss[i] = tau * v10[i]

        # ---- pressure deepened by the vortex ----
        if p["blend_pressure"]:
            ph = holland_pressure(r, trk["pc"].values[i], pi)
            ph = ph * 100.0 if sp_is_pa else ph
            sp[i] = w * ph + (1 - w) * sp[i]

        n_applied += 1

    nc["u10"].values = u10
    nc["v10"].values = v10
    nc["sp"].values = sp
    if has_stress:
        nc["metss"].values = metss
        nc["mntss"].values = mntss

    if verbose:
        print("\n--- hurricane blend ---")
        print(f"  vortex applied on {n_applied} of {len(t)} time steps")
        print(f"  track span         {track.time.min()} to {track.time.max()}")
        print(f"  peak |U10| before  {peak_before:5.1f} m/s")
        print(f"  peak |U10| after   {peak_after:5.1f} m/s")
        print(f"  R_b, R_b+W_b       {p['rb_factor']*p['rmax_km']:.0f}, "
              f"{(p['rb_factor']+p['wb_factor'])*p['rmax_km']:.0f} km")
        print(f"  stress rebuilt     {has_stress}, pressure blended {p['blend_pressure']}")
        if b_used:
            print(f"  Holland B (auto)   {min(b_used):.2f} to {max(b_used):.2f} "
                  f"(Table S3 range 1.2 to 1.7)")
        print(f"  drag cap           {p['cap_drag']}\n")

    return nc
