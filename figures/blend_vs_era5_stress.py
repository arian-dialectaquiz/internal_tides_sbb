"""
============================================================================
 HURRICANE CATARINA WIND FORCING: ERA5 vs HOLLAND-BLENDED
 Read DIRECTLY from the two forcing files, so this shows the REAL blend
 actually used to force ROMS, not a reconstruction.
----------------------------------------------------------------------------
 no_hc    : ERA5 only          (forcing_paper_2.nc)
 with_hc  : Holland-blended    (forcing_paper2_hc.nc)

 Both files are on the same regular lat/lon ERA5 grid, so no staggered-grid
 handling is needed, and Uwind/Vwind are stored directly, so no inverse bulk
 formula is needed either. The correction is simply (with_hc - no_hc).

 The only external input is the IBTrACS track, used to place the vortex
 centre for the radial profile and to validate the blended intensity against
 the reported vmax.
============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------------
NO_HC   = "/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/inputs/forcing_paper_2.nc"
WITH_HC = "/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/inputs/forcing_paper2_hc.nc"
TRACK_CSV = "/home/arian/dd_waves/pyroms_tools/scripts_xesmf/catarina_2004_full.csv"

T2_START, T2_END = "2004-03-24", "2004-03-30"

# blending radii, only for annotating panel (d); Table S3
RMAX_KM   = 30.0
RB_FACTOR = 3.5
WB_FACTOR = 2.5

# optional map crop to the SBB sector (set to None to keep the full domain)
LON_LIM = (-56, -36)
LAT_LIM = (-36, -20)

KT_TO_MS = 0.514444
FIGNAME  = "wind_blend_T2.png"


# ---------------------------------------------------------------------------
# 1. HELPERS
# ---------------------------------------------------------------------------
def haversine_km(lon1, lat1, lon2, lat2):
	R = 6371.0
	p1, p2 = np.radians(lat1), np.radians(lat2)
	a = (np.sin((p2 - p1) / 2) ** 2
		 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
	return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_forcing(path, t0, t1, lon_lim=None, lat_lim=None):
	"""Load one forcing file over the analysis window, optionally cropped."""
	ds = xr.open_dataset(path, chunks={"time": 48})
	ds = ds.sel(time=slice(t0, t1))
	if lon_lim is not None:
		ds = ds.sel(lon=slice(*lon_lim))
	if lat_lim is not None:
		# ERA5 latitude may be descending; slice accordingly
		lat = ds.lat.values
		ds = (ds.sel(lat=slice(*lat_lim)) if lat[0] < lat[-1]
			  else ds.sel(lat=slice(lat_lim[1], lat_lim[0])))
	return ds


def load_track(csv, t0, t1):
	"""IBTrACS / tropycal export. Accepts 'mslp' or 'pc' for central pressure."""
	df = pd.read_csv(csv)
	tcol = "time" if "time" in df.columns else "date"
	df[tcol] = pd.to_datetime(df[tcol])
	df = df.rename(columns={tcol: "time"}).sort_values("time")
	df = df[(df.time >= pd.Timestamp(t0)) & (df.time <= pd.Timestamp(t1))]
	df = df[np.isfinite(df.lat) & np.isfinite(df.lon)].copy()

	# central pressure: 'mslp' in IBTrACS exports, 'pc' if already renamed
	pcol = "mslp" if "mslp" in df.columns else ("pc" if "pc" in df.columns else None)
	if pcol is None:
		raise ValueError(f"no pressure column found; have {list(df.columns)}")
	pc = pd.to_numeric(df[pcol], errors="coerce")
	df["pc"] = pc.where((pc > 850) & (pc < 1030))      # reject sentinels

	df["vmax_ms"] = pd.to_numeric(df.get("vmax"), errors="coerce") * KT_TO_MS
	return df.reset_index(drop=True)
	
# ---------------------------------------------------------------------------
# 2. LOAD
# ---------------------------------------------------------------------------
print("loading forcing files ...")
ds_e = load_forcing(NO_HC,   T2_START, T2_END, LON_LIM, LAT_LIM)
ds_b = load_forcing(WITH_HC, T2_START, T2_END, LON_LIM, LAT_LIM)

assert ds_e.sizes["time"] == ds_b.sizes["time"], "time axes differ between files"
assert ds_e.sizes["lat"] == ds_b.sizes["lat"] and ds_e.sizes["lon"] == ds_b.sizes["lon"], \
	"spatial grids differ between files"

times = pd.to_datetime(ds_e.time.values)
lon1d = ds_e.lon.values.astype(float)
lat1d = ds_e.lat.values.astype(float)
lon2d, lat2d = np.meshgrid(lon1d, lat1d)
print(f"  {len(times)} steps, grid {len(lat1d)} x {len(lon1d)}, "
	  f"{times[0]} to {times[-1]}")

# stress magnitude and wind speed, both files
tau_e = np.hypot(ds_e.sustr.values, ds_e.svstr.values)
tau_b = np.hypot(ds_b.sustr.values, ds_b.svstr.values)
spd_e = np.hypot(ds_e.Uwind.values, ds_e.Vwind.values)
spd_b = np.hypot(ds_b.Uwind.values, ds_b.Vwind.values)
ub_x, ub_y = ds_b.Uwind.values, ds_b.Vwind.values

has_pair = ("Pair" in ds_e) and ("Pair" in ds_b)
if has_pair:
	pair_e = ds_e.Pair.values
	pair_b = ds_b.Pair.values

trk = load_track(TRACK_CSV, T2_START, T2_END)
print(f"  track: {len(trk)} points, "
	  f"min pc {np.nanmin(trk.pc):.0f} hPa, "
	  f"max vmax {np.nanmax(trk.vmax_ms):.1f} m/s")
# ---------------------------------------------------------------------------
# 3. WINDOW MEANS AND DIAGNOSTICS
# ---------------------------------------------------------------------------
mean_e = np.nanmean(tau_e, axis=0)
mean_b = np.nanmean(tau_b, axis=0)
diff   = mean_b - mean_e

# time of peak blended wind, and the track position then
i_peak = int(np.nanargmax(spd_b.max(axis=(1, 2))))
t_peak = times[i_peak]
lon_c = np.interp(t_peak.value, trk.time.astype("int64"), trk.lon)
lat_c = np.interp(t_peak.value, trk.time.astype("int64"), trk.lat)
r_peak = haversine_km(lon_c, lat_c, lon2d, lat2d)

print("\n--- what the blend adds ---")
print(f"peak blended wind    {np.nanmax(spd_b):.1f} m/s at {t_peak}")
print(f"peak ERA5 wind       {np.nanmax(spd_e):.1f} m/s")
print(f"observed vmax (max)  {trk.vmax_ms.max():.1f} m/s")
print(f"T2 mean |tau| max    ERA5 {np.nanmax(mean_e):.3f} Pa, "
	  f"blended {np.nanmax(mean_b):.3f} Pa "
	  f"(x{np.nanmax(mean_b)/np.nanmax(mean_e):.1f})")
print(f"largest correction   {np.nanmax(np.abs(diff)):.3f} Pa")
if has_pair:
	dp = (pair_b - pair_e)
	print(f"deepest Pair change  {np.nanmin(dp)/100:.1f} hPa")

# spatial confinement: is the correction zero outside R_b + W_b ?
far = r_peak > (RB_FACTOR + WB_FACTOR) * RMAX_KM
d_peak = np.abs(spd_b[i_peak] - spd_e[i_peak])
print(f"max |dU| beyond Rb+Wb at peak: {np.nanmax(d_peak[far]):.2f} m/s "
	  f"(inside: {np.nanmax(d_peak[~far]):.2f} m/s)")


# ---------------------------------------------------------------------------
# 4. FIGURE
# ---------------------------------------------------------------------------
print("\nplotting ...")
fig = plt.figure(figsize=(14, 8.0))
gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.4)

vmax = np.nanpercentile(mean_b, 99.5)
lev = np.linspace(0, vmax, 21)

def base_map(ax):
	ax.plot(trk.lon, trk.lat, "k-", lw=1.4, zorder=5)
	ax.plot(trk.lon, trk.lat, "k.", ms=3, zorder=5)
	ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

# (a) ERA5 only
ax = fig.add_subplot(gs[0, 0])
c = ax.contourf(lon2d, lat2d, mean_e, levels=lev, cmap=plt.cm.gist_ncar_r, extend="max")
plt.colorbar(c, ax=ax, label=r"$|\tau|$ [Pa]")
base_map(ax); ax.set_title("(a) ERA5 only, T2 mean", loc="left", fontweight="bold")

# (b) blended
ax = fig.add_subplot(gs[0, 1])
c = ax.contourf(lon2d, lat2d, mean_b, levels=lev, cmap=plt.cm.gist_ncar_r, extend="max")
plt.colorbar(c, ax=ax, label=r"$|\tau|$ [Pa]")
base_map(ax); ax.set_title("(b) Blended forcing, T2 mean", loc="left", fontweight="bold")

# (c) difference
ax = fig.add_subplot(gs[0, 2])
dmax = np.nanpercentile(np.abs(diff), 99.8)
c = ax.contourf(lon2d, lat2d, diff, levels=np.linspace(-dmax, dmax, 21),
				cmap="RdBu_r", extend="both")
plt.colorbar(c, ax=ax, label=r"$\Delta|\tau|$ [Pa]")
base_map(ax); ax.set_title("(c) Blended $-$ ERA5", loc="left", fontweight="bold")

# (d) radial profile at the time of peak intensity
ax = fig.add_subplot(gs[1, 0])
rbin = np.arange(0, 620, 20)
idx = np.digitize(r_peak.ravel(), rbin)
prof = lambda f: np.array([np.nanmean(f.ravel()[idx == k]) for k in range(1, len(rbin))])
rc = 0.5 * (rbin[:-1] + rbin[1:])
ax.plot(rc, prof(spd_e[i_peak]), "C0-", lw=2, label="ERA5")
ax.plot(rc, prof(spd_b[i_peak]), "C3-", lw=2, label="Blended")
for x in (RB_FACTOR * RMAX_KM, (RB_FACTOR + WB_FACTOR) * RMAX_KM):
	ax.axvline(x, color="grey", ls=":", lw=1)
ax.text(RB_FACTOR * RMAX_KM, ax.get_ylim()[1]*0.96, r" $R_b$", fontsize=8, va="top")
ax.text((RB_FACTOR+WB_FACTOR) * RMAX_KM, ax.get_ylim()[1]*0.96,
		r" $R_b+W_b$", fontsize=8, va="top")
ax.set_xlabel("Radius from centre [km]"); ax.set_ylabel(r"$|U_{10}|$ [m s$^{-1}$]")
ax.legend(fontsize=9)
ax.set_title(f"(d) Radial profile, {t_peak:%d %b %H}h", loc="left", fontweight="bold")

# (e) domain-maximum wind through T2, validated against IBTrACS
ax = fig.add_subplot(gs[1, 1])
ax.plot(times, spd_e.max(axis=(1, 2)), "C0-", lw=2, label="ERA5")
ax.plot(times, spd_b.max(axis=(1, 2)), "C3-", lw=2, label="Blended")
ax.plot(trk.time, trk.vmax_ms, "ks", ms=5, label="IBTrACS $V_{max}$")
ax.set_ylabel(r"max $|U_{10}|$ [m s$^{-1}$]")
ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=30)
ax.set_title("(e) Domain maximum wind", loc="left", fontweight="bold")

# (f) blended wind vectors at peak, showing the vortex structure
ax = fig.add_subplot(gs[1, 2])
sk = max(1, len(lon1d) // 25)
c = ax.contourf(lon2d, lat2d, spd_b[i_peak], levels=15, cmap="cubehelix_r")
plt.colorbar(c, ax=ax, label=r"$|U_{10}|$ [m s$^{-1}$]")
ax.quiver(lon2d[::sk, ::sk], lat2d[::sk, ::sk],
		  ub_x[i_peak][::sk, ::sk], ub_y[i_peak][::sk, ::sk],
		  scale=600, width=0.004, color="w")
ax.plot(lon_c, lat_c, "co", ms=7, mec="k", zorder=6)
base_map(ax)
ax.set_title(f"(f) Blended wind at peak", loc="left", fontweight="bold")

fig.savefig(FIGNAME, dpi=300, bbox_inches="tight")
print(f"saved {FIGNAME}")
