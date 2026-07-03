"""
Cross-scale KE transfer  Pi_tau(x,t)  via temporal coarse-graining (Eq. 9).

	Pi_tau = -rho0 * ( tau : grad(u_tilde) )
		   = -rho0 * ( tau_xx*dudx + tau_yy*dvdy + tau_xy*(dudy + dvdx) )

with the subfilter (subgrid) stress
	tau_ij = lowpass(u_i u_j) - u_i_tilde * u_j_tilde
and u_tilde = temporal LOW-PASS of u with cutoff period between M4 and M2,
so u_tilde retains subtidal + M2 and drops M4 and higher harmonics.

Pi_tau > 0  -> forward cascade (resolved (sub)tidal -> supertidal).

Output: one small NetCDF per eta-strip holding the DEPTH-INTEGRATED transfer
averaged within each M2 tidal cycle:  <int Pi_tau dz>(ocean_time, eta_rho, xi_rho)
in W m-2, where each ocean_time entry is one 12.42-h cycle mean. Depth is summed
on the fly and time is block-averaged per cycle, so the full 4-D field is never
held in memory. Separate T1/T2/T3 later by selecting ocean_time ranges.
"""

import os
import gc
import numpy as np
import xarray as xr
from scipy.signal import butter, filtfilt
import xroms
# =============================================================================
# 0. CONFIG  (to run the no-wind case later: change U_VAR/V_VAR + OUT_TAG only)
# =============================================================================
FILENAME   = '/Users/piero/arian/data1/nc_outs/avg_internal_tides_paper.nc'  # grid ds (dx, dy, dz)
S_PATH     = '/Users/piero/arian/data1/NO_WINDS_IT/velocity/'                  # velocity files
OUT_DIR    = '/Users/piero/arian/data1/NO_WINDS_IT/CG/'                        # where to save Pi
OUT_TAG    = 'no_wind'          # tag in the output filename

# IMPORTANT: coarse-graining needs the FULL, UNFILTERED velocity (must still
# contain M4 and higher harmonics). Use the raw baroclinic velocity here, NOT
# the M2-band field -- the low-pass below does the (sub)tidal/supertidal split.
U_VAR      = 'u_bc'         # unfiltered baroclinic velocity
V_VAR      = 'v_bc'

# --- physics / filter -------------------------------------------------------
RHO0          = 1025.0                     # reference density [kg m-3]
# cutoff period placed between M4 (6.2103 h) and M2 (12.4206 h).
# geometric mean ~ 8.78 h keeps subtidal+M2, removes M4+. TUNE if your paper
# uses a different design.
CUTOFF_HOURS  = float(np.sqrt(12.4206 * 6.2103))   # ~= 8.78 h
FILTER_ORDER  = 4                          # Butterworth order (zero-phase filtfilt)

# --- domain chunking (matches your loop) ------------------------------------
ETA_STEP   = 40
XI_START   = 40            # your xis = slice(40, None); velocities are pre-cropped to match
ETA_HALO   = 3            # extra rows read on each side of a strip for the d/dy stencil
DEPTH_BLOCK = 1           # how many s_rho levels held in RAM at once. 1 = lowest memory.
						  #   raise (e.g. 4, 8) only if you have RAM to spare (faster I/O).

# Averaging window (optional pre-restriction). Default = whole record; you will
# separate T1/T2/T3 from the per-cycle ocean_time axis of the OUTPUT instead.
TIME_SEL = slice(None)

# Per-cycle block averaging: each output ocean_time entry = mean over this many
# complete M2 cycles (1 = one 12.42-h tidal cycle per entry).
M2_HOURS       = 12.4206
CYCLES_PER_BIN = 1

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1. HELPERS
# =============================================================================
SPATIAL_DIMS = ('s_rho', 's_w', 'eta_rho', 'xi_rho', 'eta_v', 'xi_u', 'eta_u', 'xi_v')

def find_time_dim(da):
	for d in da.dims:
		if d not in SPATIAL_DIMS:
			return d
	raise ValueError(f"Could not find a time dim in {da.dims}")

def infer_dt_hours(da, time_dim):
	dtv = np.diff(da[time_dim].values)
	if np.issubdtype(dtv.dtype, np.timedelta64):
		return float(np.median(dtv) / np.timedelta64(1, 'h'))
	# ROMS ocean_time is usually seconds since a reference date
	return float(np.median(dtv)) / 3600.0

def m2_cycle_samples(dt_hours, n_periods=1, m2_hours=12.4206):
	"""Number of time samples spanning n_periods complete M2 cycles (rounded)."""
	cyc = int(round(n_periods * m2_hours / dt_hours))
	if cyc < 2:
		raise ValueError(f"Only {cyc} sample(s) per {n_periods} M2 cycle(s): "
						 f"dt={dt_hours:.3f} h is too coarse to bin.")
	return cyc

def cycle_time_axis(time_vals, cyc, n_cyc):
	"""One representative (mean) timestamp per cycle bin, datetime64-safe."""
	tv = time_vals[:n_cyc * cyc]
	if np.issubdtype(tv.dtype, np.datetime64):
		tn = tv.astype('datetime64[s]').astype('int64').reshape(n_cyc, cyc).mean(axis=1)
		return np.round(tn).astype('int64').astype('datetime64[s]')
	return np.asarray(tv, dtype='float64').reshape(n_cyc, cyc).mean(axis=1)

def lowpass_time(da, b, a, time_dim):
	"""Zero-phase Butterworth low-pass along the time axis. NaN (land) safe."""
	axis = da.get_axis_num(time_dim)
	arr = np.asarray(da.values, dtype=float)
	mask = ~np.isfinite(arr)
	if mask.any():
		arr = np.where(mask, 0.0, arr)     # fill land so it doesn't spread NaN in time
	filt = filtfilt(b, a, arr, axis=axis)
	if mask.any():
		filt[mask] = np.nan                # restore land mask
	return da.copy(data=filt)

def ddx(f, dx):    # d/d(xi)  -> "x"  (central diff, curvilinear-safe using local dx)
	return (f.shift(xi_rho=-1) - f.shift(xi_rho=1)) / (2.0 * dx)

def ddy(f, dy):    # d/d(eta) -> "y"
	return (f.shift(eta_rho=-1) - f.shift(eta_rho=1)) / (2.0 * dy)

# =============================================================================
# 2. LOAD
# =============================================================================
ds1 = xr.open_dataset(FILENAME, chunks={'ocean_time': 1, 's_rho': -1, 'eta_rho': 'auto', 'xi_rho': 'auto'})
ds, xgrid = xroms.roms_dataset(ds1)

# Velocities opened LAZILY (dask-backed): this reads only metadata, ~no RAM.
# Data is pulled into memory one strip at a time inside the loop, then freed.
# Adjust to how your velocity files are stored:
ss = xr.open_mfdataset(os.path.join(S_PATH, '*.nc'), chunks={},
					   data_vars='minimal', coords='minimal', compat='override')  # lazy, lean

xis      = slice(XI_START, None)
NETA     = ds.sizes['eta_rho']
TIME_DIM = find_time_dim(ss[U_VAR])

dt_hours = infer_dt_hours(ss[U_VAR], TIME_DIM)
Wn       = (1.0 / CUTOFF_HOURS) / ((1.0 / dt_hours) / 2.0)   # normalized cutoff
assert 0 < Wn < 1, f"Bad Wn={Wn:.3f}: check dt ({dt_hours:.3f} h) vs cutoff ({CUTOFF_HOURS:.3f} h)"
b, a = butter(FILTER_ORDER, Wn, btype='low')

print(f"dt = {dt_hours:.3f} h | cutoff = {CUTOFF_HOURS:.3f} h | Wn = {Wn:.3f} | rows = {NETA}")

# build strip edges that cover the WHOLE domain (your arange missed the last rows)
eta_edges = list(range(0, NETA, ETA_STEP))
if eta_edges[-1] != NETA:
	eta_edges.append(NETA)

# =============================================================================
# 3. MAIN LOOP  (outer: eta strips for I/O + halo;  inner: depth blocks for RAM)
#    Per depth block: filter over the WHOLE window, form Pi*dz, then bin-average
#    in non-overlapping M2 cycles into a 3-D accumulator:
#        accum[cyc, eta, xi] += sum_over(depth, samples-in-cycle) of  Pi * dz
#    Final field = accum / CYC  ->  <int Pi dz> per tidal cycle, in W m-2.
# =============================================================================
NS = ss.sizes['s_rho']
NT = ss[U_VAR].isel({TIME_DIM: TIME_SEL}).sizes[TIME_DIM]   # steps in the (optional) window

def _sel_t(da):
	"""Apply the time-window selection only if the array carries the time dim."""
	return da.isel({TIME_DIM: TIME_SEL}) if TIME_DIM in da.dims else da

def _order(da):
	"""Put dims in (..., s_rho, eta_rho, xi_rho) order and return a float32 ndarray."""
	lead = [d for d in da.dims if d not in ('s_rho', 'eta_rho', 'xi_rho')]
	return da.transpose(*lead, 's_rho', 'eta_rho', 'xi_rho').values.astype('float32')

# --- M2-cycle binning of the time axis (shared by every strip) ---
CYC     = m2_cycle_samples(dt_hours, CYCLES_PER_BIN, M2_HOURS)   # samples per bin
N_CYC   = NT // CYC                                              # complete bins
NT_USE  = N_CYC * CYC                                            # samples actually used
if N_CYC < 1:
	raise ValueError(f"Window has {NT} steps but one bin needs {CYC}; not enough data.")
cycle_time = cycle_time_axis(_sel_t(ss[TIME_DIM]).values, CYC, N_CYC)
print(f"{CYC} samples / bin ({CYCLES_PER_BIN} M2 cycle) | {N_CYC} bins | "
	  f"dropping {NT - NT_USE} trailing samples")

for i in range(len(eta_edges) - 1):
	eta_start, eta_end = eta_edges[i], eta_edges[i + 1]
	print(f"--- Pi strip {i}: rows {eta_start}..{eta_end} ---")

	# padded read range so the d/dy stencil has neighbors at strip boundaries
	eta_lo = max(eta_start - ETA_HALO, 0)
	eta_hi = min(eta_end + ETA_HALO, NETA)
	eta_read = slice(eta_lo, eta_hi)
	i0 = eta_start - eta_lo                 # interior offset inside the padded strip
	i1 = i0 + (eta_end - eta_start)
	eta_int = slice(eta_start, eta_end)

	# 2-D metrics for this padded strip (small)
	dx2d = ds.dx.isel(eta_rho=eta_read, xi_rho=xis)
	dy2d = ds.dy.isel(eta_rho=eta_read, xi_rho=xis)
	eta_coord, xi_coord = dx2d.eta_rho, dx2d.xi_rho
	NETA_INT = eta_end - eta_start
	NXI      = dx2d.sizes['xi_rho']

	# running 3-D accumulator: sum over depth & within-cycle samples of (Pi*dz)
	accum = np.zeros((N_CYC, NETA_INT, NXI), dtype='float64')

	# ---- inner loop over depth blocks: only DEPTH_BLOCK levels in RAM at once ----
	for k0 in range(0, NS, DEPTH_BLOCK):
		ksl = slice(k0, min(k0 + DEPTH_BLOCK, NS))

		u = _sel_t(ss[U_VAR].isel(eta_rho=eta_read, s_rho=ksl)).load().astype('float32')
		v = _sel_t(ss[V_VAR].isel(eta_rho=eta_read, s_rho=ksl)).load().astype('float32')
		u = u.assign_coords({'eta_rho': eta_coord, 'xi_rho': xi_coord})
		v = v.assign_coords({'eta_rho': eta_coord, 'xi_rho': xi_coord})

		# filtered velocity (subtidal + M2). NOTE: filter over the FULL window,
		# bin into cycles only afterwards, so the low-pass has no per-bin edges.
		u_f = lowpass_time(u, b, a, TIME_DIM)
		v_f = lowpass_time(v, b, a, TIME_DIM)

		# subfilter stress: filter product inline, drop raw product immediately
		tau_xx = lowpass_time(u * u, b, a, TIME_DIM) - u_f * u_f
		tau_xy = lowpass_time(u * v, b, a, TIME_DIM) - u_f * v_f
		tau_yy = lowpass_time(v * v, b, a, TIME_DIM) - v_f * v_f
		del u, v

		# strain (gradients of filtered velocity)
		dudx = ddx(u_f, dx2d); dudy = ddy(u_f, dy2d)
		dvdx = ddx(v_f, dx2d); dvdy = ddy(v_f, dy2d)
		del u_f, v_f

		contraction = tau_xx * dudx + tau_yy * dvdy + tau_xy * (dudy + dvdx)
		del tau_xx, tau_xy, tau_yy, dudx, dudy, dvdx, dvdy

		# Pi for this block, trimmed to the interior eta rows -> (time, blk, eta_int, xi)
		pi = (-RHO0 * contraction).isel(eta_rho=slice(i0, i1))
		del contraction

		# dz for the same levels/rows (weight for the vertical integral)
		dz_blk = _sel_t(ds.dz.isel(eta_rho=eta_int, xi_rho=xis, s_rho=ksl))

		pi_np = _order(pi)                                  # (time, blk, eta, xi)
		dz_np = _order(dz_blk)                              # (time or 1, blk, eta, xi)
		prod  = (pi_np * dz_np)[:NT_USE]                    # trim to whole cycles
		col   = np.nansum(prod, axis=1)                     # sum over depth -> (NT_USE, eta, xi)
		# bin the time axis into cycles and sum within each cycle
		accum += col.reshape(N_CYC, CYC, NETA_INT, NXI).sum(axis=1)
		del pi, dz_blk, pi_np, dz_np, prod, col
		gc.collect()
		print(f"    levels {k0}:{ksl.stop}/{NS} integrated")

	# per-cycle mean of the depth integral -> (cycle, eta, xi)
	pi_bar = (accum / CYC).astype('float32')

	# re-apply a land mask (nansum turned land into 0), broadcast over the cycle axis
	if 'mask_rho' in ds:
		water = ds['mask_rho'].isel(eta_rho=eta_int, xi_rho=xis).values > 0.5
	elif 'h' in ds:
		water = np.isfinite(ds['h'].isel(eta_rho=eta_int, xi_rho=xis).values)
	else:
		water = np.ones((NETA_INT, NXI), dtype=bool)
	pi_bar = np.where(water[None, :, :], pi_bar, np.nan)

	# --- build a small (cycle, eta, xi) dataset and save ---
	coords = {TIME_DIM: cycle_time,
			  'eta_rho': ds.eta_rho.isel(eta_rho=eta_int) if 'eta_rho' in ds.coords else np.arange(eta_start, eta_end),
			  'xi_rho':  dx2d.xi_rho}
	out = xr.Dataset(
		{'pi_int': ((TIME_DIM, 'eta_rho', 'xi_rho'), pi_bar)},
		coords=coords,
	)
	out['pi_int'].attrs.update(
		units='W m-2',
		long_name='per-M2-cycle depth-integrated cross-scale KE transfer '
				  '<int Pi_tau dz> (forward cascade positive)',
		cutoff_period_hours=CUTOFF_HOURS, rho0=RHO0, forcing=OUT_TAG,
		m2_cycle_hours=M2_HOURS * CYCLES_PER_BIN, samples_per_cycle=CYC, n_cycles=N_CYC)
	out[TIME_DIM].attrs['long_name'] = 'center time of each M2-cycle average'
	#for extra in ('lon_rho', 'lat_rho', 'h'):
	#	if extra in ds:
	#		out[extra] = ds[extra].isel(eta_rho=eta_int, xi_rho=xis)

	fn = os.path.join(OUT_DIR, f'piint_{OUT_TAG}_eta{eta_start:04d}_{eta_end:04d}.nc')
	out.to_netcdf(fn, encoding={'pi_int': {'zlib': True, 'complevel': 4}})
	print(f"    saved {fn}")

	out.close()
	del out, dx2d, dy2d, accum, pi_bar
	gc.collect()

print("Done.")

# =============================================================================
# 4. (later) combine the strips into the full 2-D map for plotting (Fig. 6 style)
# =============================================================================
# pi_all = xr.open_mfdataset(os.path.join(OUT_DIR, f'piint_{OUT_TAG}_eta*.nc'),
#                            combine='by_coords')['pi_int']
# pi_all.plot(vmin=-X, vmax=X, cmap='RdBu_r')   # warm = forward, cool = inverse