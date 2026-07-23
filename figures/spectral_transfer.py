"""
============================================================================
 DIRECTED BISPECTRAL ENERGY TRANSFER  (T_hat = Im{B}/norm = b * sin(beta))
----------------------------------------------------------------------------
 Extension of the existing bicoherence pipeline. 
	 bicoherence  b        = |B| / norm            (what you already plot)
	 directed transfer     = Im{B} / norm = b*sin(beta)   (NEW: signed)
	 biphase      beta      = arg(B)

 Sign convention (Kim & Powers 1979; Furuichi et al. 2005; Sun & Pinkel 2013):
	 T_hat > 0  -> energy flows INTO the sum frequency f1+f2   (forward)
	 T_hat < 0  -> the sum frequency DONATES to its daughters  (subharmonic/PSI)
	 T_hat ~ 0 with large b (beta ~ 0 or pi) -> a BOUND harmonic (no net flux)

 Two triads are evaluated:
	 FORWARD supertidal drain :  (M2, f)      -> sum = M2 + f   ; expect T_hat > 0
	 SUBHARMONIC (PSI) drain  :  (f,  M2 - f) -> sum = M2       ; expect T_hat < 0
 (self-interaction (M2,M2)->M4 is available with target_f1=target_f2=freq_m2)

 The absolute sign depends on the FFT convention (fixed once here); the ROBUST,
 convention-independent results are (i) the biphase clustering and (ii) the
 unforced -> wind-forced CONTRAST. NO dissipation is computed anywhere.

 This script reuses your data-loading pattern, `points` dict, station colours,
 per-point local f, and the fs=1 (wind) / fs=1/3 (no-wind) sampling you use.
============================================================================
"""

import re
import glob
import numpy as np
import xarray as xr
import xroms
import matplotlib.pyplot as plt
from scipy.signal import stft

# ---------------------------------------------------------------------------
# 0. Loading (same pattern as your bicoherence script)
# ---------------------------------------------------------------------------
def natural_keys(text):
	return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

# --- With winds
s_path = '/Users/piero/arian/data1/IT_outs/velocity/'
s_f = sorted(glob.glob(s_path + 'velocity_decomp_rho_*.nc'), key=natural_keys)
ss = xr.open_mfdataset(s_f, combine='nested', concat_dim='eta_rho',
					   parallel=True).isel(s_rho=slice(0, -1))

# --- No winds
s_path_no = '/Users/piero/arian/data1/NO_WINDS_IT/velocity/'
s_f_no = sorted(glob.glob(s_path_no + 'velocity_decomp_rho_*.nc'), key=natural_keys)
ss_no = xr.open_mfdataset(s_f_no, combine='nested', concat_dim='eta_rho',
						  parallel=True).isel(s_rho=slice(0, -1))

# Grid / depths (for z profiles and depth integration)
FILENAME = '/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc'
ds = xr.open_dataset(FILENAME, chunks={'ocean_time': 1})
ds1, xgrid = xroms.roms_dataset(ds)
ds1 = ds1.sel(xi_rho=slice(40, None))

points = {
	'A': {'xi_rho': 350, 'eta_rho': 1233, 'color': 'k',         'linestyle': ':'},
	'B': {'xi_rho': 236, 'eta_rho': 896,  'color': 'royalblue', 'linestyle': '--'},
	'C': {'xi_rho': 293, 'eta_rho': 557,  'color': 'orange',    'linestyle': '-.'},
	'D': {'xi_rho': 413, 'eta_rho': 323,  'color': 'red',       'linestyle': '-'},
}

z_means = {pt: ds1.z_rho.isel(ocean_time=150, xi_rho=c['xi_rho'],
							  eta_rho=c['eta_rho'], s_rho=slice(0, -1))
		   for pt, c in points.items()}

M2_T = 12.42          # hours
freq_m2 = 1.0 / M2_T  # cph

# Sampling frequency per run (matches your bicoherence script)
FS_WIND = 1.0        # hourly output -> 1 cph
FS_NOWIND = 1.0 / 3.0 # 3-hourly output -> 1/3 cph

# Footprint half-width (grid points) for the ~10 km ensemble to raise DOF.
# Set FOOTPRINT=0 to reproduce your single-point pipeline exactly.
FOOTPRINT = 3
NPERSEG = 256


# ---------------------------------------------------------------------------
# 1. Core: complex bispectrum -> bicoherence, directed transfer, biphase
# ---------------------------------------------------------------------------
def _stft_twoSided(sig, fs, nperseg, noverlap):
	"""Two-sided, fftshifted STFT of a complex signal. Returns f_axis, Z[f, seg]."""
	f, _, Z = stft(sig, fs=fs, window='hann', nperseg=nperseg,
				   noverlap=noverlap, return_onesided=False)
	f = np.fft.fftshift(f)
	Z = np.fft.fftshift(Z, axes=0)
	return f, Z


def _nearest(freq_axis, target):
	return int(np.abs(freq_axis - target).argmin())


def depth_bispectrum(u_block, v_block, fs, nperseg, target_f1, target_f2,
					 noverlap_frac=0.5, use_twosided=False):
	"""
	Depth profiles of the COMPLEX normalized bispectrum for triad
	(f1, f2, f1+f2).

	u_block, v_block : (time, depth) or (time, depth, npix). Extra pixels are a
					   spatial footprint whose triple products are pooled with
					   the STFT segments to increase the degrees of freedom.

	Returns
	-------
	bicoh   : (depth,)  |B|/norm            (== your existing bicoherence)
	transf  : (depth,)  Im{B}/norm = b*sin(beta)   (directed transfer, signed)
	biphase : (depth,)  arg(B)  [radians]
	dof     : int       degrees of freedom (2 * number of pooled realizations)
	"""
	if u_block.ndim == 2:
		u_block = u_block[:, :, None]
		v_block = v_block[:, :, None]
	ntime, ndepth, npix = u_block.shape
	noverlap = int(nperseg * noverlap_frac)

	# frequency axis (one representative signal)
	f_axis, _ = _stft_twoSided(u_block[:, 0, 0] + 1j * v_block[:, 0, 0],
							   fs, nperseg, noverlap)
	keep = np.ones_like(f_axis, dtype=bool) if use_twosided else (f_axis > 0)
	f_use = f_axis[keep]
	i1 = _nearest(f_use, target_f1)
	i2 = _nearest(f_use, target_f2)
	i3 = _nearest(f_use, target_f1 + target_f2)

	bicoh = np.zeros(ndepth)
	transf = np.zeros(ndepth)
	biphase = np.zeros(ndepth)
	dof = 0

	for d in range(ndepth):
		triple = []   # A1 A2 A3*
		p12 = []      # |A1 A2|^2
		p3 = []       # |A3|^2
		for p in range(npix):
			w = u_block[:, d, p] + 1j * v_block[:, d, p]
			_, Z = _stft_twoSided(w, fs, nperseg, noverlap)
			Z = Z[keep, :]
			A1, A2, A3 = Z[i1, :], Z[i2, :], Z[i3, :]
			triple.append(A1 * A2 * np.conj(A3))
			p12.append(np.abs(A1 * A2) ** 2)
			p3.append(np.abs(A3) ** 2)
		triple = np.concatenate(triple)
		p12 = np.concatenate(p12)
		p3 = np.concatenate(p3)

		B = np.mean(triple)                             # complex bispectrum
		norm = np.sqrt(np.mean(p12) * np.mean(p3))
		if norm > 0:
			beta = B / norm                             # normalized bispectrum
			bicoh[d] = np.abs(beta)                     # b
			transf[d] = np.imag(beta)                   # T_hat = b*sin(biphase)
			biphase[d] = np.angle(B)
		dof = 2 * len(triple)

	return bicoh, transf, biphase, dof


def _extract_block(dsrun, xi, eta, half):
	"""Return (time, depth, npix) arrays of u_bc, v_bc over a (2*half+1)^2 block."""
	if half <= 0:
		u = dsrun.u_bc.isel(xi_rho=xi, eta_rho=eta).values         # (time, depth)
		v = dsrun.v_bc.isel(xi_rho=xi, eta_rho=eta).values
		return u, v
	u = dsrun.u_bc.isel(xi_rho=slice(xi - half, xi + half + 1),
						eta_rho=slice(eta - half, eta + half + 1)).values
	v = dsrun.v_bc.isel(xi_rho=slice(xi - half, xi + half + 1),
						eta_rho=slice(eta - half, eta + half + 1)).values
	# dims come back as (time, depth, eta, xi) -> (time, depth, npix)
	u = u.reshape(u.shape[0], u.shape[1], -1)
	v = v.reshape(v.shape[0], v.shape[1], -1)
	return u, v


# ---------------------------------------------------------------------------
# 2. Compute the two triads at every station, both runs
# ---------------------------------------------------------------------------
# storage[run][triad][pt] = (bicoh, transf, biphase, dof)
storage = {'wind': {'fwd': {}, 'psi': {}},
		   'nowind': {'fwd': {}, 'psi': {}}}
dof_track = {'wind': 0, 'nowind': 0}

for pt, c in points.items():
	xi, eta = c['xi_rho'], c['eta_rho']

	# local inertial frequency at this station, in cph
	f_point = np.abs(ds1.f.isel(xi_rho=xi, eta_rho=eta).values)
	T_point = (2 * np.pi / f_point) / 3600.0
	freq_f = 1.0 / T_point
	print(f"Station {pt}: local f = {freq_f:.4f} cph  |  M2-f = {freq_m2-freq_f:.4f}"
		  f"  |  M2+f = {freq_m2+freq_f:.4f} cph")

	for run, dsrun, fs in (('wind', ss, FS_WIND), ('nowind', ss_no, FS_NOWIND)):
		u_blk, v_blk = _extract_block(dsrun, xi, eta, FOOTPRINT)

		# FORWARD triad: (M2, f) -> M2 + f      (T_hat > 0 = tide feeds supertidal)
		storage[run]['fwd'][pt] = depth_bispectrum(
			u_blk, v_blk, fs, NPERSEG, target_f1=freq_m2, target_f2=freq_f)

		# PSI triad: (f, M2 - f) -> M2          (T_hat < 0 = M2 donates to near-inertial)
		storage[run]['psi'][pt] = depth_bispectrum(
			u_blk, v_blk, fs, NPERSEG, target_f1=freq_f, target_f2=freq_m2 - freq_f)

		dof_track[run] = storage[run]['fwd'][pt][3]

# significance envelope for the imaginary part (|T_hat| <= b; use b floor)
sig95 = {run: np.sqrt(6.0 / dof_track[run]) for run in ('wind', 'nowind')}
print("95% bicoherence floor:", sig95)


# ---------------------------------------------------------------------------
# 3. Plot: 3 x 2  (rows: forward, PSI, depth-integrated bars; cols: no-wind|wind)
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(3, 2, figsize=(8.5, 11))

def _plot_transfer_row(ax_nw, ax_w, triad, title):
	for ax, run in ((ax_nw, 'nowind'), (ax_w, 'wind')):
		for pt, c in points.items():
			_, transf, _, _ = storage[run][triad][pt]
			ax.plot(transf, z_means[pt], color=c['color'],
					linestyle=c['linestyle'], linewidth=1.8, label=pt)
		ax.axvline(0.0, color='grey', lw=0.8)
		ax.axvline(+sig95[run], color='0.6', ls=':', lw=0.8)
		ax.axvline(-sig95[run], color='0.6', ls=':', lw=0.8)
		ax.axvspan(0, ax.get_xlim()[1], color='tab:red', alpha=0.04)
		ax.axvspan(ax.get_xlim()[0], 0, color='tab:blue', alpha=0.04)
		ax.set_ylim(-900, 0)
		ax.set_xlim(-1, 1)
		ax.grid(True, alpha=0.25)
	ax_w.axhspan(-850, -650, color='grey', alpha=0.15)  # bicoherence peak band
	ax_nw.set_ylabel('Depth [m]', fontweight='bold')

# Row 1: forward triad
_plot_transfer_row(axs[0, 0], axs[0, 1], 'fwd', r'$(M_2,f)\!\to\!M_2\!+\!f$')
axs[0, 0].text(0.03, 1.04, r'(a) $(M_2,f)\!\to\!M_2{+}f$  No Wind',
			   transform=axs[0, 0].transAxes, fontweight='bold',
			   bbox=dict(fc='white', alpha=0.8))
axs[0, 1].text(0.03, 1.04, r'(b) $(M_2,f)\!\to\!M_2{+}f$  With Wind',
			   transform=axs[0, 1].transAxes, fontweight='bold',
			   bbox=dict(fc='white', alpha=0.8))

# Row 2: PSI triad
_plot_transfer_row(axs[1, 0], axs[1, 1], 'psi', r'$(f,M_2\!-\!f)\!\to\!M_2$')
axs[1, 0].text(0.03, 1.04, r'(c) $(f,M_2{-}f)\!\to\!M_2$  No Wind',
			   transform=axs[1, 0].transAxes, fontweight='bold',
			   bbox=dict(fc='white', alpha=0.8))
axs[1, 1].text(0.03, 1.04, r'(d) $(f,M_2{-}f)\!\to\!M_2$  With Wind',
			   transform=axs[1, 1].transAxes, fontweight='bold',
			   bbox=dict(fc='white', alpha=0.8))
axs[1, 0].set_xlabel(r'Directed transfer  $\widehat{T}=b\sin\beta$')
axs[1, 1].set_xlabel(r'Directed transfer  $\widehat{T}=b\sin\beta$')

# Row 3: depth-integrated net transfer per station (bar summary)
def _depth_integrate(transf, z):
	z = np.asarray(z)
	order = np.argsort(z)
	# Replaced np.trapz with np.trapezoid
	return np.trapezoid(np.asarray(transf)[order], z[order])

stations = list(points.keys())
x = np.arange(len(stations))
w = 0.38
for ax, run, tag in ((axs[2, 0], 'nowind', '(e) Depth-integrated  No Wind'),
					 (axs[2, 1], 'wind',   '(f) Depth-integrated  With Wind')):
	fwd_int = [_depth_integrate(storage[run]['fwd'][pt][1], z_means[pt]) for pt in stations]
	psi_int = [_depth_integrate(storage[run]['psi'][pt][1], z_means[pt]) for pt in stations]
	ax.bar(x - w/2, fwd_int, w, color='tab:red',  label=r'$(M_2,f)\!\to\!M_2{+}f$')
	ax.bar(x + w/2, psi_int, w, color='tab:blue', label=r'$(f,M_2{-}f)\!\to\!M_2$')
	ax.axhline(0, color='grey', lw=0.8)
	ax.set_xticks(x); ax.set_xticklabels(stations)
	ax.set_xlabel('Station (N $\\to$ S)')
	ax.grid(True, axis='y', alpha=0.25)
	ax.text(0.03, 1.04, tag, transform=ax.transAxes, fontweight='bold',
			bbox=dict(fc='white', alpha=0.8))
axs[2, 0].set_ylabel(r'$\int \widehat{T}\,dz$', fontweight='bold')
axs[2, 1].legend(fontsize=8, loc='best')

# shared station legend (top)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
		   ncol=4, fontsize='small', framealpha=0.9)

for ax in (axs[0, 1], axs[1, 1]):
	ax.yaxis.tick_right()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('bispectral_transfer.png', dpi=300)
print("Saved bispectral_transfer.png")






# ---------------------------------------------------------------------------
# 4. OPTIONAL: reconcile the summed forward transfer with <Pi_tau>
# ---------------------------------------------------------------------------
# The frequency-domain forward flux is the sum of Im{B}/norm over all triads
# whose sum frequency exceeds the supertidal cutoff you use for Pi_tau. Filling
# the loop below over a grid of (f1, f2) with f1+f2 > f_cut, and comparing the
# depth-station-integrated result against <Pi_tau> from your coarse-graining,
# gives an independent, triad-resolved confirmation of the cascade (no
# dissipation involved). Left as a hook to keep this script focused.
#
# f_cut = 1.0 / 8.0     # e.g. between M2 and M4, as in your coarse-graining tau
# ... accumulate depth_bispectrum over the (f1,f2) grid with f1+f2 > f_cut ...