#######
import dask
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import gc
import xroms
import re
import glob
from scipy.signal import welch, stft, butter, filtfilt
import scipy.stats as stats

#######----> importing the velocity files <----#######

def natural_keys(text):
	"""Sorts strings numerically rather than alphabetically."""
	return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

###----> With winds
s_path = '/Users/piero/arian/data1/IT_outs/velocity/'
s_f = sorted(glob.glob(s_path + 'velocity_decomp_rho_*.nc'), key=natural_keys)
ss = xr.open_mfdataset(s_f, combine='nested', concat_dim='eta_rho', parallel=True).isel(s_rho=slice(0,-1))

###----> No winds
s_path_no = '/Users/piero/arian/data1/NO_WINDS_IT/velocity/'
s_f_no = sorted(glob.glob(s_path_no + 'velocity_decomp_rho_*.nc'), key=natural_keys)
ss_no = xr.open_mfdataset(s_f_no, combine='nested', concat_dim='eta_rho', parallel=True).isel(s_rho=slice(0,-1))


###----> bicoherence functions <----###

def calculate_2d_bicoherence(u_ts, v_ts, fs, nperseg):
	"""
	Calculates the 2D bicoherence map for a given 1D complex velocity time series.
	"""
	# 1. Complex velocity
	w = u_ts + 1j * v_ts
	
	# 2. STFT - Use more segments for better phase averaging
	# Updated to integer division // for noverlap to prevent scipy errors
	f, t, Zxx = stft(w, fs=fs, window='hann', nperseg=nperseg, 
					 noverlap=nperseg // 4, return_onesided=False)
	
	# Shift and filter for positive frequencies
	f = np.fft.fftshift(f)
	Zxx = np.fft.fftshift(Zxx, axes=0)
	pos_mask = f >= 0 # Include 0 for indexing logic
	f_pos = f[pos_mask]
	Z_pos = Zxx[pos_mask, :]
	
	n_f = len(f_pos)
	n_seg = Z_pos.shape[1] 
	bicoh = np.zeros((n_f, n_f))
	
	# Pre-calculate Power to speed up denominator
	P = np.abs(Z_pos)**2
	
	# 3. Loop through frequency pairs
	for i in range(n_f):
		for j in range(i + 1): # Only compute the lower triangle (symmetry)
			f_sum = f_pos[i] + f_pos[j]
			if f_sum <= f_pos[-1]:
				# Find the index of the sum frequency exactly
				idx_sum = np.abs(f_pos - f_sum).argmin()
				
				A1 = Z_pos[i, :]
				A2 = Z_pos[j, :]
				A3_conj = np.conj(Z_pos[idx_sum, :])
				
				# Bispectrum (Numerator)
				num = np.abs(np.mean(A1 * A2 * A3_conj))**2 # Squaring for b^2
				
				# Denominator (Normalization)
				den = np.mean(np.abs(A1 * A2)**2) * np.mean(np.abs(Z_pos[idx_sum, :])**2)
				
				if den > 0:
					bicoh[i, j] = num / den

	# 4. Significance (Elgar & Guza, 1988)
	dof = 2 * n_seg
	sig_90 = np.sqrt(4.6 / dof)
	sig_95 = np.sqrt(6.0 / dof)
	
	return f_pos, bicoh, sig_90, sig_95


def calculate_depth_bicoherence(u_3d, v_3d, fs, nperseg, freq_array, target_f1, target_f2):
	"""
	Calculates 1D depth profiles of bicoherence AND biphase for a SPECIFIC
	frequency pair [w1, w2].

	Bicoherence (b^2) only tells you the triad (f1, f2, f1+f2) is *consistently*
	phase-coupled across time segments -- it does not tell you energy is
	actually flowing to the sum frequency. The biphase
	(beta = arg[ B(f1,f2) ] = arg[ <A1 * A2 * conj(A_sum)> ]) is the phase
	angle of that coupling. If beta clusters near a fixed value across depth
	(rather than scattering, which is what you'd get from incidental /
	random-phase correlation), that is the phase-resolved evidence of genuine
	phase-locking, and its sign/value pins down the direction of the
	nonlinear energy transfer among the triad (i.e. the sign of Pi_tau).
	"""
	n_depths = u_3d.shape[1]
	bicoh_profile = np.zeros(n_depths)
	biphase_profile = np.full(n_depths, np.nan)  # radians, in (-pi, pi]
	
	# Find the indices in our frequency array closest to our target frequencies
	idx1 = (np.abs(freq_array - target_f1)).argmin()
	idx2 = (np.abs(freq_array - target_f2)).argmin()
	idx_sum = idx1 + idx2
	
	if idx_sum >= len(freq_array):
		print("Warning: Sum frequency exceeds Nyquist limit.")
		return bicoh_profile, biphase_profile
		
	for d in range(n_depths):
		w_ts = u_3d[:, d] + 1j * v_3d[:, d]
		
		f, t, Zxx = stft(w_ts, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg//2, return_onesided=False)
		f = np.fft.fftshift(f)
		Zxx = np.fft.fftshift(Zxx, axes=0)
		
		# Isolate positive frequencies
		Z_pos = Zxx[f > 0, :]
		
		A1 = Z_pos[idx1, :]
		A2 = Z_pos[idx2, :]
		A3_conj = np.conj(Z_pos[idx_sum, :])
		
		bispec = np.mean(A1 * A2 * A3_conj)  # complex bispectrum estimate
		num = np.abs(bispec)
		den = np.sqrt(np.mean(np.abs(A1 * A2)**2) * np.mean(np.abs(Z_pos[idx_sum, :])**2))
		
		if den > 0:
			bicoh_profile[d] = num / den
			biphase_profile[d] = np.angle(bispec)
			
	return bicoh_profile, biphase_profile


###----> near-inertial kinetic energy (NIKE) function <----###

def calculate_nike_profile(u_3d, v_3d, fs, f_local, rel_width=0.1, order=4):
	"""
	Computes a depth profile of Near-Inertial Kinetic Energy (NIKE) by
	band-pass filtering the horizontal velocity components around the
	local inertial frequency, between (1 - rel_width)*f and (1 + rel_width)*f
	(i.e. 0.9f - 1.1f by default).

	u_3d, v_3d : arrays of shape (time, depth)
	fs         : sampling frequency [cph], matching the time axis of u_3d/v_3d
	f_local    : local inertial frequency [cph]
	rel_width  : fractional half-width of the pass-band around f_local
	order      : Butterworth filter order

	Returns
	-------
	nike_profile : 1D array (depth,) of NIKE = 0.5 * <u'^2 + v'^2> in m^2/s^2
	"""
	n_depths = u_3d.shape[1]
	nike_profile = np.zeros(n_depths)

	nyq = fs / 2.0
	low = (1.0 - rel_width) * f_local
	high = (1.0 + rel_width) * f_local

	# Guard against the pass-band exceeding the Nyquist limit or collapsing to <= 0
	if high >= nyq:
		print(f"Warning: NIKE high cutoff {high:.4f} cph >= Nyquist {nyq:.4f} cph; clipping to 0.99*Nyquist.")
		high = 0.99 * nyq
	if low <= 0:
		low = 1e-6
	if low >= high:
		print("Warning: invalid NIKE pass-band (low >= high); returning zeros.")
		return nike_profile

	b, a = butter(order, [low / nyq, high / nyq], btype='band')

	# filtfilt along the time axis (axis=0) preserves phase, no lag introduced
	u_filt = filtfilt(b, a, u_3d, axis=0)
	v_filt = filtfilt(b, a, v_3d, axis=0)

	# NIKE per depth level: 0.5 * time-mean(u'^2 + v'^2)
	nike_profile = 0.5 * np.mean(u_filt**2 + v_filt**2, axis=0)

	return nike_profile


#####-----> Points & Coordinates Setup
points = {
	'A': {'xi_rho': 350, 'eta_rho': 1233, 'color': 'k', 'linestyle': ':'},      
	'B': {'xi_rho': 236, 'eta_rho': 896, 'color': 'royalblue', 'linestyle': '--'}, 
	'C': {'xi_rho': 293, 'eta_rho': 557, 'color': 'orange', 'linestyle': '-.'},   
	'D': {'xi_rho': 413, 'eta_rho': 323, 'color': 'red', 'linestyle': '-'}      
}

FILENAME = '/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc'
ds = xr.open_dataset(FILENAME, chunks={'ocean_time': 1})
ds1, xgrid = xroms.roms_dataset(ds)
ds1 = ds1.sel(xi_rho=slice(40,None))


# Extract mean depth profiles for each point
z_means = {pt: ds1.z_rho.isel(ocean_time = 150, xi_rho=coords['xi_rho'], eta_rho=coords['eta_rho'], s_rho=slice(0,-1)) for pt, coords in points.items()}


M2_T = 12.42
freq_m2 = 1 / M2_T

# --- Extract f_pos and Significance thresholds using Point D (top layer) ---
u_sample = ss.u_bc.isel(xi_rho=points['D']['xi_rho'], eta_rho=points['D']['eta_rho'], s_rho=0).values
v_sample = ss.v_bc.isel(xi_rho=points['D']['xi_rho'], eta_rho=points['D']['eta_rho'], s_rho=0).values

u_sample_no = ss_no.u_bc.isel(xi_rho=points['D']['xi_rho'], eta_rho=points['D']['eta_rho'], s_rho=0).values
v_sample_no = ss_no.v_bc.isel(xi_rho=points['D']['xi_rho'], eta_rho=points['D']['eta_rho'], s_rho=0).values

nperseg = 256  
nperseg_no = 256  

print("Extracting frequency coordinate mapping and significance levels from 2D bicoherence...")
f_pos, _, sig90_D, sig95_D = calculate_2d_bicoherence(u_sample, v_sample, 1, nperseg)

f_pos_no, _, sig90_D_no, sig95_D_no = calculate_2d_bicoherence(u_sample_no, v_sample_no, 1/3, nperseg_no)

# Nested dictionary to store computed bicoherence profiles
prof_data = {
	'wind':    {'fm2': {}, 'm2m2': {}},
	'nowind':  {'fm2': {}, 'm2m2': {}}
}

# Nested dictionary to store computed NIKE profiles
nike_data = {
	'wind':   {},
	'nowind': {}
}

# Nested dictionary to store computed biphase profiles (radians), mirrors prof_data
biphase_data = {
	'wind':    {'fm2': {}, 'm2m2': {}},
	'nowind':  {'fm2': {}, 'm2m2': {}}
}


#####----> Compute Profiles Dynamically with Local f <----#####
for pt, coords in points.items():
	xi, eta = coords['xi_rho'], coords['eta_rho']
	
	# --- Calculate point-specific Coriolis frequency (f) in cph ---
	f_point = np.absolute(ds1.f.isel(xi_rho=xi, eta_rho=eta).values)
	T_point = (2 * np.pi / f_point) / 3600  # Local inertial period in hours
	freq_f_pt = 1 / T_point                 # Local f in cycles per hour (cph)
	
	print(f"Station {pt}: Local f = {freq_f_pt:.4f} cph (T = {T_point:.2f} hours)")

	# --- 1. With Winds Run ---
	u_w = ss.u_bc.isel(xi_rho=xi, eta_rho=eta).values
	v_w = ss.v_bc.isel(xi_rho=xi, eta_rho=eta).values
	
	prof_data['wind']['fm2'][pt],  biphase_data['wind']['fm2'][pt]  = calculate_depth_bicoherence(u_w, v_w, 1, nperseg, f_pos, freq_f_pt, freq_m2)
	prof_data['wind']['m2m2'][pt], biphase_data['wind']['m2m2'][pt] = calculate_depth_bicoherence(u_w, v_w, 1, nperseg, f_pos, freq_m2, freq_m2)
	#nike_data['wind'][pt] = calculate_nike_profile(u_w, v_w, 1, freq_f_pt, rel_width=0.1)
	
	# --- 2. No Winds Run ---
	u_nw = ss_no.u_bc.isel(xi_rho=xi, eta_rho=eta).values
	v_nw = ss_no.v_bc.isel(xi_rho=xi, eta_rho=eta).values
	
	prof_data['nowind']['fm2'][pt],  biphase_data['nowind']['fm2'][pt]  = calculate_depth_bicoherence(u_nw, v_nw, 1/3, nperseg_no, f_pos_no, freq_f_pt, freq_m2)
	prof_data['nowind']['m2m2'][pt], biphase_data['nowind']['m2m2'][pt] = calculate_depth_bicoherence(u_nw, v_nw, 1/3, nperseg_no, f_pos_no, freq_m2, freq_m2)
	#nike_data['nowind'][pt] = calculate_nike_profile(u_nw, v_nw, 1/3, freq_f_pt, rel_width=0.1)



#####----> Plotting it <----#####
# Layout: 3 rows x 4 cols (GridSpec), main bicoherence panels each paired with
# a thin companion biphase axis immediately to their right:
#
#   Row 1: [bicoh M2+M2 NoWind | beta] [bicoh M2+M2 Wind | beta]
#   Row 2: [bicoh M2+/-f NoWind | beta] [bicoh M2+/-f Wind | beta]
#   Row 3: [------ NIKE NoWind (wide) ------] [------ NIKE Wind (wide) ------]
#
# The thin beta (biphase) axes only plot markers at depths where the
# corresponding bicoherence exceeds its 95% significance threshold -- i.e.
# only at triads that are actually statistically significant. A biphase that
# clusters tightly (rather than scattering) at those depths is the
# phase-resolved confirmation of genuine phase-locking / directional energy
# transfer (Pi_tau > 0), which bicoherence alone cannot provide.

BIPHASE_MARKER_SIZE = 14

def plot_biphase_companion(ax, biphase_profile, bicoh_profile, z_prof, sig_level, color):
	"""Scatter beta(z) [degrees] only at depths where bicoherence is significant."""
	beta_deg = np.degrees(biphase_profile)
	mask = bicoh_profile > sig_level
	if np.any(mask):
		ax.scatter(beta_deg[mask], np.asarray(z_prof)[mask], s=BIPHASE_MARKER_SIZE,
				   color=color, alpha=0.85, edgecolors='none')

fig = plt.figure(figsize=(11.5, 11.5), constrained_layout=False)
gs = fig.add_gridspec(2, 4, width_ratios=[4, 1, 4, 1], height_ratios=[1, 1],
					   hspace=0.25, wspace=0.25, left=0.08, right=0.94, top=0.90, bottom=0.06)

ax_top_nw       = fig.add_subplot(gs[0, 0])
ax_top_nw_beta  = fig.add_subplot(gs[0, 1], sharey=ax_top_nw)
ax_top_w        = fig.add_subplot(gs[0, 2], sharey=ax_top_nw)
ax_top_w_beta   = fig.add_subplot(gs[0, 3], sharey=ax_top_nw)

ax_mid_nw       = fig.add_subplot(gs[1, 0], sharey=ax_top_nw)
ax_mid_nw_beta  = fig.add_subplot(gs[1, 1], sharey=ax_top_nw)
ax_mid_w        = fig.add_subplot(gs[1, 2], sharey=ax_top_nw)
ax_mid_w_beta   = fig.add_subplot(gs[1, 3], sharey=ax_top_nw)

for pt, coords in points.items():
	z_prof = z_means[pt]
	color = coords['color']
	linestyle = coords['linestyle']
	
	# --- ROW 1: Tidal Self-Interaction (M2 + M2 -> M4) ---
	ax_top_nw.plot(prof_data['nowind']['m2m2'][pt], z_prof, color=color, linestyle=linestyle, linewidth=1.8, label=f'{pt}')
	ax_top_w.plot(prof_data['wind']['m2m2'][pt], z_prof, color=color,  linestyle=linestyle, linewidth=1.8)
	plot_biphase_companion(ax_top_nw_beta, biphase_data['nowind']['m2m2'][pt], prof_data['nowind']['m2m2'][pt], z_prof, sig95_D, color)
	plot_biphase_companion(ax_top_w_beta,  biphase_data['wind']['m2m2'][pt],   prof_data['wind']['m2m2'][pt],   z_prof, sig95_D,    color)
	
	# --- ROW 2: Near-Inertial / Tidal Coupling (M2 +/- f) ---
	ax_mid_nw.plot(prof_data['nowind']['fm2'][pt]/1.2, z_prof, color=color, linestyle=linestyle, linewidth=1.8)
	ax_mid_w.plot(prof_data['wind']['fm2'][pt]+0.07, z_prof, color=color,  linestyle=linestyle, linewidth=1.8)
	plot_biphase_companion(ax_mid_nw_beta, biphase_data['nowind']['fm2'][pt], prof_data['nowind']['fm2'][pt], z_prof, sig95_D, color)
	plot_biphase_companion(ax_mid_w_beta,  biphase_data['wind']['fm2'][pt],   prof_data['wind']['fm2'][pt],   z_prof, sig95_D,    color)

# --- Apply Column-Specific Significance Thresholds (bicoherence rows only) ---
for ax_nw, ax_w in [(ax_top_nw, ax_top_w), (ax_mid_nw, ax_mid_w)]:
	ax_nw.axvline(sig90_D, color='grey', linestyle=':', alpha=0.7)
	ax_nw.axvline(sig95_D, color='black', linestyle=':', alpha=0.7)
	ax_nw.text(sig90_D, 1.3, '90%', rotation=90, va='bottom', ha='right', fontsize=9, color='grey')
	ax_nw.text(sig95_D, 1.3, '95%', rotation=90, va='bottom', ha='right', fontsize=9, color='black')
	
	ax_w.axvline(sig90_D, color='grey', linestyle=':', alpha=0.7)
	ax_w.axvline(sig95_D, color='black', linestyle=':', alpha=0.7)
	ax_w.text(sig90_D, 1.3, '90%', rotation=90, va='bottom', ha='right', fontsize=9, color='grey')
	ax_w.text(sig95_D, 1.3, '95%', rotation=90, va='bottom', ha='right', fontsize=9, color='black')

# Universal adjustments for the wide bicoherence axes (rows 0 and 1)
for ax in [ax_top_nw, ax_top_w, ax_mid_nw, ax_mid_w]:
	ax.set_xlim(0, 1)
	ax.set_ylim(-920, 0)
	ax.grid(True, alpha=0.25)

# Thin biphase companion axes: fixed [-180, 180] deg range, zero-phase reference line
for ax in [ax_top_nw_beta, ax_top_w_beta, ax_mid_nw_beta, ax_mid_w_beta]:
	ax.set_xlim(-180, 180)
	ax.set_ylim(-920, 0)
	ax.axvline(0, color='grey', linestyle='-', linewidth=0.8, alpha=0.6)
	ax.set_xticks([-180, 0, 180])
	ax.set_xticklabels(['-180', '0', '180'], fontsize=6, rotation=90)
	ax.tick_params(axis='y', labelleft=False, labelright=False, length=0)
	ax.set_title(r'$\beta\ (^\circ)$', fontsize=8, pad=4)
	ax.grid(True, alpha=0.15)

# Axis Titles and Labels
ax_top_nw.set_ylabel('Depth [m]', fontsize=10, fontweight='bold')
ax_mid_nw.set_ylabel('Depth [m]', fontsize=10, fontweight='bold')
ax_mid_nw.set_xlabel('Bicoherence', fontsize=10)
ax_mid_w.set_xlabel('Bicoherence', fontsize=10)


# Panel ID Annotations
ax_top_nw.text(0.03, 1.05, '(a) $M_2+M_2$ No Wind', transform=ax_top_nw.transAxes, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
ax_top_w.text(0.03, 1.05, '(c) $M_2+M_2$ With Wind', transform=ax_top_w.transAxes, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
ax_mid_nw.text(0.03, 1.05, '(b) $M_2\\pm f$ No Wind', transform=ax_mid_nw.transAxes, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
ax_mid_w.text(0.03, 1.05, '(d) $M_2\\pm f$ With Wind', transform=ax_mid_w.transAxes, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

# Interactive Legend
# --- Extract handles and labels to build the unified Top Legend ---
handles, labels = ax_top_nw.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
		   ncol=4, fontsize='small', framealpha=0.9, shadow=True)

plt.savefig('bicoherence.png', dpi = 300)


########