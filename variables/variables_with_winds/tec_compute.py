
import os
import xarray as xr
import dask
import xroms
from dask.diagnostics import ProgressBar
import numpy as np
import gc
from scipy.linalg import svd

filename = '/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc' 
pressure_path = '/Users/piero/arian/pressure/'  # Directory containing p_bc_slice_*.nc
out_path = '/Users/piero/arian/data1/IT_outs/tec/'
os.makedirs(out_path, exist_ok=True)


# --- Configuration ---
g = 9.81
NUM_MODES = 5
WINDOW_SIZE = 25  # 2 tidal cycles for cleaner fit
T_M2 = 12.4206012  # hours
eta_steps = np.arange(0, 1360, 80) # Smaller steps for modal decomposition
xis = slice(40, None)
# 1. Open Dataset Lazily
ds1 = xr.open_dataset(filename, chunks={'ocean_time': "auto", 's_rho': -1, 'eta_rho': "auto", 'xi_rho': "auto"})
ds, xgrid = xroms.roms_dataset(ds1)

# 2. Prepare Harmonic Projections
t_hours = (ds.ocean_time - ds.ocean_time[0]).astype(float) / 1e9 / 3600.0
omega_m2 = 2 * np.pi / T_M2
cos_proj = np.cos(omega_m2 * t_hours).values # Small enough to keep in RAM
sin_proj = np.sin(omega_m2 * t_hours).values

# Pre-calculate velocity components lazily on rho grid
ub_all = xroms.to_rho(ds.ubar, xgrid)
vb_all = xroms.to_rho(ds.vbar, xgrid)


def compute_modal_harmonics(p_block, dz_block, c_proj, s_proj, n_modes=5):
	"""
	p_block: (12, Depth)
	c_proj: (12,)
	s_proj: (12,)
	Returns: p_cos_n(modes), p_sin_n(modes)
	"""
	dz_m = np.mean(dz_block, axis=0)
	weight = np.sqrt(dz_m)
	p_anom = (p_block * weight) - np.mean(p_block * weight, axis=0)
	
	p_cos_n = np.zeros(n_modes)
	p_sin_n = np.zeros(n_modes)
	
	try:
		U, S, Vh = svd(p_anom, full_matrices=False)
		for n in range(n_modes):
			# Reconstruct bottom pressure time series for mode n (12 points)
			p_bot_n = (U[:, n] * S[n]) * (Vh[n, 0] / weight[0])
			# Project to harmonics
			p_cos_n[n] = np.mean(p_bot_n * c_proj) * 2
			p_sin_n[n] = np.mean(p_bot_n * s_proj) * 2
	except:
		pass
	return p_cos_n, p_sin_n

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================
for i in range(len(eta_steps)):
	print(f"--- Processing Slice {i} ---")
	eta_slice = slice(eta_steps[i], eta_steps[i+1])
	ds_sub = ds.isel(eta_rho=eta_slice, xi_rho=xis)
	
	# 1. Prepare Velocity & Gradients (Full time series for the slice)
	dhdx = (ds_sub.h.differentiate('xi_rho') * ds_sub.pm).values
	dhdy = (ds_sub.h.differentiate('eta_rho') * ds_sub.pn).values
	print(f"--- Preparing dz ---")
	# 2. Prepare Pressure & DZ
	dz_all = ds_sub.dz.values
	# 1. Load Precomputed Full-Depth Baroclinic Pressure Slice
	p_file = os.path.join(pressure_path, f'p_bc_slice_{i}.nc')
	if not os.path.exists(p_file):
		print(f"Warning: {p_file} not found. Skipping.")
		continue
		
	with xr.open_dataset(p_file) as ds_p:
		p_var = list(ds_p.data_vars)[0]
		p_bc_all = ds_p[p_var].values  # Keep full depth profile: (nt, n_depth, n_eta, n_xi)
	
	print(f"--- Preparing window ---")
	# 3. Setup Windowing
	nt, n_eta, n_xi = u_all.shape
	n_windows = nt // WINDOW_SIZE
	tec_out = np.zeros((NUM_MODES, n_windows, n_eta, n_xi), dtype=np.float32)
	time_coords = []
	print(f"--- loop over tidal windows ---")
	# 4. Loop over Tidal Windows
	for w in range(n_windows):
		t_idx = slice(w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE)
		time_coords.append(ds_sub.ocean_time[w * WINDOW_SIZE + WINDOW_SIZE // 2].values)
		
		# Harmonic basis for this window
		c_win = np.cos(omega_m2 * t_hours_all[t_idx]).values
		s_win = np.sin(omega_m2 * t_hours_all[t_idx]).values
		
		for ey in range(n_eta):
			for ex in range(n_xi):
				if np.isnan(p_bc_all[0, 0, ey, ex]): continue
				
				# A. Velocity Harmonics for this window
				uc = np.mean(u_all[t_idx, ey, ex] * c_win) * 2
				us = np.mean(u_all[t_idx, ey, ex] * s_win) * 2
				vc = np.mean(v_all[t_idx, ey, ex] * c_win) * 2
				vs = np.mean(v_all[t_idx, ey, ex] * s_win) * 2
				
				# B. Pressure Modal Harmonics for this window
				p_cos_n, p_sin_n = compute_modal_harmonics(
					p_bc_all[t_idx, :, ey, ex], dz_all[t_idx, :, ey, ex], c_win, s_win, n_modes=NUM_MODES
				)
				print(f"--- multiplying tec ---")
				# C. Combine: 0.5 * ( [u_cos*dhdx + v_cos*dhdy]*p_cos + [u_sin*dhdx + v_sin*dhdy]*p_sin )
				term_cos = (uc * dhdx[ey, ex] + vc * dhdy[ey, ex])
				term_sin = (us * dhdx[ey, ex] + vs * dhdy[ey, ex])
				
				for n in range(NUM_MODES):
					tec_out[n, w, ey, ex] = 0.5 * (term_cos * p_cos_n[n] + term_sin * p_sin_n[n])
	print(f"--- saving tec slice {i} ---")
	# 5. Save Modal Result for the Slice
	da_tec = xr.DataArray(
		tec_out, 
		dims=['mode', 'ocean_time', 'eta_rho', 'xi_rho'],
		coords={'mode': np.arange(NUM_MODES), 'ocean_time': time_coords, 
				'eta_rho': ds_sub.eta_rho, 'xi_rho': ds_sub.xi_rho},
		name='tec_harmonic'
	)
	
	save_file = os.path.join(out_path, f'tec_slice_{i}.nc')
	da_tec.to_netcdf(save_file)
	
	del u_all, v_all, p_bc_all, dz_all, tec_out, da_tec
	gc.collect()

ds1.close()