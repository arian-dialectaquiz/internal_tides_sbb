import os
import xarray as xr
import dask
import xroms
from dask.diagnostics import ProgressBar
import numpy as np
import gc
from scipy.linalg import svd


# --- Paths ---
filename = '/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc' 
pressure_path = '/Users/piero/arian/data1/IT_outs/pressure/'  
out_path = '/Users/piero/arian/data1/IT_outs/tec/'
os.makedirs(out_path, exist_ok=True)

# --- Configuration ---
NUM_MODES = 5  
WINDOW_SIZE = 25  
T_M2 = 12.4206012  # hours
ETA_STEP = 10      # Reduced from 80 to make spatial chunks smaller and safer for RAM
eta_steps = np.arange(0, 1360, ETA_STEP) 
xis = slice(40, None)
TARGET_S_RHO = 1  

# 1. Open Dataset Lazily
ds1 = xr.open_dataset(filename, chunks={'ocean_time': WINDOW_SIZE, 's_rho': -1, 'eta_rho': "auto", 'xi_rho': "auto"})
ds, xgrid = xroms.roms_dataset(ds1)

# 2. Prepare Harmonic Projections
t_hours = (ds.ocean_time - ds.ocean_time[0]).astype(float) / 1e9 / 3600.0
omega_m2 = 2 * np.pi / T_M2
t_hours_np = t_hours.values  

# Keep velocity handles lazy
ub_all = xroms.to_rho(ds.ubar, xgrid)
vb_all = xroms.to_rho(ds.vbar, xgrid)
# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================
for i in range(len(eta_steps) - 1):
	print(f"--- Processing TEC Slice {i} (Rows {eta_steps[i]} to {eta_steps[i+1]}) ---")
	eta_slice = slice(eta_steps[i], eta_steps[i+1])
	ds_sub = ds.isel(eta_rho=eta_slice, xi_rho=xis)
	
	# --- FIXED FILE MAPPING ---
	# Find which 80-row pressure file contains our current 10-row slice
	p_idx = eta_steps[i] // 80
	p_file = os.path.join(pressure_path, f'p_bc_slice_{p_idx}.nc')
	
	if not os.path.exists(p_file):
		print(f"Warning: {p_file} not found. Skipping.")
		continue
		
	# Open the 80-row file lazily
	ds_p = xr.open_dataset(p_file, chunks={'ocean_time': WINDOW_SIZE})
	p_var = list(ds_p.data_vars)[0]
	
	# Lazily isolate ONLY the 10 matching eta rows inside the pressure dataset
	p_block_lazy = ds_p[p_var].sel(eta_rho=ds_sub.eta_rho)
	
	# Compute static topographic gradients for the 10 rows
	dhdx = (ds_sub.h.differentiate('xi_rho') * ds_sub.pm).values
	dhdy = (ds_sub.h.differentiate('eta_rho') * ds_sub.pn).values
	
	nt = len(ds_sub.ocean_time)
	n_windows = nt // WINDOW_SIZE
	n_eta = eta_steps[i+1] - eta_steps[i]  # Will be exactly 10
	n_xi = len(ds_sub.xi_rho)
	
	tec_out = np.zeros((NUM_MODES, n_windows, n_eta, n_xi), dtype=np.float32)
	time_coords = []
	
	# 4. Loop over Tidal Windows
	for w in range(n_windows):
		t_idx = slice(w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE)
		time_coords.append(ds_sub.ocean_time[w * WINDOW_SIZE + WINDOW_SIZE // 2].values)
		
		# Load only 25 frames for the 10 rows into RAM
		u_win = ub_all.isel(eta_rho=eta_slice, xi_rho=xis, ocean_time=t_idx).values
		v_win = vb_all.isel(eta_rho=eta_slice, xi_rho=xis, ocean_time=t_idx).values
		p_win = p_block_lazy.isel(ocean_time=t_idx).values   # (25, Depth, 10, Xi)
		dz_win = ds_sub.dz.isel(ocean_time=t_idx).values    # (25, Depth, 10, Xi)
		
		c_win = np.cos(omega_m2 * t_hours_np[t_idx])
		s_win = np.sin(omega_m2 * t_hours_np[t_idx])
		
		# Vectorized Velocity Harmonics
		uc = np.mean(u_win * c_win[:, None, None], axis=0) * 2
		us = np.mean(u_win * s_win[:, None, None], axis=0) * 2
		vc = np.mean(v_win * c_win[:, None, None], axis=0) * 2
		vs = np.mean(v_win * s_win[:, None, None], axis=0) * 2
		
		# Vectorized Modal Decomposition via Stacked SVD
		dz_m = np.mean(dz_win, axis=0)  
		weight = np.sqrt(dz_m)
		
		p_win_t = p_win.transpose(2, 3, 0, 1)
		weight_t = weight.transpose(1, 2, 0)
		
		p_weighted = p_win_t * weight_t[:, :, None, :]
		p_anom = p_weighted - np.mean(p_weighted, axis=2, keepdims=True)
		p_anom = np.nan_to_num(p_anom, nan=0.0)
		
		U, S, Vh = np.linalg.svd(p_anom, full_matrices=False)
		
		for n in range(NUM_MODES):
			U_n = U[..., :, n]            
			S_n = S[..., n, None]         
			Vh_n_bot = Vh[..., n, TARGET_S_RHO]  
			w_bot = weight_t[..., TARGET_S_RHO]  
			
			w_bot_safe = np.where(w_bot == 0, 1.0, w_bot)
			p_bot_n = (U_n * S_n) * (Vh_n_bot / w_bot_safe)[..., None]  
			
			pc_n = np.mean(p_bot_n * c_win[None, None, :], axis=2) * 2
			ps_n = np.mean(p_bot_n * s_win[None, None, :], axis=2) * 2
			
			term_x = 0.5 * (uc * pc_n + us * ps_n) * dhdx
			term_y = 0.5 * (vc * pc_n + vs * ps_n) * dhdy
			
			tec_out[n, w, :, :] = np.where(w_bot == 0, np.nan, term_x + term_y)
			
		del u_win, v_win, p_win, dz_win, p_anom, U, S, Vh
		
	ds_p.close()  # Free file handle cache

	print(f"--- Saving Modal TEC Slice {i} ---")
	da_tec = xr.DataArray(
		tec_out, 
		dims=['mode', 'ocean_time', 'eta_rho', 'xi_rho'],
		coords={'mode': np.arange(NUM_MODES), 'ocean_time': time_coords, 
				'eta_rho': ds_sub.eta_rho, 'xi_rho': ds_sub.xi_rho},
		name='topographic_energy_conversion'
	)
	
	# Saved using the matching 10-row loop slice ID index (0 to 134)
	save_file = os.path.join(out_path, f'tec_slice_{i}.nc')
	da_tec.to_netcdf(save_file)
	
	del tec_out, da_tec, dhdx, dhdy
	gc.collect()

ds1.close()
print("Processing complete!")






