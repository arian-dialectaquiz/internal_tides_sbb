#####

import os
import glob
import re
import xarray as xr
import dask
import xroms
import numpy as np
import gc

# --- Paths & Directories ---
s_path = '/Users/piero/arian/data1/IT_outs/velocity/'
filename = '/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc' 
out_path = '/Users/piero/arian/data1/IT_outs/ke_time_modes/'
os.makedirs(out_path, exist_ok=True)

# --- Configuration ---
NUM_MODES = 5  
WINDOW_SIZE = 24 
ETA_STEP = 40      
eta_steps = np.arange(0, 1360, ETA_STEP) 
xis = slice(40, None)

# 1. Open Multi-File Velocity and Base Grid Datasets Lazily
s_f = sorted(glob.glob(os.path.join(s_path, 'velocity_decomp_rho_*.nc')))
ss = xr.open_mfdataset(s_f, combine='nested', concat_dim='eta_rho', parallel=True, chunks={'ocean_time': WINDOW_SIZE})

ds1 = xr.open_dataset(filename, chunks={'ocean_time': WINDOW_SIZE, 's_rho': -1, 'eta_rho': "auto", 'xi_rho': "auto"})
ds, xgrid = xroms.roms_dataset(ds1)

# =============================================================================
# 2. MAIN UNIFIED SPLICED LOOP (TIME-DEPENDENT KE)
# =============================================================================
for i in range(len(eta_steps) - 1):
	eta_start = eta_steps[i]
	eta_end = eta_steps[i+1]
	print(f"--- Processing KE Slice {i} (Rows {eta_start} to {eta_end}) ---")
	eta_slice = slice(eta_start, eta_end)
	
	# Subset grid data and velocity datasets lazily (Syncing horizontal dimensions)
	ds_sub = ds.isel(eta_rho=eta_slice, xi_rho=xis)
	u_sub = ss.u_m2.isel(eta_rho=eta_slice)
	v_sub = ss.v_m2.isel(eta_rho=eta_slice)
	dz_sub = ds.dz.isel(eta_rho=eta_slice, xi_rho=xis)
	
	# Explicit coordinate synchronization
	u_sub = u_sub.assign_coords({'eta_rho': dz_sub.eta_rho, 'xi_rho': dz_sub.xi_rho})
	v_sub = v_sub.assign_coords({'eta_rho': dz_sub.eta_rho, 'xi_rho': dz_sub.xi_rho})

	# Array dimension parsing
	nt = len(u_sub.ocean_time)
	n_windows = nt // WINDOW_SIZE
	n_depth = len(u_sub.s_rho)
	n_eta = eta_end - eta_start  
	n_xi = len(u_sub.xi_rho)
	
	# Allocate memory blocks tracking full time-series matching your dimensions
	ke_total_out = np.zeros((n_windows * WINDOW_SIZE, n_depth, n_eta, n_xi), dtype=np.float32)
	ke_modes_out = np.zeros((NUM_MODES, n_windows * WINDOW_SIZE, n_depth, n_eta, n_xi), dtype=np.float32)
	#ke_modes_int_out = np.zeros((NUM_MODES, n_windows * WINDOW_SIZE, n_eta, n_xi), dtype=np.float32)
	time_coords = []

	# 3. Loop over Tidal Windows
	for w in range(n_windows):
		t_idx = slice(w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE)
		out_t_idx = slice(w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE)
		time_coords.extend(u_sub.ocean_time[t_idx].values)
		
		# Pull array metrics into active RAM frame windows
		u_win = u_sub.isel(ocean_time=t_idx).values
		v_win = v_sub.isel(ocean_time=t_idx).values
		dz_win = dz_sub.isel(ocean_time=t_idx).values
		
		# Calculate 3D Total Kinetic Energy for this window frame
		ke_total_out[out_t_idx, ...] = 1025 * 0.5 * (u_win**2 + v_win**2)
		
		# SVD Depth Matrix Weighting setup
		dz_m = np.mean(dz_win, axis=0)  
		weight = np.sqrt(dz_m)
		
		# Transpose profiles to position Space (Eta, Xi) first, then internal (Time, Depth)
		u_win_t = u_win.transpose(2, 3, 0, 1)
		v_win_t = v_win.transpose(2, 3, 0, 1)
		weight_t = weight.transpose(1, 2, 0)
		
		u_weighted = np.nan_to_num(u_win_t * weight_t[:, :, None, :], nan=0.0)
		v_weighted = np.nan_to_num(v_win_t * weight_t[:, :, None, :], nan=0.0)
		
		u_anom = u_weighted - np.mean(u_weighted, axis=2, keepdims=True)
		v_anom = v_weighted - np.mean(v_weighted, axis=2, keepdims=True)
		
		# Run SVD on velocity structures to establish physical baroclinic structures
		U_u, S_u, Vh_u = np.linalg.svd(u_anom, full_matrices=False)
		U_v, S_v, Vh_v = np.linalg.svd(v_anom, full_matrices=False)
		
		for n in range(NUM_MODES):
			# Extract structures and build unweighted modal profiles (phi)
			Vh_un = Vh_u[..., n, :]
			Vh_vn = Vh_v[..., n, :]
			phi_un = Vh_un / np.where(weight_t == 0, 1.0, weight_t)
			phi_vn = Vh_vn / np.where(weight_t == 0, 1.0, weight_t)
			
			# Extract time-dependent modal amplitude coefficients
			u_amp = np.sum(u_win_t * Vh_un[:, :, None, :] * weight_t[:, :, None, :], axis=-1)
			v_amp = np.sum(v_win_t * Vh_vn[:, :, None, :] * weight_t[:, :, None, :], axis=-1)
			
			# Reconstruct 3D physical velocity fields belonging to Mode n
			u_modal = (u_amp[..., None] * phi_un[:, :, None, :]).transpose(2, 3, 0, 1)
			v_modal = (v_amp[..., None] * phi_vn[:, :, None, :]).transpose(2, 3, 0, 1)
			
			# Compute 3D and Depth-Integrated Modal Kinetic Energy structures
			ke_modes_out[n, out_t_idx, ...] = 1025 * 0.5 * (u_modal**2 + v_modal**2)
			#ke_modes_int_out[n, out_t_idx, ...] = 1025 * 0.5 * (u_amp**2 + v_amp**2).transpose(2, 0, 1)
			
		del u_win, v_win, dz_win, u_anom, v_anom, U_u, S_u, Vh_u, U_v, S_v, Vh_v
		
	# =============================================================================
	# 4. COMPILING LIGHTWEIGHT NETCDF OUTPUTS PER SLICE
	# =============================================================================
	print(f"    > Saving SVD-Decoded KE Slice {i}...")
	
	ds_out = xr.Dataset(
		{
			'ke_total': (['ocean_time', 's_rho', 'eta_rho', 'xi_rho'], ke_total_out),
			'ke_modes': (['mode', 'ocean_time', 's_rho', 'eta_rho', 'xi_rho'], ke_modes_out)
			
		}, 
		coords={
			'mode': np.arange(NUM_MODES),
			'ocean_time': time_coords,
			's_rho': u_sub.s_rho,
			'eta_rho': u_sub.eta_rho,
			'xi_rho': u_sub.xi_rho
		}
	)
	
	save_file = os.path.join(out_path, f'ke_modes_slice_{i}.nc')
	encoding = {
		'ke_total': {'zlib': True, 'complevel': 1},
		'ke_modes': {'zlib': True, 'complevel': 1}	
	}
	
	ds_out.to_netcdf(save_file, encoding=encoding)
	
	# Wipe tracking arrays to drop RAM footprints completely before moving on
	del ke_total_out, ke_modes_out, ds_out
	gc.collect()

ds1.close()
ss.close()
print("All velocity slices successfully decomposed into time-series modal KE datasets!")




