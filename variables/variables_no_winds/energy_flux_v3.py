######

import os
import glob
import gc
import numpy as np
import xarray as xr
import xroms

# =============================================================================
# 1. SETUP & ALIGNMENT
# =============================================================================
g = 9.81
rho0 = 1025.0
filename = '/Users/piero/arian/data1/nc_outs/avg_internal_tides_paper.nc' 
s_path = '/Users/piero/arian/data1/NO_WINDS_IT/velocity/'
out_path = '/Users/piero/arian/data1/NO_WINDS_IT/flux/'
pressure_path =  '/Users/piero/arian/data1/NO_WINDS_IT/pressure/'

os.makedirs(out_path, exist_ok=True)

# Uniform Configuration
NUM_MODES = 5  
WINDOW_SIZE = 8 #24 h hours  
T_M2 = 12.4206012  # hours
omega_m2 = 2 * np.pi / T_M2

# Standardized Micro-Slicing Setup (10-row speed spacing)
ETA_STEP = 40      
eta_steps = np.arange(0, 1360, ETA_STEP) 
xis = slice(40, None)

# Open Multi-File and Base Datasets Lazily
ds1 = xr.open_dataset(filename, chunks={'ocean_time': WINDOW_SIZE, 's_rho': -1, 'eta_rho': "auto", 'xi_rho': "auto"})
ds, xgrid = xroms.roms_dataset(ds1)

s_f = sorted(glob.glob(os.path.join(s_path, 'velocity_decomp_rho_*.nc')))
ss = xr.open_mfdataset(s_f, combine='nested', concat_dim='eta_rho', parallel=True, chunks={'ocean_time': WINDOW_SIZE})

# Universal timeline base 
t_roms = ss.ocean_time.values
t_hours_np = (t_roms - t_roms[0]).astype(float) / 1e9 / 3600.0

# =============================================================================
# 2. MAIN UNIFIED SPLICED LOOP
# =============================================================================
for i in range(len(eta_steps) - 1):
	eta_start = eta_steps[i]
	eta_end = eta_steps[i+1]
	print(f"--- Processing Flux Slice {i} (Rows {eta_start} to {eta_end}) ---")
	eta_slice = slice(eta_start, eta_end)
	
		
	# Subset grid data and velocity datasets lazily
	ds_sub = ds.isel(eta_rho=eta_slice, xi_rho=xis)
	u_sub = ss.u_m2.isel(eta_rho=eta_slice)
	v_sub = ss.v_m2.isel(eta_rho=eta_slice)
	dz_sub = ds.dz.isel(eta_rho=eta_slice, xi_rho=xis)
	
	# Explicit spatial metadata coordinate sync to safely line calculations up
	u_sub = u_sub.assign_coords({'eta_rho': dz_sub.eta_rho, 'xi_rho': dz_sub.xi_rho})
	v_sub = v_sub.assign_coords({'eta_rho': dz_sub.eta_rho, 'xi_rho': dz_sub.xi_rho})
	# ==========================
	# Compute p_bc INLINE
	# ==========================
	#rho_sub = ds_sub.rho
#
	#p_total = (
		#g * rho_sub * dz_sub
	#).reindex(
		#s_rho=ds_sub.s_rho[::-1]
	#).cumsum(
		#dim='s_rho'
	#).reindex(
		#s_rho=ds_sub.s_rho
	#)
#
	#p_bc_sub = p_total - p_total.mean(dim='s_rho')
	#p_bc_sub = p_bc_sub.assign_coords({
		#'eta_rho': dz_sub.eta_rho,
		#'xi_rho': dz_sub.xi_rho
	#})
	# ==========================
	# Oppening pbc
	# ==========================
	p_file = os.path.join(pressure_path, f'p_bc_slice_{i}.nc')
	p_bc_sub = xr.open_dataset(p_file).p_bc

	# Array dimension parsing
	nt = len(u_sub.ocean_time)
	n_windows = nt // WINDOW_SIZE
	n_depth = len(u_sub.s_rho)
	n_eta = eta_end - eta_start  # Exactly 10
	n_xi = len(u_sub.xi_rho)
	
	# Allocate empty memory blocks for current 10-row block calculations
	fx_out = np.zeros((NUM_MODES, n_windows * WINDOW_SIZE, n_depth, n_eta, n_xi), dtype=np.float32)
	fy_out = np.zeros((NUM_MODES, n_windows * WINDOW_SIZE, n_depth, n_eta, n_xi), dtype=np.float32)
	fx_int_out = np.zeros((NUM_MODES, n_windows * WINDOW_SIZE, n_eta, n_xi), dtype=np.float32)
	fy_int_out = np.zeros((NUM_MODES, n_windows * WINDOW_SIZE, n_eta, n_xi), dtype=np.float32)
	time_coords = []

	# 3. Loop over Tidal Windows
	for w in range(n_windows):
		t_idx = slice(w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE)
		time_coords.extend(u_sub.ocean_time[t_idx].values)
		
		# Pull array metrics into active memory frame windows
		u_win = u_sub.isel(ocean_time=t_idx).values
		v_win = v_sub.isel(ocean_time=t_idx).values
		dz_win = dz_sub.isel(ocean_time=t_idx).values
		p_bc = p_bc_sub.isel(ocean_time=t_idx).values
		
		# Window Harmonic projection setup
		c_win = np.cos(omega_m2 * t_hours_np[t_idx])
		s_win = np.sin(omega_m2 * t_hours_np[t_idx])
		
		# M2 Tidal Filter Extraction
		p_cos = np.mean(p_bc * c_win[:, None, None, None], axis=0) * 2
		p_sin = np.mean(p_bc * s_win[:, None, None, None], axis=0) * 2
		p_m2 = (p_cos * c_win[:, None, None, None]) + (p_sin * s_win[:, None, None, None])
		
		# SVD Setup via Axis Transformations
		dz_m = np.mean(dz_win, axis=0)  
		weight = np.sqrt(dz_m)
		
		p_win_t = p_bc.transpose(2, 3, 0, 1)  
		weight_t = weight.transpose(1, 2, 0)  
		
		p_weighted = p_win_t * weight_t[:, :, None, :]
		p_anom = p_weighted - np.mean(p_weighted, axis=2, keepdims=True)
		p_anom = np.nan_to_num(p_anom, nan=0.0)
		
		# Micro-matrix SVD execution block
		U, S, Vh = np.linalg.svd(p_anom, full_matrices=False)
		
		p_m2_t = p_m2.transpose(2, 3, 0, 1)
		u_win_t = u_win.transpose(2, 3, 0, 1)
		v_win_t = v_win.transpose(2, 3, 0, 1)
		
		out_t_idx = slice(w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE)
		
		for n in range(NUM_MODES):
			Vh_n = Vh[..., n, :]  
			phi_n = Vh_n / np.where(weight_t == 0, 1.0, weight_t)  
			
			# Extract time-dependent modal amplitude coefficients
			p_amp = np.sum(p_m2_t * Vh_n[:, :, None, :] * weight_t[:, :, None, :], axis=-1)
			u_amp = np.sum(u_win_t * Vh_n[:, :, None, :] * weight_t[:, :, None, :], axis=-1)
			v_amp = np.sum(v_win_t * Vh_n[:, :, None, :] * weight_t[:, :, None, :], axis=-1)
			
			# Reconstruct isolated physical modal fields
			p_modal = (p_amp[..., None] * phi_n[:, :, None, :]).transpose(2, 3, 0, 1)
			u_modal = (u_amp[..., None] * phi_n[:, :, None, :]).transpose(2, 3, 0, 1)
			v_modal = (v_amp[..., None] * phi_n[:, :, None, :]).transpose(2, 3, 0, 1)
			
			# Assign modal flux components
			fx_out[n, out_t_idx, ...] = u_modal * p_modal
			fy_out[n, out_t_idx, ...] = v_modal * p_modal
			
			# Capture depth integration via amplitude orthogonality mappings
			fx_int_out[n, out_t_idx, ...] = (u_amp * p_amp).transpose(2, 0, 1)
			fy_int_out[n, out_t_idx, ...] = (v_amp * p_amp).transpose(2, 0, 1)
			
		del u_win, v_win, dz_win, p_bc, p_m2, p_anom, U, S, Vh
		


	# =============================================================================
	# 4. HORIZONTAL DIVERGENCE & NETCDF OUTPUT
	# =============================================================================
	print("    > Calculating Spatial Divergence for this 10-row slice...")
	
	# Turn integrated variables into Xarray objects to leverage .differentiate()
	da_fx_int = xr.DataArray(
		fx_int_out, dims=['mode', 'ocean_time', 'eta_rho', 'xi_rho'],
		coords={'mode': np.arange(NUM_MODES), 'ocean_time': time_coords, 'eta_rho': u_sub.eta_rho, 'xi_rho': u_sub.xi_rho}
	)
	da_fy_int = xr.DataArray(
		fy_int_out, dims=['mode', 'ocean_time', 'eta_rho', 'xi_rho'],
		coords={'mode': np.arange(NUM_MODES), 'ocean_time': time_coords, 'eta_rho': u_sub.eta_rho, 'xi_rho': u_sub.xi_rho}
	)
	
	# Differentiate across coordinates using spatial metrics from ds_sub
	div_Fx = da_fx_int.differentiate('xi_rho') * ds_sub.pm
	div_Fy = da_fy_int.differentiate('eta_rho') * ds_sub.pn
	div_Fbc = (div_Fx + div_Fy).astype('float32')
	div_Fbc.name = 'div_Fbc'
	
	# Compile the final comprehensive NetCDF structure
	ds_out = xr.Dataset({
		'Fx': (['mode', 'ocean_time', 's_rho', 'eta_rho', 'xi_rho'], fx_out),
		'Fy': (['mode', 'ocean_time', 's_rho', 'eta_rho', 'xi_rho'], fy_out),
		'div_Fbc': div_Fbc
	}, coords={
		'mode': np.arange(NUM_MODES),
		'ocean_time': time_coords,
		's_rho': u_sub.s_rho,
		'eta_rho': u_sub.eta_rho,
		'xi_rho': u_sub.xi_rho
	})
	
	save_file = os.path.join(out_path, f'fbc_slice_{i}.nc')
	encoding = {
		'Fx': {'zlib': True, 'complevel': 1},
		'Fy': {'zlib': True, 'complevel': 1},
		'div_Fbc': {'zlib': True, 'complevel': 1}
	}
	
	ds_out.to_netcdf(save_file, encoding=encoding)
	print(f"    > Saved Slice: {save_file}")
	
	# Wipe memory structures clean before processing the next block
	del fx_out, fy_out, fx_int_out, fy_int_out, da_fx_int, da_fy_int, div_Fx, div_Fy, div_Fbc, ds_out
	gc.collect()

ds1.close()
ss.close()
print("All slices processed successfully with synchronized indexing variables!")


