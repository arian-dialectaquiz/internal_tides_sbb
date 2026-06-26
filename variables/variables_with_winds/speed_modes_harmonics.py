######

import os
import sys
import gc
import numpy as np
import xarray as xr
import xroms
from scipy import signal

# =============================================================================
# 1. SETUP & PATHS
# =============================================================================
filename = '/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc' 
out_path = '/Users/piero/arian/data1/IT_outs/velocity/'
os.makedirs(out_path, exist_ok=True)

# Standardized Micro-Slicing Configuration
ETA_STEP = 40      
eta_steps = np.arange(0, 1360, ETA_STEP) 
xis = slice(40, None)

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def get_bandpass_sos(T1, T2, dt_hours, order=4):
	"""Creates a bandpass filter SOS (Second-Order Sections) for a given range."""
	nyquist = 0.5 * (1 / dt_hours)
	f_low = (1 / T2) / nyquist
	f_high = (1 / T1) / nyquist
	sos = signal.butter(order, [f_low, f_high], btype='band', output='sos')
	return sos

def vectorized_m2_solve(t_days, data_matrix):
	"""Fits Mean, Linear Trend, and M2 Harmonic to a 2D matrix (Time, Points)."""
	Nt, Npts = data_matrix.shape
	omega_M2 = 2 * np.pi / (12.4206012 / 24.0) # rad/day
	
	A = np.column_stack([
		np.ones(Nt), t_days,
		np.cos(omega_M2 * t_days), np.sin(omega_M2 * t_days)
	])
	
	valid_mask = np.isfinite(data_matrix[0, :])
	m2_recon = np.full((Nt, Npts), np.nan, dtype=np.float32)
	mean_recon = np.full((Nt, Npts), np.nan, dtype=np.float32)
	
	if np.sum(valid_mask) == 0:
		return mean_recon, m2_recon

	Y = data_matrix[:, valid_mask]
	A_pinv = np.linalg.pinv(A) 
	coeffs = A_pinv @ Y
	
	part_mean = A[:, :2] @ coeffs[:2, :]
	part_m2 = A[:, 2:] @ coeffs[2:, :]
	
	mean_recon[:, valid_mask] = part_mean
	m2_recon[:, valid_mask] = part_m2
	
	return mean_recon, m2_recon

def apply_sos_filter_with_nans(data_flat, sos_filter):
	"""Safely applies SOS filter to flat arrays containing NaNs (land masks)."""
	mask_nan = np.isnan(data_flat)
	data_safe = np.copy(data_flat)
	data_safe[mask_nan] = 0.0 # Safety for filter
	
	filtered = signal.sosfiltfilt(sos_filter, data_safe, axis=0)
	filtered[mask_nan] = np.nan # Restore NaNs
	return filtered

# =============================================================================
# 3. INITIALIZATION & FILTER PREPARATION
# =============================================================================
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})
ds, xgrid = xroms.roms_dataset(ds1)


T_M2 = 12.4206012 # M2 tidal period in hours
freq_M2 = 1.0 / T_M2

# Prepare Time Coordinates
t_roms = ds.ocean_time.values
t_days = (t_roms - t_roms[0]).astype(float) / 1e9 / 86400.0
t_hours = (t_roms - t_roms[0]).astype(float) / 1e9 / 3600.0
dt_hours = t_hours[1] - t_hours[0]


# Create Grid-Transformed Lazy Objects
u_rho = xroms.to_rho(ds.u, xgrid)
v_rho = xroms.to_rho(ds.v, xgrid)
w_rho = xroms.to_s_rho(ds.w, xgrid)
ubar_rho = xroms.to_rho(ds.ubar, xgrid)
vbar_rho = xroms.to_rho(ds.vbar, xgrid)

# Prepare 4D Cosine Broadcaster
omega_m2 = 2 * np.pi / T_M2
cos_term_4d = np.cos(omega_m2 * t_hours)[:, None, None, None]

# =============================================================================
# 4. MAIN SPLICED PROCESSING LOOP
# =============================================================================
for i in range(len(eta_steps) - 1):
	print(f'\n=== Processing Velocity Slice {i}/{len(eta_steps)-1} (Rows {eta_steps[i]} to {eta_steps[i+1]}) ===')
	eta_slice = slice(eta_steps[i], eta_steps[i+1])
	results = {}

	# Gather precise spatial coordinates for metadata completeness
	u_sub_coords = u_rho.isel(eta_rho=eta_slice, xi_rho=xis)
	coords_to_keep = {
		'ocean_time': ds1.ocean_time,
		's_rho': ds1.s_rho,
		'eta_rho': u_sub_coords.eta_rho,
		'xi_rho': u_sub_coords.xi_rho
	}

	# --- PROCESS U COMPONENT ---
	print("   > Working U...")
	u_vals = u_rho.isel(eta_rho=eta_slice, xi_rho=xis).values
	ubar_vals = ubar_rho.isel(eta_rho=eta_slice, xi_rho=xis).values
	nt, nz, ny, nx = u_vals.shape
	
	u_bc = u_vals - ubar_vals[:, None, :, :]
	del u_vals, ubar_vals
	
	u_flat = u_bc.reshape(nt, -1)
	u_m2 = u_bc * cos_term_4d

	#u_ni = apply_sos_filter_with_nans(u_flat, ni_sos_filter).reshape(nt, nz, ny, nx)
	#u_plus = apply_sos_filter_with_nans(u_flat, plus_sos_filter).reshape(nt, nz, ny, nx)
	#u_minus = apply_sos_filter_with_nans(u_flat, minus_sos_filter).reshape(nt, nz, ny, nx)
	
	results['u_bc'] = u_bc.astype(np.float32)
	results['u_m2'] = u_m2.astype(np.float32)
	#results['u_ni'] = u_ni.astype(np.float32)
	#results['u_plus'] = u_plus.astype(np.float32)
	#results['u_minus'] = u_minus.astype(np.float32)
	del u_bc, u_flat, u_m2; gc.collect()

	# --- PROCESS V COMPONENT ---
	print("   > Working V...")
	v_vals = v_rho.isel(eta_rho=eta_slice, xi_rho=xis).values
	vbar_vals = vbar_rho.isel(eta_rho=eta_slice, xi_rho=xis).values
	
	v_bc = v_vals - vbar_vals[:, None, :, :]
	del v_vals, vbar_vals
	
	v_flat = v_bc.reshape(nt, -1)
	v_m2 = v_bc * cos_term_4d
	
	#v_ni = apply_sos_filter_with_nans(v_flat, ni_sos_filter).reshape(nt, nz, ny, nx)
	#v_plus = apply_sos_filter_with_nans(v_flat, plus_sos_filter).reshape(nt, nz, ny, nx)
	#v_minus = apply_sos_filter_with_nans(v_flat, minus_sos_filter).reshape(nt, nz, ny, nx)
	
	results['v_bc'] = v_bc.astype(np.float32)
	results['v_m2'] = v_m2.astype(np.float32)
	#results['v_ni'] = v_ni.astype(np.float32)
	#results['v_plus'] = v_plus.astype(np.float32)
	#results['v_minus'] = v_minus.astype(np.float32)
	del v_bc, v_flat, v_m2; gc.collect()

	# --- PROCESS W COMPONENT ---
	print("   > Working W...")
	w_vals = w_rho.isel(eta_rho=eta_slice, xi_rho=xis).values
	
	w_bc = w_vals
	w_flat = w_vals.reshape(nt, -1)
	w_m2 = w_bc * cos_term_4d
	
	#w_ni = apply_sos_filter_with_nans(w_flat, ni_sos_filter).reshape(nt, nz, ny, nx)
	#w_plus = apply_sos_filter_with_nans(w_flat, plus_sos_filter).reshape(nt, nz, ny, nx)
	#w_minus = apply_sos_filter_with_nans(w_flat, minus_sos_filter).reshape(nt, nz, ny, nx)
#	
	results['w_bc'] = w_vals.astype(np.float32)
	results['w_m2'] = w_m2.astype(np.float32)
	#results['w_ni'] = w_ni.astype(np.float32)
	#results['w_plus'] = w_plus.astype(np.float32)
	#results['w_minus'] = w_minus.astype(np.float32)
	del w_vals, w_flat, w_bc, w_m2; gc.collect()

	# =============================================================================
	# 5. PACKAGING AND SAVING
	# =============================================================================
	print("   > Packaging Dataset...")
	dims = ('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
	
	ds_out = xr.Dataset(
		data_vars={
			'u_bc': (dims, results['u_bc']), 'u_m2': (dims, results['u_m2']),		
			'v_bc': (dims, results['v_bc']), 'v_m2': (dims, results['v_m2']),			
			'w_bc': (dims, results['w_bc']), 'w_m2': (dims, results['w_m2']),		
		},
		coords=coords_to_keep
	)
	
	# Metadata Attributes
	ds_out.attrs['description'] = 'Baroclinic, M2 Velocities'
		
	# NetCDF compression encoding setup
	comp = dict(zlib=True, complevel=1, _FillValue=np.nan)
	encoding = {var: comp for var in ds_out.data_vars}
	
	# Saved using sequential index string to map seamlessly into the energy flux script
	filename_out = os.path.join(out_path, f"velocity_decomp_rho_{i}.nc")
	print("   > Saving to disk...")
	ds_out.to_netcdf(filename_out, encoding=encoding, engine='netcdf4')
	
	del results, ds_out, u_sub_coords
	gc.collect()
	print(f"   > Successfully Saved: {filename_out}")

ds1.close()
print("\nAll velocity decomposition slices processed and saved successfully.")

