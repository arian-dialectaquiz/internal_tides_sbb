######

import sys
from pathlib import Path
import os
import numpy as np
import xarray as xr
import xroms
import dask
import pandas as pd
import gc
import scipy.signal as signal

# ---- Bandpass filter parameters ----#
def filter_da_bandpass(da, T1, T2, fs, time_dim='ocean_time', order=4):
	"""
	Apply a Butterworth bandpass filter along the time dimension of a 4D DataArray.

	Parameters:
	-----------
	da : xr.DataArray
		The input 4D DataArray with dimensions (time, s_rho, eta_rho, xi_rho)
	T1, T2 : float
		Period limits in hours (e.g., 28h and 44h)
	time_dim : str
		The name of the time dimension
	order : int
		The order of the Butterworth filter

	Returns:
	--------
	xr.DataArray
		The filtered 4D DataArray, same dims as input
	"""
	# Get time step in hours
	time_vals = da[time_dim].values
	dt_hours = (time_vals[1] - time_vals[0]) / np.timedelta64(1, 'h')
	#fs = 1 / dt_hours  # sampling frequency in cycles per hour

	# Bandpass frequency range in cph
	f_low = 1 / T2  # Higher period = lower frequency
	f_high = 1 / T1  # Lower period = higher frequency

	# Normalised frequencies
	nyq = 0.5 * fs
	low = f_low / nyq
	high = f_high / nyq

	# Butterworth bandpass
	b, a = signal.butter(order, [low, high], btype='band')

	# Reshape to (time, -1) for filtering
	original_shape = da.shape
	reshaped = da.data.reshape((original_shape[0], -1))

	# Apply zero-phase filter along time axis
	filtered = signal.filtfilt(b, a, reshaped, axis=0, padlen=3*max(len(b), len(a)))

	# Restore shape and return as DataArray
	da_filtered = xr.DataArray(
		filtered.reshape(original_shape),
		dims=da.dims,
		coords=da.coords,
		attrs=da.attrs
	)

	return da_filtered


# Configuration
out_path = '/Users/piero/arian/data1/IT_outs/temperature/'

filename = f'/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc'
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 1})

# Define eta and xi slices
eta = np.arange(0, 1280, 1)
etas = np.reshape(eta, (32, 40))
xis = slice(40, None)    
ds, xgrid = xroms.roms_dataset(ds1)

fs = 1 
T1 = 11
T2 = 13


# Loop over all spatial slice groups
for i in range(etas.shape[0]):
	print(f"\n=========================================")
	print(f"Processing spatial block {i} (eta: {etas[i][0]} to {etas[i][-1]})")
	print(f"=========================================")
	
	eta_slice = slice(etas[i][0], etas[i][-1])
	
	# 1. Bring the full time series for this spatial slice entirely into memory 
	print("Loading full time-series slice for theta and dz...")
	theta_full = ds.temp.isel(eta_rho=eta_slice, xi_rho=xis).compute()
	dz_full = ds.dz.isel(eta_rho=eta_slice, xi_rho=xis).compute()
	
	# 2. Compute vertical gradient dT/dz before filtering
	print("Computing vertical differential (dT/dz)...")
	dtheta = theta_full.diff(dim='s_rho')
	
	# Align dz grid size to the differential midpoint dimensions (N-1)
	if 's_rho' in dz_full.dims:
		dz_full = dz_full.isel(s_rho=slice(None, dtheta.sizes['s_rho']))
		
	dT_dz_unfiltered = (dtheta / dz_full).transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
	
	# Clean up intermediate differential variables
	del dtheta, dz_full
	gc.collect()
	
	# 3. Apply the bandpass filter across the continuous time axis
	print("Applying band-pass filter to full continuous theta array...")
	theta_prime = filter_da_bandpass(theta_full, T1, T2, fs).transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
	
	print("Applying band-pass filter to full continuous dT/dz array...")
	dT_dz_prime = filter_da_bandpass(dT_dz_unfiltered, T1, T2, fs).transpose('ocean_time', 's_rho', 'eta_rho', 'xi_rho')
	
	# 4. Save the final results directly to disk without temporal partitioning
	print("Saving band-pass filtered files...")
	# Process theta_prime
	ds_theta = theta_prime.to_dataset(name='theta_prime')
	ds_theta = ds_theta.drop_vars('z_rho', errors='ignore')

	# Process dT_dz_prime
	ds_dtdz = dT_dz_prime.to_dataset(name='dT_dz_prime')
	ds_dtdz = ds_dtdz.drop_vars('z_rho', errors='ignore')

	theta_prime_file = os.path.join(out_path, f'theta_prime_slice_{i}.nc')
	ds_theta.to_netcdf(theta_prime_file)
	
	dT_dz_prime_file = os.path.join(out_path, f'dT_dz_prime_slice_{i}.nc')
	ds_dtdz.to_netcdf(dT_dz_prime_file)
	
	del theta_full, dT_dz_unfiltered, theta_prime, dT_dz_prime, ds_theta, ds_dtdz, 
	gc.collect()

del ds, ds1, xgrid
gc.collect()
print("All slices filtered and processed cleanly.")



########