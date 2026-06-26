
#import sys
# Assuming the necessary imports (numpy, xarray) are handled within utils_roms_p1 or implicitly used by the code

#p1_dir = '/Users/piero/arian/dd_waves/IT_paper/'
#sys.path.append(p1_dir)
#from utils_roms_p1 import * # Assumes this includes numpy (np) and xarray (xr)
import os
import xarray as xr
import xgcm
import dask
import xroms
from dask.diagnostics import ProgressBar
import numpy as np
import gc

filename = '/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc' 
out_path = '/Users/piero/arian/data1/IT_outs/pressure/'
os.makedirs(out_path, exist_ok=True)


g = 9.81
# Chunking: Time=1 is the safest way to prevent RAM spikes on 1TB files.
# It forces Dask to process one time-slice across all depths/space at a time.
chunks = {'ocean_time': 1}
xis = slice(40, None)
# 1. Open Dataset Lazily
ds1 = xr.open_dataset(filename, chunks=chunks)
ds,xgrid = xroms.roms_dataset(ds1)


def compute_p_bc(ds_slice):
	"""
	Computes baroclinic pressure lazily.
	"""
	# dz is calculated per time step based on the free surface
	dz = ds_slice.dz
	
	# Hydrostatic integration: Surface to Bottom
	# We reindex to start from the surface (s_rho top index), cumsum, then reindex back.
	p_total = (g * ds_slice.rho * dz).reindex(s_rho=ds_slice.s_rho[::-1]).cumsum(dim='s_rho').reindex(s_rho=ds_slice.s_rho)
	
	# Remove depth mean to get baroclinic component
	p_bc = p_total - p_total.mean(dim='s_rho')
	p_bc.name = 'p_bc'
	p_bc.attrs['units'] = 'Pa'
	p_bc.attrs['long_name'] = 'baroclinic hydrostatic pressure'
	
	return p_bc

# 2. Processing in Spatial Slices 
# (Even with Dask, slicing prevents "Too many open files" errors and large graph overhead)
ETA_STEP = 40      
eta_steps = np.arange(0, 1360, ETA_STEP)

for i in range(len(eta_steps)):
	eta_slice = slice(eta_steps[i], eta_steps[i+1])
	print(f"--- Processing p_bc Slice {i} (Rows {eta_steps[i]} to {eta_steps[i+1]}) ---")
	
	ds_sub = ds.isel(eta_rho=eta_slice,xi_rho=xis)
	p_bc_slice = compute_p_bc(ds_sub)
	# Drop z_rho if it exists in the dataset/dataarray to avoid saving it
	
	p_bc_slice = p_bc_slice.drop_vars('z_rho', errors='ignore')
	
	# Save each slice
	save_file = os.path.join(out_path, f'p_bc_slice_{i}.nc')
	
	# This triggers the computation and streams it to disk
	with ProgressBar():
		p_bc_slice.to_netcdf(save_file)
	
	# Cleanup
	del p_bc_slice
	gc.collect()
ds1.close()




