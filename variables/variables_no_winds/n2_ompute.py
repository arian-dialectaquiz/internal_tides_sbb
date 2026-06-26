
######

import os
import sys
import gc
import numpy as np
import xarray as xr
import xroms
from scipy import signal


filename = '/Users/piero/arian/data1/nc_outs/avg_internal_tides_paper.nc' 
out_path = '/Users/piero/arian/data1/NO_WINDS_IT/N2/'
os.makedirs(out_path, exist_ok=True)

ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})
	
# Standardized Micro-Slicing Configuration
ETA_STEP = 40      
eta_steps = np.arange(0, 1360, ETA_STEP) 
xis = slice(40, None)
# Open dataset lazily with chunked time coordinates
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 1, 's_rho': -1, 'eta_rho': 'auto', 'xi_rho': 'auto'})
ds, xgrid = xroms.roms_dataset(ds1)
# =============================================================================
# 2. MAIN SPLICED PROCESSING LOOP
# =============================================================================
for i in range(len(eta_steps) - 1):
	print(f"\n=== Processing N2 Slice {i}/{len(eta_steps)-1} (Rows {eta_steps[i]} to {eta_steps[i+1]}) ===")
	eta_slice = slice(eta_steps[i], eta_steps[i+1])
	
	# 1. Step out to a localized spatial subset lazily (All times processed together)
	#ds_sub = ds1.isel(eta_rho=eta_slice, xi_rho=xis)
	
	
	print("   > Calculating EOS Density and buoyancy frequency (N2)...")
	# 2. Vectorized equation of state calculation across the entire timeline
	rho_ini = ds.rho + 1000
	
	# 3. Calculate stratification frequency on vertical w-faces
	N2_ini = xroms.N2(rho_ini, xgrid)            
	
	# 4. Interpolate N2 down to s_rho cell centers to match your velocity/flux grids
	N2_s_lazy = xroms.to_s_rho(N2_ini, xgrid)
	
	print("   > Computing and pushing to RAM...")
	# 5. Trigger computation for this 10-row slice across all times
	N2_computed = N2_s_lazy.isel(eta_rho=eta_slice, xi_rho=xis).compute()
	
	# 6. Wrap into an isolated dataset structure
	ds_out = N2_computed.to_dataset(name='N2')
	
	# Keep attributes clean and useful
	ds_out.attrs['description'] = f'Buoyancy Frequency N2 for eta slice {eta_steps[i]}-{eta_steps[i+1]}'
	
	# 7. Configure light compression on-the-fly (eliminating the old re-saving step)
	encoding = {'N2': {"zlib": True, "complevel": 1}}
	eta_filename = os.path.join(out_path, f'N2_slice_{i}.nc')
	
	print(f"   > Saving compressed NetCDF: {eta_filename}")
	ds_out.to_netcdf(eta_filename, encoding=encoding, engine='netcdf4')
	
	# 8. Complete memory clear-out for next block iteration
	del rho_ini, N2_ini, N2_s_lazy, N2_computed, ds_out
	gc.collect()

ds1.close()
print("\nAll N2 slices processed, compressed, and saved successfully with zero temporary files!")



