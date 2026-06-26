####
import xroms
import dask
import pandas as pd
import gc
from scipy.signal import butter, filtfilt
from pathlib import Path
import xarray as xr
import numpy as np

# Configuration
out_path = Path('/Users/piero/arian/data1/NO_WINDS_IT/temperature/')
out_path.mkdir(parents=True, exist_ok=True)

#filename = '/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/dia_paper_2_1km.nc'    

filename = f'/Users/piero/arian/data1/nc_outs/avg_internal_tides_paper.nc'
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 'auto'})
# Define eta and xi slices

# Define eta and xi slices
eta = np.arange(0, 1280, 1)
etas = np.reshape(eta, (32, 40))
xis = slice(40, None)
ds, xgrid = xroms.roms_dataset(ds1)

# Ensure the vertical coordinate transformation is done globally or lazily before slicing
print("Transforming AKt to s_rho grid...")
akt_s_rho = xroms.to_s_rho(ds.AKt, xgrid)

# Loop over all 32 grid/slice configurations in etas
for i in range(etas.shape[0]):
	# Extract the block of 40 eta targets for this iteration
	eta_targets = etas[i, :]
	print(f"--- Processing Slice Group {i+1}/{etas.shape[0]} ---")
	print(f"Eta range: {eta_targets[0]} to {eta_targets[-1]}")
	
	# Slice directly on both xi_rho and eta_rho at once using the group indices
	# We use .compute() at the end to trigger Dask execution efficiently
	ak = akt_s_rho.isel(
		xi_rho=xis, 
		eta_rho=eta_targets
	).compute()
	
	# Save the current slice block out to NetCDF
	output_file = out_path / f'AKt_slice_{i}.nc'
	ak.to_dataset(name='AKt').to_netcdf(output_file)
	print(f"Saved {output_file}")
	
	# Explicit garbage collection to keep memory profile low
	del ak
	gc.collect()

print("All slices processed successfully.")



