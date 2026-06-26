#########
import os
import sys
import gc
import numpy as np
import xarray as xr
import xroms
from scipy import signal


filename = '/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc' 
n2_path = '/Users/piero/arian/data1/IT_outs/N2/'
out_path = '/Users/piero/arian/data1/IT_outs/alpha/'

os.makedirs(out_path, exist_ok=True)


# =============================================================================
# CONSTANTS & PARAMETERS
# =============================================================================
g = 9.81
T_M2 = 12.4206012  # hours
omega_m2 = (2 * np.pi) / (T_M2 * 3600.0)  # Convert Rad/hour to Rad/second

# Standardized Micro-Slicing Configuration
ETA_STEP = 10      
eta_steps = np.arange(0, 1360, ETA_STEP) 
xis = slice(40, None)

# Open base dataset lazily for grid metrics and Coriolis
ds1 = xr.open_dataset(filename, chunks={'ocean_time': 1, 's_rho': -1, 'eta_rho': 'auto', 'xi_rho': 'auto'})
ds, xgrid = xroms.roms_dataset(ds1)

# Extract Coriolis parameter (f) and Bathymetry (h) globally
f_cor = ds.f
h_bathy = ds.h

# =============================================================================
# MAIN PROCESSING LOOP (Loading Pre-Computed N2 Slices)
# =============================================================================
for i in range(len(eta_steps) - 1):
	print(f"\n=== Processing Alpha Slice {i}/{len(eta_steps)-1} (Rows {eta_steps[i]} to {eta_steps[i+1]}) ===")
	eta_slice = slice(eta_steps[i], eta_steps[i+1])
	
	# 1. Load the pre-computed N2 slice file lazily
	n2_file = os.path.join(n2_path, f'N2_slice_{i}.nc')
	if not os.path.exists(n2_file):
		print(f"   > Warning: File {n2_file} not found. Skipping...")
		continue
	
	ds_n2 = xr.open_dataset(n2_file, chunks={'ocean_time': 1})
	
	# Subset local physical horizontal metrics needed for spatial derivatives
	ds_sub = ds.isel(eta_rho=eta_slice, xi_rho=xis)
	pm = ds_sub.pm
	pn = ds_sub.pn
	f_sub = f_cor.isel(eta_rho=eta_slice, xi_rho=xis)
	h_sub = h_bathy.isel(eta_rho=eta_slice, xi_rho=xis)
	
	# 2. Calculate Topographic Slope: grad(h) using Xarray differentiation tools
	dh_dxi = h_sub.differentiate('xi_rho') * pm
	dh_deta = h_sub.differentiate('eta_rho') * pn
	topo_slope = np.sqrt(dh_dxi**2 + dh_deta**2).compute()
	
	print("   > Extracting bottom N2 layer from saved slice...")
	# 3. Extract N2 right above the ocean floor (s_rho = 1) from the loaded slice data
	N2_bottom = ds_n2.N2.isel(s_rho=1).compute()
	print("   > working on negative N2...")
	# Clip sub-zero stratification values safely to zero
	N2_bottom_clipped = xr.where(N2_bottom > 0, N2_bottom, 0.0)
	
	# 4. Compute Wave Ray-Path Slope (s)
	# Equation 18: s = sqrt((omega^2 - f^2) / (N^2 - omega^2))
	num = omega_m2**2 - f_sub**2
	denom = N2_bottom_clipped - omega_m2**2
	
	# Protect against supercritical or unstratified zones where denom <= 0 or num < 0
	#ray_slope_sq = xr.where((denom > 0) & (num >= 0), num / denom, np.nan)
	ray_slope = np.sqrt(num/denom).compute()
	print("   > computing ratio...")
	# 5. Compute Alpha Ratio (alpha = topo_slope / ray_slope)
	alpha = topo_slope / ray_slope

	alpha.name = 'alpha'
	alpha.attrs['long_name'] = 'internal tide steepness parameter'
	alpha.attrs['description'] = 'sub-critical (<0.8), critical (0.8-1.5), super-critical (>1.5)'
	
	# 6. Wrap into an isolated dataset structure
	ds_out = xr.Dataset({
		'alpha': alpha,
		
	})
	
	ds_out.attrs['description'] = f'Topography Steepness Alpha for eta slice {eta_steps[i]}-{eta_steps[i+1]}'
	
	# Configure compression settings
	encoding = {
		'alpha': {"zlib": True, "complevel": 1}		
	}
	alpha_filename = os.path.join(out_path, f'alpha_slice_{i}.nc')
	
	print(f"   > Saving compressed NetCDF: {alpha_filename}")
	ds_out.to_netcdf(alpha_filename, encoding=encoding, engine='netcdf4')
	
	# 7. Complete memory clear-out for next block iteration
	ds_n2.close()
	del ds_n2, N2_bottom, N2_bottom_clipped, ray_slope, alpha, ds_out
	gc.collect()

ds1.close()
print("\nAll alpha calculations completed successfully!")


