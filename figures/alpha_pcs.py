
import xarray as xr
import numpy as np
import dask
import re
import glob
import cartopy
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, NullFormatter, FixedLocator
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmocean as cmo
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import pandas as pd
import tropycal.tracks as tracks
import glob

def natural_keys(text):
	"""Sorts strings numerically rather than alphabetically."""
	return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

#####------> Dates 
before = [("2004-03-16", "2004-03-23")]

during = [("2004-03-24", "2004-03-30")]

after = [("2004-03-31", "2004-04-06")]


before_start, before_end = "2004-03-16", "2004-03-23"
during_start, during_end = "2004-03-24", "2004-03-30"
after_start, after_end   = "2004-03-31", "2004-04-06"

south_top = 420
central_top = 900



A = [310, 1231]
B = [252, 767]
C = [444, 35]

tec_path = "/Users/piero/arian/data1/IT_outs/tec/"

tec_f  = sorted(glob.glob(tec_path + 'tec_slice_*.nc'), key=natural_keys)
tec = xr.open_mfdataset(tec_f, combine='nested', concat_dim='eta_rho', parallel=True)


####-----> Importing Alpha files  with winds<----#####

alpha_path = "/Users/piero/arian/data1/IT_outs/alpha/"

alpha_f  = sorted(glob.glob(alpha_path + 'alpha_slice_*.nc'), key=natural_keys)
alpha = xr.open_mfdataset(alpha_f, combine='nested', concat_dim='eta_rho', parallel=True).isel(eta_rho=tec.eta_rho.values).rolling(ocean_time=25, center=True, min_periods=1).mean()
aa = alpha.sel(ocean_time=tec.ocean_time, method='nearest').alpha

lat_rho = alpha.lat_rho.compute()
lon_rho = alpha.lon_rho.compute()

a_before_map = aa.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time').compute()
a_during_map = aa.sel(ocean_time=slice(during_start, during_end)).mean(dim='ocean_time').compute()
a_after_map = aa.sel(ocean_time=slice(after_start, after_end)).mean(dim='ocean_time').compute()


####-----> Importing Alpha files  no winds<----#####


import xroms
ds1 = xr.open_dataset('/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc', chunks={'ocean_time': 1, 's_rho': -1, 'eta_rho': 'auto', 'xi_rho': 'auto'})
ds, xgrid = xroms.roms_dataset(ds1)

ds_fix = ds.isel(eta_rho=slice(0, 1280), xi_rho=slice(40, None))

h = ds_fix.h.compute()
lat_h = h.lat_rho
lon_h = h.lon_rho

alpha_path_no = "/Users/piero/arian/data1/NO_WINDS_IT/alpha/"

alpha_n  = sorted(glob.glob(alpha_path_no + 'alpha_slice_*.nc'), key=natural_keys)
alpha_no = xr.open_mfdataset(alpha_n, combine='nested', concat_dim='eta_rho', parallel=True).isel(eta_rho=tec.eta_rho.values).rolling(ocean_time=8, center=True, min_periods=1).mean()

aa_no = alpha_no.sel(ocean_time=tec.ocean_time, method='nearest').sel(eta_rho=slice(0,1279)).alpha

a_no_before_map = aa_no.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time').compute()
a_no_during_map = aa_no.sel(ocean_time=slice(during_start, during_end)).mean(dim='ocean_time').compute()
a_no_after_map = aa_no.sel(ocean_time=slice(after_start, after_end)).mean(dim='ocean_time').compute()

DA_bf = a_before_map - a_no_before_map
DA_dr = a_during_map - a_no_during_map
DA_af = a_after_map - a_no_after_map

###_ Mean values
mask_shelf = h < 250
mask_slope = h>=250

DA_bf_shelf = float(DA_bf.where(mask_shelf).mean())
DA_dr_shelf = float(DA_dr.where(mask_shelf).mean())
DA_af_shelf = float(DA_af.where(mask_shelf).mean())

DA_bf_deep = float(DA_bf.where(mask_slope).mean())
DA_dr_deep = float(DA_dr.where(mask_slope).mean())
DA_af_deep = float(DA_af.where(mask_slope).mean())



####################################################################
######------> PCA <---#############################################

# 2. Define regional boundaries using .isel() along the eta_rho axis
regions = {
	'South': aa.isel(eta_rho=slice(0, 420)),
	'Central': aa.isel(eta_rho=slice(420, 900)),
	'North': aa.isel(eta_rho=slice(900, None))
}

# --- NEW: Dictionary to store results for downstream plotting ---
eof_results = {}

# 3. Process each region individually
for name, da in regions.items():
	print(f"--- Processing {name} Region ---")
	
	# Transpose to put time first (Standard requirement for EOF/PCA analysis)
	da_ordered = da.transpose('ocean_time', 'eta_rho', 'xi_rho')
	
	# Flatten the 2D spatial dimensions into a single 1D 'spatial' dimension
	da_stacked = da_ordered.stack(spatial=('eta_rho', 'xi_rho'))
	
	# Convert Dask array to NumPy array in memory
	data_matrix = da_stacked.values  # Shape: (ocean_time, spatial)
	
	# Handle NaNs (Land Masks): Identify grid cells that are valid across all times
	valid_spatial_indices = ~np.isnan(data_matrix).any(axis=0)
	clean_matrix = data_matrix[:, valid_spatial_indices]
	
	if clean_matrix.shape[1] == 0:
		print(f"Skipping {name} region: Contains no valid data points.")
		continue

	# Remove the temporal mean to get anomalies
	temporal_mean = np.mean(clean_matrix, axis=0)
	anomaly_matrix = clean_matrix - temporal_mean
	
	# 4. Fit EOF/PCA using Pure NumPy SVD
	U, S, Vt = np.linalg.svd(anomaly_matrix, full_matrices=False)
	
	# Calculate Variance Explained by each mode
	variance_explained = (S**2) / np.sum(S**2) * 100
	print(f"Mode 1 Variance Explained: {variance_explained[0]:.2f}%")
	
	# Extract Mode 1 components
	pc1 = U[:, 0] * S[0]   # PC1 (Temporal evolution)
	eof1 = Vt[0, :]        # EOF1 (Spatial footprint)
	
	# 5. Reconstruct the Spatial Map (inserting NaNs back into land mask areas)
	eof1_spatial_flat = np.full(data_matrix.shape[1], np.nan)
	eof1_spatial_flat[valid_spatial_indices] = eof1
	
	# Create an Xarray DataArray to facilitate unstacking back to 2D
	eof1_spatial_da = xr.DataArray(
		eof1_spatial_flat, 
		coords=[da_stacked.spatial], 
		dims=['spatial']
	)
	eof1_matrix = eof1_spatial_da.unstack('spatial')
	
	# Slice the corresponding lat/lon coordinates for mapping
	lon_reg = alpha.lon_rho.isel(eta_rho=da.eta_rho)
	lat_reg = alpha.lat_rho.isel(eta_rho=da.eta_rho)
	
	# --- NEW: Package all variables for this region into the dictionary ---
	eof_results[name] = {
		'eof1_matrix': eof1_matrix,
		'pc1': pc1,
		'ocean_time': da_ordered.ocean_time.values,
		'lon': lon_reg.values,
		'lat': lat_reg.values,
		'variance_explained': variance_explained[0]
	}

print("\nAll regions processed! Variables are safely stored in 'eof_results'.")

	
north_data = eof_results['North']
central_data = eof_results['Central']
south_data = eof_results['South']


# --- slope calcualtion
time_windows = {
	'Before': ("2004-03-16", "2004-03-23"),
	'During': ("2004-03-24", "2004-03-30"),
	'After':  ("2004-03-31", "2004-04-06")
}

regional_datasets = {
	'North': north_data,
	'Central': central_data,
	'South': south_data
}

# Dictionary to store the calculated slopes: {Region: {Period: Slope}}
calculated_slopes = {region: {} for region in regional_datasets.keys()}

# --- 2. Calculate slopes using linear regression ---
for r_name, r_data in regional_datasets.items():
	# Convert arrays to pandas Series to make time-string slicing simple
	pc_series = pd.Series(r_data['pc1'], index=pd.to_datetime(r_data['ocean_time']))
	
	for period_name, (start_dt, end_dt) in time_windows.items():
		# Slice data for the specific time window
		window_data = pc_series.loc[start_dt:end_dt]
		
		if len(window_data) > 1:
			# Convert timestamps to numeric "windows elapsed since start of window"
			time_days = (window_data.index - window_data.index[0]).total_seconds() / (25 * 3600)
			y_values = window_data.values
			
			# Linear fit (y = mx + b). The first output [0] is the slope (m)
			slope, intercept = np.polyfit(time_days, y_values, 1)
			
			# Store the slope (Amplitude change per day)
			calculated_slopes[r_name][period_name] = slope
		else:
			calculated_slopes[r_name][period_name] = np.nan

# Print values out to verify
for r_name, periods in calculated_slopes.items():
	print(f"Slopes for {r_name} (Units: Amplitude unit / Window):")
	for p_name, slope_val in periods.items():
		print(f"  {p_name}: {slope_val:+.3f}")



####################################################################
######------> alpha + hurricane track <---##########################
from scipy.stats import binned_statistic
ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc_neumann',catarina=True)

storm = ibtracs.get_storm(('catarina',2004))
# =====================================================================
# 1. CALCULATE THE ANOMALY FIELD 
# =====================================================================
print("Calculating pre-storm baseline (mean condition)...")
# Average the spatial grid over the calm 'before' timeline to get the background state
alpha_baseline = aa.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time')

print("Generating the spatial anomaly fields...")
# Subtract the static baseline from every single time step
aa_anomaly = aa - alpha_baseline

# =====================================================================
# 2. TRACK ALIGNMENT (WITHOUT EXTRAPOLATION ARTIFACTS)
# =====================================================================
storm_df = pd.DataFrame({
	'lat': storm.lat,
	'lon': storm.lon
}, index=pd.to_datetime(storm.time))

# Strip out duplicate times if present
storm_df = storm_df[~storm_df.index.duplicated(keep='first')]

# Map to model's ocean_time axis. 
# By avoiding out-of-bounds filling, times before/after the track will remain NaN.
target_times = pd.to_datetime(aa.ocean_time.values)
combined_indices = storm_df.index.union(target_times)
storm_interpolated = storm_df.reindex(combined_indices).interpolate(method='time').loc[target_times]

track_lats = storm_interpolated['lat'].values
track_lons = storm_interpolated['lon'].values


# =====================================================================
# 3. HA VERSINE DISTANCE METRIC
# =====================================================================
def calculate_radial_distance(grid_lon, grid_lat, center_lon, center_lat):
	R_earth = 6371.0  # Earth radius in kilometers
	
	lon1, lat1 = np.radians(grid_lon), np.radians(grid_lat)
	lon2, lat2 = np.radians(center_lon), np.radians(center_lat)
	
	dlon = lon1 - lon2
	dlat = lat1 - lat2
	
	a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
	c = 2 * np.arcsin(np.sqrt(a))
	
	return R_earth * c


# =====================================================================
# 4. RADIAL BINNING SETUP
# =====================================================================
lon_grid = alpha.lon_rho.values
lat_grid = alpha.lat_rho.values

max_radius_km = 500
bin_width_km = 3
bin_edges = np.arange(0, max_radius_km + bin_width_km, bin_width_km)
bin_centers = bin_edges[:-1] + (bin_width_km / 2)

num_times = len(aa.ocean_time)
heatmap_anomaly_matrix = np.full((len(bin_centers), num_times), np.nan)


# =====================================================================
# 5. AZIMUTHAL AVERAGING LOOP (ANOMALIES ON TRACK)
# =====================================================================
print("Processing storm-relative coordinate transformation on anomalies...")

for t_idx in range(num_times):
	c_lon = track_lons[t_idx]
	c_lat = track_lats[t_idx]
	
	# Safely skip pre/post storm environments where track is unavailable
	if np.isnan(c_lon) or np.isnan(c_lat):
		continue
		
	# Get distances from the moving eye to the spatial grid at time t
	distance_km_2d = calculate_radial_distance(lon_grid, lat_grid, c_lon, c_lat)
	
	# CRITICAL: Pull the anomaly slice rather than the raw data slice
	anomaly_slice_2d = aa_anomaly.isel(ocean_time=t_idx).values
	
	flat_dist = distance_km_2d.flatten()
	flat_anomaly = anomaly_slice_2d.flatten()
	valid_mask = ~np.isnan(flat_anomaly)
	
	# Compute azimuthal average of deviations
	bin_means, _, _ = binned_statistic(
		flat_dist[valid_mask], 
		flat_anomaly[valid_mask], 
		statistic='mean', 
		bins=bin_edges
	)
	
	heatmap_anomaly_matrix[:, t_idx] = bin_means


# =====================================================================
# 6. STRUCTURE AS AN XARRAY DATAARRAY FOR DOWNSTREAM USE
# =====================================================================
anomaly_radius_time_heatmap = xr.DataArray(
	heatmap_anomaly_matrix,
	dims=['radius', 'ocean_time'],
	coords={
		'radius': bin_centers,
		'ocean_time': aa.ocean_time
	},
	attrs={
		'long_name': 'Azimuthally Averaged Alpha Anomaly',
		'units': aa.attrs.get('units', 'meter-1'),
		'description': 'Net physical response of wave path slope to hurricane forcing (baseline removed)'
	}
)

def add_inset_colorbar(ax, cmap, norm, label, loc='lower right', extend='both'):
	"""Creates a stylized colorbar inside the given axes."""
	# Create an inset axes instance
	ax_cb = inset_axes(ax, width="35%", height="5%", loc=loc, borderpad=1.5)
	ax_cb.set_facecolor('lightgray')
	
	# Draw the colorbar
	cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, 
								   extend=extend, orientation='horizontal')
	cb.set_label(label, size=9)
	ax_cb.xaxis.set_ticks_position('bottom')
	ax_cb.tick_params(axis='x', labelsize=8, rotation=15)
	return cb

def format_time_axis(ax):
	"""Standardizes time axis formatting."""
	ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
	ax.tick_params(axis='x', labelsize=10, rotation=25)
	ax.set_xlabel(None)


##########------>>>>> Plotting <<<-----#########



fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(nrows=3, ncols=6, height_ratios=[1,1,1])
gs.update(left=0.07, right=0.98, wspace=0.3, hspace=0.3, top=0.98, bottom=0.05)


####------> EOF map
unit_map = 'EOF Weight'
cmap_map = plt.cm.bwr

vmin = -0.04
vmax = 0.04
norm_map = mpl.colors.Normalize(vmin=vmin, vmax=vmax)


a_bf_m = plt.subplot(gs[0, 0:3],projection=ccrs.PlateCarree())
a_bf_m.set_aspect('auto')

a_bf_m.text(0.88, 0.95, '(a)', transform=a_bf_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
a_bf_m.set_ylim(bottom=-32.6, top=-22)
a_bf_m.set_xlim(left=-52, right=-39.2)

a_bf_m.contourf(south_data['lon'], south_data['lat'], -1*south_data['eof1_matrix'], levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())
a_bf_m.contourf(central_data['lon'], central_data['lat'], -1*central_data['eof1_matrix'], levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())
a_bf_m.contourf(north_data['lon'], north_data['lat'], -1*north_data['eof1_matrix'], levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

ax_map_cb = inset_axes(a_bf_m, 
					   width="45%", 
					   height="3%", 
					   loc='upper left',
					   bbox_to_anchor=(0.08, -0.03, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=a_bf_m.transAxes,
					   borderpad=0)

ax_map_cb.set_facecolor('lightgray')

cb_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_map, norm=norm_map, extend='both', orientation='horizontal')
cb_map.set_label(unit_map, size=10, loc='left')
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)

a_bf_m.coastlines()
a_bf_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
a_bf_m.patch.set_edgecolor('black')
gl = a_bf_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 10, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 10, 'color': 'dimgrey'}


####------> PC1 timeline
pcx = plt.subplot(gs[0, 3:6])

pcx.text(0.05, 0.97, '(b)', transform=pcx.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
pcx.set_ylim(bottom=-30, top=30)
pcx.set_title('PC1 Amplitude',fontsize = 10, loc ='left')

pcx.plot(north_data['ocean_time'], -1*north_data['pc1'], label='North', linestyle=':', color='green')
pcx.plot(central_data['ocean_time'], -1*central_data['pc1'], label='Central', linestyle='--', color='orange')
pcx.plot(south_data['ocean_time'], -1*south_data['pc1'], label='South', linestyle='-', color='cyan')
for i, (start, end) in enumerate(during):
	pcx.axvspan(pd.to_datetime(start),
				pd.to_datetime(end),
				color='red',
				alpha=0.2,
				label='Hurricane' if i == 0 else None)
pcx.axhline(0,ls = ':', lw = 2, color = 'k')
# Create an organized textbox on the plot to demonstrate the sharper variation
text_str = "Linear Slopes ($\Delta$ Amp / Window):\n"
for r_name, col in zip(['North', 'Central', 'South'], ['green', 'orange', 'darkturquoise']):
	s_bef = -1*calculated_slopes[r_name]['Before']
	s_dur = -1*calculated_slopes[r_name]['During']
	s_aft = -1*calculated_slopes[r_name]['After']
	text_str += f"{r_name}:\n  Bef: {s_bef:+.1f} | Dur: {s_dur:+.1f} | Aft: {s_aft:+.1f}\n"

# Placing text box in the upper right quadrant of the axis 
pcx.text(0.25, 0.35, text_str, transform=pcx.transAxes, fontsize=7, 
		 fontfamily='monospace', verticalalignment='top', horizontalalignment='center',
		 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

format_time_axis(pcx)
pcx.xaxis.set_major_locator(mdates.DayLocator(interval=4))
pcx.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
pcx.legend(loc=1, fontsize = 10)
#pcx.set_ylabel('PC1 Amplitude',fontsize = 10)

###############################
######---> Radial decomposition <---#######
vmin, vmax = -0.08, 0.08
cmap_r = plt.cm.bwr
norm_r = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
rx = plt.subplot(gs[1, 0:])
rx.text(0.05, 0.95, '(c)', transform=rx.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

mesh = rx.pcolormesh(
	anomaly_radius_time_heatmap.ocean_time.values,
	anomaly_radius_time_heatmap.radius.values,
	-1*anomaly_radius_time_heatmap.values,
	cmap=cmap_r,  # Red = Positive Anomaly, Blue = Negative Anomaly
	vmin=vmin,
	vmax=vmax,
	shading='auto'
)
start_date = anomaly_radius_time_heatmap.ocean_time.values[0]
end_date = pd.to_datetime("2004-03-31")

#cbar = plt.colorbar(mesh, ax=rx, pad=0.02, extend='both')
#cbar.set_label(r'$\Delta\alpha$', fontsize=10, fontweight='demibold')

#--> cbar
rx_cb = inset_axes(rx, width="30%", 
					   height="3%", 
					   loc='lower left',
					   bbox_to_anchor=(0.01, 0.2, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=rx.transAxes,
					   borderpad=0)

rx_cb.set_facecolor('lightgray')

rx_map = mpl.colorbar.ColorbarBase(rx_cb, cmap=cmap_r, 	norm = norm_r ,extend='both', orientation='horizontal')
rx_map.set_label(r'$\Delta\alpha$', fontsize=10, fontweight='demibold')
rx_cb.xaxis.set_ticks_position('bottom')
rx_cb.tick_params(axis='x', labelsize='small', rotation=15)


# --- 6. Final Formatting and Styling ---
rx.set_xlim(start_date, end_date)
rx.set_ylabel('Distance from Moving Storm Center (km)', fontsize=10)
rx.set_xlabel(None)

rx.set_ylim(0, 400)  # Limit radius view up to 400 km
rx.grid(True, linestyle=':', alpha=0.5)

# Beautifully format the time axis
rx.xaxis.set_major_locator(mdates.DayLocator(interval=2))
rx.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.setp(rx.get_xticklabels(), rotation=30, ha='right')

###############################
######---> Delta alpha <---#######


## isobaths
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]

levels1 = np.asarray(levels_1)
levels2 = np.asarray(levels_2)
levels3 = np.asarray(levels_3)
levels4 = np.asarray(levels_4)

unit_map = r'$\Delta\alpha$'

cmap_map = plt.cm.bwr

vmin = -0.3
vmax = 0.3
norm_map = mpl.colors.Normalize(vmin=vmin, vmax=vmax)


####---> Before Map
a_bf_m = plt.subplot(gs[2, 0:2],projection=ccrs.PlateCarree())
a_bf_m.set_aspect('auto')
a_bf_m.text(0.5, 1.05, 'T1',
			transform=a_bf_m.transAxes,
			rotation=0,
			fontsize=10,
			fontweight='bold',
			va='center',
			ha='center') 

a_bf_m.text(0.88, 0.9, '(d)', transform=a_bf_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
a_bf_m.set_ylim(bottom=-33, top=-20)
a_bf_m.set_xlim(left=-52, right=-39.2)

ax_map_cb = inset_axes(a_bf_m, 
					   width="60%", 
					   height="3%", 
					   loc='upper left',
					   bbox_to_anchor=(0.08, -0.03, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=a_bf_m.transAxes,
					   borderpad=0)

ax_map_cb.set_facecolor('lightgray')

cb_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_map, norm=norm_map, extend='both', orientation='horizontal')
cb_map.set_label(unit_map, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


a_bf_m.contourf(lon_rho, lat_rho, DA_bf, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

a_bf_m.coastlines()
a_bf_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=5)
a_bf_m.patch.set_edgecolor('black')
gl = a_bf_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 10, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 10, 'color': 'dimgrey'}

#-->isobaths
c1 = a_bf_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = a_bf_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = a_bf_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = a_bf_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

stats_text = (
	r"Shelf mean: $%.2f \times 10^{-3}$" % (DA_bf_shelf * 1e3) + "\n" +
	r"Deep mean: $%.2f \times 10^{-3}$" % (DA_bf_deep * 1e3)
)
a_bf_m.text(0.95, 0.05, stats_text, 
			transform=a_bf_m.transAxes, 
			fontsize=8, 
			fontweight='bold', 
			va='bottom', 
			ha='right',
			zorder=12,
			bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.85))


####---> During Map
a_dr_m = plt.subplot(gs[2, 2:4],projection=ccrs.PlateCarree())
a_dr_m.set_aspect('auto')

a_dr_m.text(0.5, 1.05, 'T2',
			transform=a_dr_m.transAxes,
			rotation=0,
			fontsize=10,
			fontweight='bold',
			va='center',
			ha='center')

a_dr_m.text(0.88, 0.9, '(e)', transform=a_dr_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
a_dr_m.set_ylim(bottom=-33, top=-20)
a_dr_m.set_xlim(left=-52, right=-39.2)

ax_map_cb = inset_axes(a_dr_m, 
					   width="60%", 
					   height="3%", 
					   loc='upper left',
					   bbox_to_anchor=(0.08, -0.03, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=a_dr_m.transAxes,
					   borderpad=0)

ax_map_cb.set_facecolor('lightgray')

cb_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_map, norm=norm_map, extend='both', orientation='horizontal')
cb_map.set_label(unit_map, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


a_dr_m.contourf(lon_rho, lat_rho, DA_dr, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

a_dr_m.coastlines()
a_dr_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=5)
a_dr_m.patch.set_edgecolor('black')
gl = a_dr_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 10, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 10, 'color': 'dimgrey'}

#-->isobaths
c1 = a_dr_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = a_dr_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = a_dr_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = a_dr_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

stats_text = (
	r"Shelf mean: $%.2f \times 10^{-3}$" % (DA_dr_shelf * 1e3) + "\n" +
	r"Deep mean: $%.2f \times 10^{-3}$" % (DA_dr_deep * 1e3)
)
a_dr_m.text(0.95, 0.05, stats_text, 
			transform=a_dr_m.transAxes, 
			fontsize=8, 
			fontweight='bold', 
			va='bottom', 
			ha='right',
			zorder=12,
			bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.85))



####---> After Map
a_af_m = plt.subplot(gs[2, 4:6],projection=ccrs.PlateCarree())
a_af_m.set_aspect('auto')
a_af_m.text(0.5, 1.05, 'T3',
			transform=a_af_m.transAxes,
			rotation=0,
			fontsize=10,
			fontweight='bold',
			va='center',
			ha='center')

a_af_m.text(0.88, 0.9, '(f)', transform=a_af_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
a_af_m.set_ylim(bottom=-33, top=-20)
a_af_m.set_xlim(left=-52, right=-39.2)

ax_map_cb = inset_axes(a_af_m, 
					   width="60%", 
					   height="3%", 
					   loc='upper left',
					   bbox_to_anchor=(0.08, -0.03, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=a_af_m.transAxes,
					   borderpad=0)

ax_map_cb.set_facecolor('lightgray')

cb_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_map, norm=norm_map, extend='both', orientation='horizontal')
cb_map.set_label(unit_map, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


a_af_m.contourf(lon_rho, lat_rho, DA_af, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

a_af_m.coastlines()
a_af_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=5)
a_af_m.patch.set_edgecolor('black')
gl = a_af_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = False
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 10, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 10, 'color': 'dimgrey'}

#-->isobaths
c1 = a_af_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = a_af_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = a_af_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = a_af_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
stats_text = (
	r"Shelf mean: $%.2f \times 10^{-3}$" % (DA_af_shelf * 1e3) + "\n" +
	r"Deep mean: $%.2f \times 10^{-3}$" % (DA_af_deep * 1e3)
)
a_af_m.text(0.95, 0.05, stats_text, 
			transform=a_af_m.transAxes, 
			fontsize=8, 
			fontweight='bold', 
			va='bottom', 
			ha='right',
			zorder=12,
			bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.85))



plt.savefig('fig_pca_track.png', dpi = 300)


