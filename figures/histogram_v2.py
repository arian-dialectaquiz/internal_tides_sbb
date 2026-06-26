




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
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import pandas as pd
import xroms

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

south_slice   = slice(0, 420)
central_slice = slice(421, 900)
north_slice   = slice(901, 1280)

A = [310, 1220]
B = [252, 767]
C = [444, 35]

A_xi, A_eta = A[0], A[1]
B_xi, B_eta = B[0], B[1]
C_xi, C_eta = C[0], C[1]

regions = {
	'South': slice(0, 420),
	'Central': slice(421, 900),
	'North': slice(901, 1280)
}

time_periods = {
	'T1': ("2004-03-16", "2004-03-23"),
	'T2': ("2004-03-24", "2004-03-30"),
	'T3': ("2004-03-31", "2004-04-06")
}

####-----> Importing TEC files <----#####
tec_path = "/Users/piero/arian/data1/IT_outs/tec/"

tec_f  = sorted(glob.glob(tec_path + 'tec_slice_*.nc'), key=natural_keys)
tec = xr.open_mfdataset(tec_f, combine='nested', concat_dim='eta_rho', parallel=True)
tt= -2*tec.topographic_energy_conversion
C = tt.where(tt > 0, 0)

####-----> Histograms (W) <----#####
filename = '/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc' 
xis = slice(40, None)

ds1 = xr.open_dataset(filename, chunks={'ocean_time': 1, 's_rho': -1, 'eta_rho': 'auto', 'xi_rho': 'auto'})
ds, xgrid = xroms.roms_dataset(ds1)

h_bathy = ds.h.isel(eta_rho=slice(0,1280), xi_rho=xis).compute()
pm =  ds.pm.isel(eta_rho=slice(0,1280), xi_rho=xis).compute()
pn =  ds.pn.isel(eta_rho=slice(0,1280), xi_rho=xis).compute()

# Define the depth bins for your histogram (e.g., every 50 meters from 0 to 2000m)
depth_bins = np.arange(0, 3000, 50)
bin_centers = 0.5 * (depth_bins[:-1] + depth_bins[1:])
dA = 1.0 / (pm * pn)

####-----> With winds <----#####
hist_winds = {reg: {per: {} for per in time_periods} for reg in regions}

for period_name, (start, end) in time_periods.items():
	print(f"Processing time period: {period_name} with winds...")
	C_period = C.sel(ocean_time=slice(start, end)).mean(dim='ocean_time').compute()
	
	for reg_name, reg_slice in regions.items():
		h_reg = h_bathy.isel(eta_rho=reg_slice).values.flatten()
		
		if isinstance(dA, xr.DataArray):
			dA_reg = dA.isel(eta_rho=reg_slice).values.flatten()/1e6
		else:
			dA_reg = dA/1e6
			
		C_reg = C_period.isel(eta_rho=reg_slice)
		
		for m in range(5):
			C_mode_flat = C_reg.sel(mode=m).values.flatten()
			weights = C_mode_flat * dA_reg
			hist_vals, _ = np.histogram(h_reg, bins=depth_bins, weights=weights)
			hist_winds[reg_name][period_name][f'Mode {m+1}'] = hist_vals
			
		C_total_flat = C_reg.sum(dim='mode').values.flatten()
		weights_total = C_total_flat * dA_reg
		hist_vals_total, _ = np.histogram(h_reg, bins=depth_bins, weights=weights_total)
		hist_winds[reg_name][period_name]['Total'] = hist_vals_total

####-----> Importing TEC NO files <----#####
tec_path = "/Users/piero/arian/data1/NO_WINDS_IT/tec/"

tec_f_no  = sorted(glob.glob(tec_path + 'tec_slice_*.nc'), key=natural_keys)
tec_no = xr.open_mfdataset(tec_f_no, combine='nested', concat_dim='eta_rho', parallel=True)
tt_no= -1*tec_no.topographic_energy_conversion
C_no = tt_no.where(tt_no > 0, 0).isel(eta_rho=slice(0,1280))

####-----> No winds <----#####
hist_no = {reg: {per: {} for per in time_periods} for reg in regions}

for period_name, (start, end) in time_periods.items():
	print(f"Processing time period: {period_name} without winds...")
	C_period = C_no.sel(ocean_time=slice(start, end)).mean(dim='ocean_time').compute()
	
	for reg_name, reg_slice in regions.items():
		h_reg = h_bathy.isel(eta_rho=reg_slice).values.flatten()
		
		if isinstance(dA, xr.DataArray):
			dA_reg = dA.isel(eta_rho=reg_slice).values.flatten()/1e6
		else:
			dA_reg = dA/1e6
			
		C_reg = C_period.isel(eta_rho=reg_slice)
		
		for m in range(5):
			C_mode_flat = C_reg.sel(mode=m).values.flatten()
			weights = C_mode_flat * dA_reg
			hist_vals, _ = np.histogram(h_reg, bins=depth_bins, weights=weights)
			hist_no[reg_name][period_name][f'Mode {m+1}'] = hist_vals
			
		C_total_flat = C_reg.sum(dim='mode').values.flatten()
		weights_total = C_total_flat * dA_reg
		hist_vals_total, _ = np.histogram(h_reg, bins=depth_bins, weights=weights_total)
		hist_no[reg_name][period_name]['Total'] = hist_vals_total

# =====================================================================
# 4. PLOT THE RESULTS (Layered Bars, Percentiles & Cumulative curves)
# =====================================================================
fig = plt.figure(figsize=(16, 11))

outer_gs = gridspec.GridSpec(nrows=3, ncols=3)
outer_gs.update(left=0.08, right=0.92, wspace=0.15, hspace=0.15, top=0.92, bottom=0.07)

modes_to_plot = ['Mode 1', 'Mode 2', 'Mode 3']
bar_colors = ['grey', 'magenta', 'gold'] 
bin_width = depth_bins[1] - depth_bins[0]

plot_counter = 0  
for idx_p, period_name in enumerate(time_periods.keys()):
	for idx_r, reg_name in enumerate(regions.keys()):
		
		inner_gs = gridspec.GridSpecFromSubplotSpec(
			nrows=2, ncols=1, 
			subplot_spec=outer_gs[idx_p, idx_r], 
			hspace=0.08
		)
		
		ax_no = plt.subplot(inner_gs[0, 0])    
		ax_wind = plt.subplot(inner_gs[1, 0])  
		ax_no.sharex(ax_wind)
		
		# Create twin configurations for cumulative lines
		ax_no_twin = ax_no.twinx()
		ax_wind_twin = ax_wind.twinx()
		
		# Link twin Y axes limits
		ax_no_twin.set_ylim(0, 1.05)
		ax_wind_twin.set_ylim(0, 1.05)
		
		# Process configurations for both scenarios loop
		scenarios = [
			(ax_no, ax_no_twin, hist_no[reg_name][period_name], 'Without Winds'),
			(ax_wind, ax_wind_twin, hist_winds[reg_name][period_name], 'With Winds')
		]
		
		for ax, ax_twin, data_dict, label_text in scenarios:
			total_vals = data_dict['Total']
			
			# 1. Plot continuous total background (Light Grey)
			ax.bar(bin_centers, total_vals, width=bin_width, color='royalblue', edgecolor='none', alpha=0.9, zorder=1)
			
			# 2. Plot Overlaid bars from baseline 0 (Higher modes over lower modes)
			for i, mode in enumerate(modes_to_plot):
				mode_vals = data_dict[mode]
				# Setting bottom=0 and increasing zorder ensures sequential forward layering
				ax.bar(bin_centers, mode_vals, width=bin_width, bottom=0,
					   color=bar_colors[i], edgecolor='none', alpha=0.85, zorder=2 + i)
			
			# 3. Calculate and Plot Cumulative Fraction & Percentiles
			cum_total = np.cumsum(total_vals)
			max_cum = cum_total[-1] if cum_total[-1] > 0 else 1.0
			cum_frac_total = cum_total / max_cum
			
			# Prepend 0 to starting depth to smoothly ground lines at 0
			plot_depths = np.insert(bin_centers, 0, depth_bins[0])
			
			# Draw Total Cumulative Line (Dark grey/black curve)
			ax_twin.plot(plot_depths, np.insert(cum_frac_total, 0, 0), color='royalblue', linewidth=1.75, zorder=10)
			
			# Draw individual mode cumulative lines
			for i, mode in enumerate(modes_to_plot):
				cum_mode = np.cumsum(data_dict[mode])
				cum_frac_mode = cum_mode / max_cum
				ax_twin.plot(plot_depths, np.insert(cum_frac_mode, 0, 0), color=bar_colors[i], linewidth=1.2, zorder=11 + i)
			
			# 4. Compute percentile depths using linear interpolation
			if cum_total[-1] > 0:
				p25_depth = np.interp(0.25, cum_frac_total, bin_centers)
				p50_depth = np.interp(0.50, cum_frac_total, bin_centers)
				p90_depth = np.interp(0.90, cum_frac_total, bin_centers)
				
				# Plot dynamic percentile lines
				for p_depth, p_label in zip([p25_depth, p50_depth, p90_depth], ['25th %', '50th %', '90th %']):
					ax.axvline(p_depth, color='black', linestyle='--', linewidth=0.7, alpha=0.6, zorder=20)
					ax.text(p_depth, 0.95, p_label, transform=ax.get_xaxis_transform(),
							rotation=90, va='top', ha='right', fontsize=7, alpha=0.7, fontweight='bold', zorder=21)
			
			# Formatting within sub-grids
			ax.grid(True, linestyle=':', alpha=0.3, axis='y')
			ax.text(0.7, 0.92, label_text, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', ha='right')
			
			# Maintain clean inside panels by hiding inner right labels
			if idx_r < 2:
				ax_twin.tick_params(labelright=False)
				
		# Structural Axis Annotations
		ax_no.tick_params(labelbottom=False)
		panel_letter = f"({chr(97 + plot_counter)})"
		ax_no.text(0.02, 0.92, panel_letter, transform=ax_no.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')
		plot_counter += 1 
		
		if idx_p == 0:
			ax_no.set_title(f'{reg_name}', fontsize=12, fontweight='bold', pad=12)
			
		if idx_r == 0:
			fig.text(0.04, (ax_no.get_position().y0 + ax_wind.get_position().y1)/2, 
					 f'{period_name}\n [MW]', 
					 fontsize=11, fontweight='bold', ha='center', va='center', rotation=90)
					 
		if idx_r == 2:
			ax_no_twin.set_ylabel('Cumulative fraction', fontsize=9, fontweight='bold')
			ax_wind_twin.set_ylabel('Cumulative fraction', fontsize=9, fontweight='bold')
			
		if idx_p == 2:
			ax_wind.set_xlabel('Depth [m]', fontsize=11, fontweight='bold')
			
		ax_no.set_xlim(0, depth_bins.max())
		
		# --- Cleaned up dynamic Y-limits to match your scaling setup ---
		y_limits_lookup = {
    		'T1': {'South': 10, 'Central': 10, 'North': 15},
    		'T2': {'South': 10, 'Central': 10, 'North': 15},
    		'T3': {'South': 5,  'Central': 5,  'North': 10}}
		y_max = y_limits_lookup[period_name][reg_name]
		ax_no.set_ylim(0, y_max)
		ax_wind.set_ylim(0, y_max)

# Generate a global clean legend matching custom elements
legend_elements = [Line2D([0], [0], color='royalblue', lw=2, label='total')] + \
				  [Line2D([0], [0], color=bar_colors[i], lw=2, label=f'mode-{i+1}') for i in range(len(modes_to_plot))]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99),
		   ncol=5, fontsize=10, framealpha=0.9, edgecolor='black')

plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

plt.savefig('fig_hists_v2.png', dpi=300, bbox_inches='tight')



