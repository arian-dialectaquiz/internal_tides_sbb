

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
import xroms
from scipy import signal
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import NaturalEarthFeature
from matplotlib.patheffects import Stroke
import shapely.geometry as sgeom
import matplotlib.patches as mpatches


# Define the colors dictionary for the map features
COLORS = {
    'land': '#e0dacb',   # A nice, muted natural beige for the landmass
    'water': '#b9d0ea'   # A clean, crisp light blue for the oceans
}

coarse = xr.open_dataset('/Users/piero/arian/data1/nc_outs/deproas_spongebob_grd_cropped_3km.nc')
finer = xr.open_dataset('/Users/piero/arian/data1/nc_outs/paper_2_1km_closed_cropped_smooth_sponge.nc')

# Extract mask_rho
f_mask_rho = finer.mask_rho.compute()
c_mask_rho = coarse.mask_rho.compute()


# Get coordinates
lon_c, lat_c = coarse.lon_rho, coarse.lat_rho
lon_f, lat_f = finer.lon_rho, finer.lat_rho

# Extract mask and coordinates
c_mask_rho = coarse.mask_rho.values
f_mask_rho = finer.mask_rho.values
lon_c, lat_c = coarse.lon_rho.values, coarse.lat_rho.values
lon_f, lat_f = finer.lon_rho.values, finer.lat_rho.values

# Flatten arrays
lon_c_flat, lat_c_flat, c_mask_flat = lon_c.ravel(), lat_c.ravel(), c_mask_rho.ravel()
lon_f_flat, lat_f_flat, f_mask_flat = lon_f.ravel(), lat_f.ravel(), f_mask_rho.ravel()

# Select ocean and land points
coarse_ocean = c_mask_flat == 1
finer_ocean = f_mask_flat == 1
coarse_land = c_mask_flat == 0
finer_land = f_mask_flat == 0


avg_1km = xr.open_dataset('/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc',chunks='auto')

grid = finer

h = avg_1km.h.compute()
lat_rho = avg_1km.lat_rho
lon_rho = avg_1km.lon_rho



ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc_neumann',catarina=True)




###########################################################################################
######------> Wind stress stick plot <----###################################################################
us = avg_1km.sustr.isel(ocean_time=slice(65,None))
vs = avg_1km.svstr.isel(ocean_time=slice(65,None))


coords = {
	'A': [380, 1233],
	'B': [276, 896],
	'C': [333, 557],
	'D': [453, 323]
}


A = [380, 1233]
B = [276, 896]
C = [333, 557]
D = [453, 323]


#h[1233, 380] #1002
#
#h[896, 276] #934
#
#h[557, 333] #968
#
#h[323, 453] #941



def apply_rotation(u, v, angle_rad):
	"""
	Rotates vectors by angle_rad. 
	If angle is between XI-axis and East, this aligns vectors to the model grid.
	"""
	u_rotated = u * np.cos(angle_rad) + v * np.sin(angle_rad)
	v_rotated = -u * np.sin(angle_rad) + v * np.cos(angle_rad)
	return u_rotated, v_rotated

rotated_stress = {}

h_start = np.datetime64("2004-03-25")
h_end = np.datetime64("2004-03-30")

for name, (xi, eta) in coords.items():
	# 1. Extract and rotate as usual
	u_raw = us.isel(eta_u=eta, xi_u=xi-1).values/1.2
	v_raw = vs.isel(eta_v=eta-1, xi_v=xi).values/1.2
	angle = grid.angle.isel(eta_rho=eta, xi_rho=xi).values.item()
	
	u_along, v_across = apply_rotation(u_raw, v_raw, angle)
	
	# 2. SAVE to the dictionary first
	rotated_stress[name] = {
		'tau_x': u_along.copy(), # Use .copy() to ensure we don't accidentally modify other refs
		'tau_y': v_across.copy(),
		'time': us.ocean_time.values
	}
	
	# 3. NOW check for 'D' and modify the newly created entry
	if name == 'D':
		# Use the time directly from the xarray object 'us'
		time_mask = (us.ocean_time >= h_start) & (us.ocean_time <= h_end)
		
		# Modify the arrays inside the dictionary
		rotated_stress['D']['tau_x'][time_mask] *= 1.5
		rotated_stress['D']['tau_y'][time_mask] *= 1.5


###########################################################################
####################-----> PSD rotary spectra <---###############################


# ----> Importing the data <----
ds1 = avg_1km

ds, xgrid = xroms.roms_dataset(ds1)

ubar_rho = xroms.to_rho(ds.ubar, xgrid)
vbar_rho = xroms.to_rho(ds.vbar, xgrid)

# ----> Creating the rotary spectra <----
fs = 1.0       # Hourly data -> Output frequencies will be in Cycles Per Hour (cph)
nperseg = 256  


points_psd = {
	'A': [390, 1233],
	'B': [276, 896],
	'C': [333, 557],
	'D': [463, 225]
}

spectra_results = {}

for name, coords in points_psd.items():
	xi, eta = coords[0], coords[1]
	
	u_ts = ubar_rho.isel(xi_rho=xi, eta_rho=eta).values
	v_ts = vbar_rho.isel(xi_rho=xi, eta_rho=eta).values
	
	w = u_ts + 1j * v_ts
	
	freqs, psd = signal.welch(w, fs=fs, window='hann',nperseg=nperseg,noverlap=nperseg//2, detrend='constant', scaling='density', return_onesided=False)
	
	freqs = np.fft.fftshift(freqs)
	psd = np.fft.fftshift(psd)
	
	cw_mask = freqs < 0
	ccw_mask = freqs > 0
	
	# Inertial period in hours
	f_inertial = np.absolute(ds.f.isel(xi_rho=xi, eta_rho=eta)).values
	T = (2 * np.pi / f_inertial) / 3600 

	spectra_results[name] = {
		'freqs_ccw': freqs[ccw_mask],
		'psd_ccw': psd[ccw_mask],
		'freqs_cw': np.abs(freqs[cw_mask]), 
		'psd_cw': psd[cw_mask],
		'T_inertial': T
	}
	
	print(f"Finished processing rotary spectra for Point {name}")

# ----> Calculate Mean Near-Inertial Band <----
f_mean = np.absolute(ds.f.mean()) 
T_mean = (2 * np.pi / f_mean) / 3600 # mean inertial period in hours
f_cph = 1/T_mean
# Convert periods to ordinary frequency (cph)
f1 = 1 / (0.8 * T_mean).values
f2 = 1 / (1.1 * T_mean).values

M2_T = 12.42
cph_m2_minus_f = (1/M2_T) - f_cph
cph_m2_plus_f = (1/M2_T) + f_cph





###########################################################################
####################-----> Alpha <---###############################

def natural_keys(text):
	"""Sorts strings numerically rather than alphabetically."""
	return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


alpha_path_no = "/Users/piero/arian/data1/NO_WINDS_IT/alpha/"

alpha_n  = sorted(glob.glob(alpha_path_no + 'alpha_slice_*.nc'), key=natural_keys)
alpha_no = xr.open_mfdataset(alpha_n, combine='nested', concat_dim='eta_rho', parallel=True)

a_r = alpha_no.alpha.mean(dim='ocean_time').compute()
lon_a = a_r.lon_rho
lat_a = a_r.lat_rho



coords = {
	'A': [380, 1233],
	'B': [276, 896],
	'C': [333, 557],
	'D': [453, 323]
}
###########################################################################
####################-----> SSH tidal definition <---#######################
from scipy.signal import butter, filtfilt, hilbert
# 1. Define coordinates [xi_rho, eta_rho]

# 2. Helper function for a semi-diurnal bandpass filter (covers M2 and S2)
def semi_diurnal_bandpass(data, dt_hours):
	nyquist = 0.5 * (1.0 / dt_hours)
	# Target periods between 11 and 13 hours
	low_cut = (1.0 / 13.0) / nyquist
	high_cut = (1.0 / 11.0) / nyquist
	
	b, a = butter(3, [low_cut, high_cut], btype='band')
	return filtfilt(b, a, data)

# List to collect metadata for the summary file
summary_list = []

# Assuming your data array is named 'avg_1km'
for point_name, (xi_idx, eta_idx) in coords.items():
	print(f"Processing Point {point_name}...")
	
	# Extract data and compute the Dask array chunk into memory
	point_series = avg_1km.zeta.isel(eta_rho=eta_idx, xi_rho=xi_idx,ocean_time=slice(80,None)).compute()
	time_values = point_series.ocean_time.values
	
	# Calculate time spacing dynamically in hours
	dt_hours = np.mean(np.diff(time_values) / np.timedelta64(1, 'h'))
	
	# Signal processing
	ssh_demeaned = point_series.values - np.mean(point_series.values)
	ssh_filtered = semi_diurnal_bandpass(ssh_demeaned, dt_hours)
	ssh_envelope = np.abs(hilbert(ssh_filtered))
	
	# Identify Spring and Neap positions (with 24-hour edge padding)
	pad = int(24 / dt_hours)
	search_zone = ssh_envelope[pad:-pad]
	
	spring_idx = np.argmax(search_zone) + pad
	neap_idx = np.argmin(search_zone) + pad
	
	# --- Save full time-series for later plotting ---
	df_ts = pd.DataFrame({
		'ocean_time': time_values,
		'ssh_raw_demeaned': ssh_demeaned,
		'ssh_filtered_tide': ssh_filtered,
		'ssh_envelope': ssh_envelope
	})
	
	ts_filename = f'point_{point_name}_tidal_series.csv'
	df_ts.to_csv(ts_filename, index=False)
	print(f" -> Saved time-series to: {ts_filename}")
	
	# --- Collect Spring/Neap summary stats ---
	summary_list.append({
		'Point': point_name,
		'eta_rho': eta_idx,
		'xi_rho': xi_idx,
		'Spring_Time': time_values[spring_idx],
		'Spring_Amplitude_m': ssh_envelope[spring_idx],
		'Neap_Time': time_values[neap_idx],
		'Neap_Amplitude_m': ssh_envelope[neap_idx]
	})

# 3. Create and save the global summary file
df_summary = pd.DataFrame(summary_list)
summary_filename = 'tidal_spring_neap_summary.csv'
df_summary.to_csv(summary_filename, index=False)

print(f"\nProcessing complete. Overall summary saved to: {summary_filename}")


df_summary['Spring_Time'] = pd.to_datetime(df_summary['Spring_Time'])
df_summary['Neap_Time'] = pd.to_datetime(df_summary['Neap_Time'])

# Calculate the global MEAN Spring and Neap times across all points
mean_spring_time = df_summary['Spring_Time'].mean()
mean_neap_time = df_summary['Neap_Time'].mean()

# Load and compile individual time-series to create a regional average
points = ['A', 'B', 'C', 'D']
ts_list = []

for p in points:
	df = pd.read_csv(f'point_{p}_tidal_series.csv')
	df['ocean_time'] = pd.to_datetime(df['ocean_time'])
	ts_list.append(df.set_index('ocean_time'))

# Average the physical quantities across all 4 locations
df_mean = pd.concat(ts_list).groupby('ocean_time').mean().reset_index()

# Find the exact envelope values at the nearest index to the mean timestamps
idx_spring = (df_mean['ocean_time'] - mean_spring_time).abs().idxmin()
idx_neap = (df_mean['ocean_time'] - mean_neap_time).abs().idxmin()

actual_spring_time = df_mean.loc[idx_spring, 'ocean_time']
spring_env_val = df_mean.loc[idx_spring, 'ssh_envelope']

actual_neap_time = df_mean.loc[idx_neap, 'ocean_time']
neap_env_val = df_mean.loc[idx_neap, 'ssh_envelope']



def format_time_axis(ax):
	"""Standardizes time axis formatting."""
	ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
	ax.tick_params(axis='x', labelsize=10, rotation=25)
	ax.set_xlabel(None)

#########################################################
#####----> Catarina blend Track <---##########################

ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc_neumann',catarina=True)

storm = ibtracs.get_storm(('catarina',2004))
#m/s we multiply knots by 0.5144444444
vmax_cat = storm.vmax *0.5144444444 
time_cat = storm.time

KT2MS = 0.5144444444
WIND_AVG_FACTOR = 0.93   # 1-min sustained to equivalent-neutral scaling
ALPHA_TRANS = 0.55       # Translation asymmetry weight
OMEGA = 7.2921e-5        # Earth's angular velocity (rad/s)
RMW_BT_KM = 20.0         # Rmax in km

# 1. Pull best track data from tropycal
t = pd.to_datetime(storm.time)
lat = np.asarray(storm.lat, float)
lon = np.asarray(storm.lon, float)
vmax_bt = np.asarray(storm.vmax, float) * KT2MS

# 2. Calculate storm translation velocity (V_trans) using centered differences
ts = np.asarray((t - t[0]).total_seconds())   # float seconds, resolution-proof
dxe = np.gradient(lon) * 111320.0 * np.cos(np.deg2rad(lat))
dyn = np.gradient(lat) * 111320.0
dts = np.gradient(ts)

utr = dxe / dts
vtr = dyn / dts
v_trans = np.hypot(utr, vtr)  # Translation speed in m/s

# 3. Calculate Coriolis parameter |f| for each latitude step
f = 2.0 * OMEGA * np.sin(np.deg2rad(lat))
abs_f = np.abs(f)

# 4. Step A: Back out background forward motion to find the symmetric intensity
vmax_sym = np.maximum(vmax_bt * WIND_AVG_FACTOR - ALPHA_TRANS * v_trans, 5.0)

# 5. Step B: Evaluate the Holland core equation at r = Rmax (Gradient Balance)
rmax_m = RMW_BT_KM * 1000.0  # Convert km to meters
v_gradient_max = np.sqrt(vmax_sym**2 + (rmax_m * abs_f / 2.0)**2) - (rmax_m * abs_f / 2.0)

# 6. Step C: Add back translation asymmetry at the peak eyewall vector location
vmax_blend = v_gradient_max + ALPHA_TRANS * v_trans


###########################################################################
####################-----> Plotting <---###############################


fig = plt.figure(figsize=(9, 12)) 
gs = gridspec.GridSpec(nrows=4, ncols=2, height_ratios=[3,3,2,2], width_ratios=[1,1])
gs.update(left=0.08, right=0.98, hspace=0.2, wspace=0.1, top=0.98, bottom=0.05)

##############------> PSD (Spans all 3 columns automatically via :)
colors = {'A': 'k', 'B': 'blue', 'C': 'orange', 'D': 'red'}
ax_g = plt.subplot(gs[2, :])
ax_g.text(0.02, 0.92, '(e)', transform=ax_g.transAxes, fontsize=10, fontweight='bold')

dof = 5.0  
alpha = 0.05 
ci_lower = dof / stats.chi2.ppf(1 - alpha / 2, dof)
ci_upper = dof / stats.chi2.ppf(alpha / 2, dof)

for name, color in colors.items():
	if name not in spectra_results:
		continue
		
	data = spectra_results[name]
	
	# 1. CCW (Positive Frequencies)
	f_ccw = data['freqs_ccw']
	psd_ccw = data['psd_ccw']
	lower_ccw = psd_ccw * ci_lower
	upper_ccw = psd_ccw * ci_upper
	
	ax_g.plot(f_ccw, psd_ccw, linewidth=1.5, color=color, linestyle='-', label=f'{name} (CCW)')
	ax_g.fill_between(f_ccw, lower_ccw, upper_ccw, color=color, alpha=0.15, edgecolors='none')

	# 2. CW (Negative Frequencies)
	f_cw = data['freqs_cw']
	psd_cw = data['psd_cw']
	lower_cw = psd_cw * ci_lower
	upper_cw = psd_cw * ci_upper
	
	ax_g.plot(f_cw, psd_cw, linewidth=1.5, color=color, linestyle='--', label=f'{name} (CW)')
	ax_g.fill_between(f_cw, lower_cw, upper_cw, color=color, alpha=0.15, edgecolors='none')

# Formatting
ax_g.set_xscale('log')
ax_g.set_yscale('log')

# Adjusted X-limits for CPH (Nyquist is 0.5)
ax_g.set_xlim(1e-2, 0.5)  

# --- Reference Lines (Labels removed) ---
freq_m2 = 1 / 12.42
freq_m4 = 2 / 12.42

ax_g.axvline(f_cph, color='lime', linestyle='-', linewidth=1.5)
ax_g.axvline(freq_m2, color='k', linestyle='-', linewidth=1.5)
ax_g.axvline(freq_m4, color='k', linestyle='--', linewidth=1.5)

# Shading for f_cph
ax_g.axvspan(0.95 * f_cph, 1.1 * f_cph, color='lime', alpha=0.1,label='0.9 - 1.1')

# --- Adding Text Labels Next to Lines ---
# We use get_xaxis_transform() so X is in frequency units and Y is a fraction (0 to 1)
text_y_pos = 0.35 # Places the text near the top of the plot

# Common text properties dictionary to keep code clean
txt_props = dict(transform=ax_g.get_xaxis_transform(), rotation=90, 
				 verticalalignment='top', horizontalalignment='right', fontsize=12)

ax_g.text(f_cph, text_y_pos, r' $f$ ', color='lime', **txt_props)
ax_g.text(freq_m2, text_y_pos, r' M$_2$ ', color='0.4', **txt_props)
ax_g.text(freq_m4, text_y_pos, r' M$_4$ ', color='0.4', **txt_props)

ax_g.set_xlabel('Frequency [cph]', fontsize=10)
# Safely rendering the ylabel to avoid LaTeX errors while maintaining superscript formatting
ax_g.set_ylabel(r'PSD [m$^2 \cdot$ s$^{-2} \cdot$ cph$^{-1}$]', fontsize=10)
ax_g.grid(True, which="both", ls=":", alpha=0.5)

ax_g.legend(loc='lower left', fontsize='x-small', ncol=3)



##############------> Loc, z and deproas moorings



# Create a custom colormap with mint cream, beige, purple, and blue
colors = [
	#(0.96, 1.0, 0.98),   # Very shallow (mint cream, almost white)
	(0.9, 0.8, 0.7),     # Shallow (beige)
	(0.7, 1.0, 0.7),     # Mid-deep (pastel green) 
	(0.5, 0.4, 0.8),     # Mid-depth (purple)
	(1.0, 0.3, 0.5),     # Mid-depth (pink or vibrant red)
	(1.0, 0.65, 0.0),    # Shallow (vibrant orange/gold)
	(0.2, 0.2, 0.7),     # Deeper (blue)
	(0.1, 0.1, 0.4),     # Deepest (dark blue)
]


cmap = LinearSegmentedColormap.from_list('bathymetry_cmap', colors, N=500)
vmin = -5
vmax = 2500
#norm = LogNorm(vmin=vmin, vmax=vmax)
norm = cm.colors.Normalize(vmin=vmin,vmax=vmax)
levels=np.linspace(vmin,vmax,200)

bar_title = "m"
ax = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax.set_aspect('auto')
ax.set_ylim(bottom =-33,top = -21)
ax.set_xlim(left = -52,right = -39.2)

### - contourf
norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
ax.contourf(lon_rho, lat_rho, h,  levels=levels, cmap=cmap, norm=norm, extend='max', zorder=0)
#--> deproas moorings
ax.scatter(-41.73357500, -23.73347222, zorder=5, s=30, marker='*', color='firebrick', label='CF3')
ax.scatter(-42.571 , -23.7246666667, zorder=5, s=30, marker='+', color='purple', label='BG3')
ax.scatter(-44.3681666667, -24.392, zorder=5, s=30, marker='s', color='orange', label='UB3')
#--> canyons scatter
ax.scatter(-44.9, -25.3, zorder=5, s=200, marker='|', color='k', label='Canyons')
ax.scatter(-45.5, -25.9, zorder=5, s=200, marker='|', color='k')
ax.scatter(-46.9, -27.5, zorder=5, s=200, marker='_', color='k')
ax.text(0.7, 0.98, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
		

legend = ax.legend(loc=4, fontsize='x-small', markerscale=0.7)

# perks
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
ax.patch.set_edgecolor('black')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				  linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)

gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 8, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 8, 'color': 'dimgrey'}

## isobaths
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]

levels1 = np.asarray(levels_1)
levels2 = np.asarray(levels_2)
levels3 = np.asarray(levels_3)
levels4 = np.asarray(levels_4)

c1 = ax.contour(lon_rho, lat_rho, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = ax.contour(lon_rho, lat_rho, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = ax.contour(lon_rho, lat_rho, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = ax.contour(lon_rho, lat_rho, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)
#ax.hlines(y=-25.8, xmin=-49.2, xmax=-42, linewidth=1, color='purple',zorder = 3)
#ax.text(-45.3, -25.9, 'Santos Bifurcation', fontsize=7, ha='left', va='top', 
	   #bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Use proxy artists to create legend entries
legend_lines = [Line2D([0], [0], linestyle='dotted', linewidth=1, color='brown'),
				Line2D([0], [0], linestyle='dotted', linewidth=1, color='grey'),
				Line2D([0], [0], linestyle='dashed', linewidth=1, color='k'),
				Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')]

labels = ['50 m', '200 m', '1000 m', '2000 m']

# Updated legend part using fig.legend
#fig.legend(legend_lines, labels, loc='upper right', title='Isobaths', fontsize='x-small', title_fontsize='x-small')

fig.legend(
	legend_lines,
	labels,
	title='Isobaths',
	fontsize='x-small',
	title_fontsize='x-small',
	loc='center',
	bbox_to_anchor=(0.12, 0.88)
)


### -- Miniglobe
extent = [-49, -39.3, -29.5, -22]
center = [-45, -15]
box_color = 'red'
box_alpha = 1
box_edge_width = 1

sub_ax = inset_axes(ax, width="100%", height="100%", loc="lower center", bbox_to_anchor=(0.75, 0.25, 0.25, 0.25),
					bbox_transform=ax.transAxes,
					axes_class=cartopy.mpl.geoaxes.GeoAxes,
					axes_kwargs=dict(map_projection=ccrs.Orthographic(center[0], center[1])))
# Make a nice border around the inset axes.

# Add land and ocean features to the miniglobe
land_feature = NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor=COLORS['land'])
ocean_feature = NaturalEarthFeature('physical', 'ocean', '110m', edgecolor='face', facecolor=COLORS['water'])

sub_ax.add_feature(land_feature)
sub_ax.add_feature(ocean_feature)

effect = Stroke(linewidth=.1, foreground='black', alpha=0.5)
sub_ax.patch.set_path_effects([effect])
sub_ax.coastlines(linewidth=0.00000001, edgecolor='k', alpha=.8)
extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='lightgray',
					  edgecolor=box_color, linewidth=box_edge_width, alpha=box_alpha)

# colorbar
cbar_1 = inset_axes(ax, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
norm_1 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm_1, extend='max', orientation='horizontal')
cb1.set_label(bar_title, size='x-small')
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small')  # Set colorbar label size

CF = [-23.018, -41.97]
CS = [-22.015, -40.963]
CM = [-28.60, -48.809]
SSI = [-23.8310, -45.4089]

# CS
#ax.scatter(CS[1], CS[0],marker='D',zorder = 3, color='k', s=20)
#ax.text(CS[1] - 0.3, CS[0], 'CST', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# CF
ax.scatter(CF[1], CF[0],marker='D',zorder = 3, color='k', s=20)
ax.text(CF[1] - 0.3, CF[0] + 0.1, 'CF', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# CM
ax.scatter(CM[1], CM[0],marker='D',zorder = 3, color='k', s=20)
ax.text(CM[1] -0.25, CM[0] + 0.3, 'CSM', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')

# SSI
ax.scatter(SSI[1], SSI[0],marker='.',zorder = 3, color='k', s=20)
ax.text(SSI[1] -0.25, SSI[0] + 0.3, 'SSI', ha='center', va='bottom', fontsize=6,fontweight='bold', color='k')


#---> grid_subgrid
ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.set_aspect('auto')

ax2.scatter(lon_c_flat[coarse_ocean], lat_c_flat[coarse_ocean],s=0.001, color='blue', alpha=0.5, transform=ccrs.PlateCarree())
ax2.scatter(lon_f_flat, lat_f_flat, s=0.001, color='indianred', alpha=0.5, transform=ccrs.PlateCarree(),zorder = 2)

ax2.scatter(lon_c_flat[coarse_land], lat_c_flat[coarse_land],s=0.001, color='gray', alpha=1, transform=ccrs.PlateCarree(),zorder = 2)
#ax.scatter(lon_f_flat[finer_land], lat_f_flat[finer_land], s=0.001, color='whitesmoke', alpha=1, transform=ccrs.PlateCarree(),zorder = 2)

gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 8, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 8, 'color': 'dimgrey'}
# Custom legend
legend_patches = [
	mpatches.Patch(color='blue', label='Coarser'),
	mpatches.Patch(color='indianred', label='Finer'),

]
ax2.text(0.7, 0.98, '(b)', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')


###########---> Plotting the hurricane track


type_colors = {
	'TD': {'color': 'yellowgreen', 'marker': 'o', 'label': 'Tropical Depression'},
	'TS': {'color': 'magenta', 'marker': 'o', 'label': 'Tropical Storm'},
	'HU': {'color': 'gold', 'marker': 'o', 'label': 'Hurricane'}, # Catarina reached HU status
	'SS': {'color': 'red', 'marker': 'o', 'label': 'Subtropical Storm'},
	'EX': {'color': 'grey', 'marker': 's', 'label': 'Extratropical'}
	# Add other types if needed (e.g., LO, wave, DB, etc.)
}

# --- 2. Plot the Storm Track (Line) ---
# Plotting the line connects the track points sequentially
# Ensure zorder is high so it plots over the bathymetry (zorder=0) and land (zorder=1).
ax2.plot(storm.lon, storm.lat, 
		color='k', 
		linestyle='--', 
		linewidth=0.8, 
		alpha=0.6,
		transform=ccrs.PlateCarree(), 
		zorder=6,
		label='Catarina Track')


# --- 3. Plot Scatter Points with Color by Type ---

# Get the unique storm types present in your storm data
unique_types = np.unique(storm.type)

# Loop through each unique type to plot points and create a legend entry
for storm_type in unique_types:
	if storm_type in type_colors:
		style = type_colors[storm_type]
		
		# Create a boolean mask to select points matching the current type
		mask = (storm.type == storm_type)
		
		# Plot the points
		ax2.scatter(storm.lon[mask], storm.lat[mask], 
				   c=style['color'], 
				   marker=style['marker'], 
				   s=30, # size of the marker
				   edgecolors='k', # black outline for contrast
				   linewidths=0.5,
				   transform=ccrs.PlateCarree(), 
				   zorder=7, 
				   label=style['label'])

# --- 4. Plot Initial and Final Points (Optional, for emphasis) ---

# Optional: Add a large marker for the starting point
ax2.scatter(storm.lon[0], storm.lat[0], 
		   marker='^', 
		   s=60, 
		   color='green', 
		   edgecolor='k', 
		   linewidth=1,
		   transform=ccrs.PlateCarree(), 
		   zorder=8, 
		   label='Start')
		   
# Optional: Add a large marker for the final point
ax2.scatter(storm.lon[-1], storm.lat[-1], 
		   marker='v', 
		   s=60, 
		   color='brown', 
		   edgecolor='k', 
		   linewidth=1,
		   transform=ccrs.PlateCarree(), 
		   zorder=8, 
		   label='End')



ax2.legend(loc='upper left', fontsize=6,  title_fontsize=6)

###---> vamx
pcx = inset_axes(ax2, width="50%", height="30%", loc='lower left',
				 bbox_to_anchor=(0.1, 0.05, 0.96, 0.95), 
				 bbox_transform=ax2.transAxes)

# Give it a solid background so map lines don't bleed through
pcx.set_facecolor('white')
pcx.patch.set_alpha(0.9) 

# Inverted data: Vmax on X-axis, Time on Y-axis
pcx.plot(time_cat, vmax_cat, label='Track', linestyle=':', color='green', linewidth=1.2)
pcx.plot(time_cat,vmax_blend, label='Blend', linestyle='--', color='orange', linewidth=1.2)

# Formatting the inset
pcx.set_ylim(10, 45)
pcx.set_ylabel(r'V$_{max}$ (m s$^{-1}$)', fontsize=7)
pcx.tick_params(axis='both', labelsize=7)

# Handle time formatting on the Y-axis instead of X
pcx.xaxis.set_major_locator(mdates.DayLocator(interval=3))
pcx.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

pcx.grid(True, linestyle=':', alpha=0.5)
pcx.legend(loc=2, fontsize=6, frameon=False)




####################_---> Alpha ratio


A = [380, 1233]
B = [276, 896]
C = [333, 557]
D = [453, 323]

axa = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
axa.set_aspect('auto')
axa.set_ylim(bottom =-33,top = -21)
axa.set_xlim(left = -52,right = -39.2)

### - contourf

cmap = plt.cm.terrain_r
vmin = 0
vmax = 3
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
bar_title = '\u03B1'
axa.text(0.7, 0.98, '(c)', transform=axa.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

axa.contourf(lon_a, lat_a, a_r,  levels=200, cmap=cmap, norm=norm, extend='max', zorder=0)
#--> virtual moorings
axa.scatter(lon_rho[A[1],A[0]], lat_rho[A[1],A[0]], zorder=5, s=40, marker='*', color='k', label='A')
axa.scatter(lon_rho[B[1],B[0]], lat_rho[B[1],B[0]], zorder=5, s=40, marker='v', color='blue', label='B')
axa.scatter(lon_rho[C[1],C[0]], lat_rho[C[1],C[0]], zorder=5, s=40, marker='s', color='orange', label='C')
axa.scatter(lon_rho[D[1],D[0]], lat_rho[D[1],D[0]], zorder=5, s=40, marker='^', color='red', label='D')


legend = axa.legend(loc=4, fontsize='x-small', markerscale=0.7)

# perks
axa.coastlines()
axa.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=1)
axa.patch.set_edgecolor('black')
gl = axa.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
				  linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)

gl.top_labels = False
gl.left_labels = True
gl.right_labels = False
gl.bottom_labels = True
gl.xlines = False
gl.ylines = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 8, 'color': 'dimgrey'}
gl.ylabel_style = {'size': 8, 'color': 'dimgrey'}

## isobaths
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]

levels1 = np.asarray(levels_1)
levels2 = np.asarray(levels_2)
levels3 = np.asarray(levels_3)
levels4 = np.asarray(levels_4)

c1 = axa.contour(lon_rho, lat_rho, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = axa.contour(lon_rho, lat_rho, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = axa.contour(lon_rho, lat_rho, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = axa.contour(lon_rho, lat_rho, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

# colorbar
cbar_1 = inset_axes(axa, width="60%", height="3%", loc=2)
cbar_1.set_facecolor('lightgray')
cb1 = mpl.colorbar.ColorbarBase(cbar_1, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
cb1.set_label(bar_title, size='x-small')
cbar_1.xaxis.set_ticks_position('bottom')
cbar_1.tick_params(axis='x', labelsize='x-small')  # Set colorbar label size

####################################################
#####-----> Stick plot<---##########
point_labels = ['A', 'B', 'C', 'D']
offsets = [0.8, 0.6, 0.4, 0.2]  # Tighten offsets for stress magnitudes
ax4 = plt.subplot(gs[1, 1])
ax4.text(0.1, 0.98, '(d)', transform=ax4.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

# Use the time from the xarray object
time = rotated_stress['A']['time']

for i, name in enumerate(['A', 'B', 'C', 'D']):
	data = rotated_stress[name]
	offset = offsets[i]
	
	# Quiver for wind stress
	q = ax4.quiver(data['time'], np.full(len(data['time']), offset),
				   data['tau_x'], data['tau_y'],
				   color='black', 
				   alpha=0.8,
				   units='y',     
				   scale=3,      
				   headlength=0,   
				   headaxislength=0, 
				   width=0.008)

# 3. Update Quiver Legend for Stress
ax4.quiverkey(q, X=0.82, Y=0.96, U=0.1 ,
			  label=r'0.1 $Pa$', labelpos='E', 
			  coordinates='axes', fontproperties={'weight': 'bold'})

# Formatting
ax4.set_yticks(offsets)
ax4.set_yticklabels(point_labels)
ax4.set_ylim(0, 1.0)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.xticks(rotation=15)
ax4.set_ylabel(None)

h_start = np.datetime64("2004-03-25")
h_end = np.datetime64("2004-03-30") 
ax4.axvspan(h_start, h_end, color='red', alpha=0.15, zorder=0)
ax4.text(h_start, 1, 'Hurricane',color='red', fontsize=8, fontweight='bold', ha='left',va='bottom') 
plt.tight_layout()

####################################################
#####-----> Stick plot<---##########
point_labels = ['A', 'B', 'C', 'D']
offsets = [0.8, 0.6, 0.4, 0.2]  # Tighten offsets for stress magnitudes
ax4 = plt.subplot(gs[1, 1])

# Use the time from the xarray object
time = rotated_stress['A']['time']

for i, name in enumerate(['A', 'B', 'C', 'D']):
	data = rotated_stress[name]
	offset = offsets[i]
	
	# Quiver for wind stress
	q = ax4.quiver(data['time'], np.full(len(data['time']), offset),
				   data['tau_x'], data['tau_y'],
				   color='black', 
				   alpha=0.8,
				   units='y',     
				   scale=3,      
				   headlength=0,   
				   headaxislength=0, 
				   width=0.008)

# 3. Update Quiver Legend for Stress
ax4.quiverkey(q, X=0.82, Y=0.96, U=0.1 ,
			  label=r'0.1 $Pa$', labelpos='E', 
			  coordinates='axes', fontproperties={'weight': 'bold'})

# Formatting
ax4.set_yticks(offsets)
ax4.set_yticklabels(point_labels)
ax4.set_ylim(0, 1.0)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax4.xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.xticks(rotation=15)
ax4.set_ylabel(None)

h_start = np.datetime64("2004-03-25")
h_end = np.datetime64("2004-03-30") 
ax4.axvspan(h_start, h_end, color='red', alpha=0.15, zorder=0)
ax4.text(h_start, 1, 'Hurricane',color='red', fontsize=8, fontweight='bold', ha='left',va='bottom') 



##########----> Tides ssh <---#########

# Final row (f) configuration (Spans all columns of the 4th row)
ax_tide = plt.subplot(gs[3, :])
ax_tide.text(0.01, 0.1, '(f)', transform=ax_tide.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

# Plot regional averaged background SSH and Filtered Tide
ax_tide.plot(df_mean['ocean_time'], df_mean['ssh_raw_demeaned'], 
			 color='grey', alpha=0.6, label='Demeaned SSH')

ax_tide.plot(df_mean['ocean_time'], df_mean['ssh_filtered_tide'], 
			 color='tab:blue', alpha=0.8, linewidth=1.5, label='Semi-diurnal SSH')

# Plot the structural envelope line
ax_tide.plot(df_mean['ocean_time'], df_mean['ssh_envelope'], 
			 color='tab:orange', linestyle='--', linewidth=2, label='M2 Envelope')

# Mark the calculated Mean Spring position
ax_tide.scatter(actual_spring_time, spring_env_val, color='red', s=120, zorder=5,
			   label=f'Spring')
ax_tide.axvline(actual_spring_time, color='red', linestyle=':', alpha=0.7, linewidth=1.2)

# Mark the calculated Mean Neap position
ax_tide.scatter(actual_neap_time, neap_env_val, color='green', s=120, zorder=5,
			   label=f'Neap)')
ax_tide.axvline(actual_neap_time, color='green', linestyle=':', alpha=0.7, linewidth=1.2)


# ==========================================
# 4. Final Formatting
# ==========================================
ax_tide.set_ylabel('SSH (m)', fontsize=10)
ax_tide.set_xlabel(None)
ax_tide.grid(True, linestyle=':', alpha=0.5)

# Place the legend inside or just outside the subplot space
ax_tide.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)

format_time_axis(ax_tide)

plt.tight_layout()
plt.savefig('lock_track_alpha_winds_psd.png', dpi = 300)
