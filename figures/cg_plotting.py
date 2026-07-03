#################################


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
import dask

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

##############################################################################
###########################-----> NO WINDS <----#############################
##############################################################################


no_path = '/Users/piero/arian/data1/NO_WINDS_IT/CG/' 

cg_no_f  = sorted(glob.glob(no_path + 'piint_no_wind_*.nc'), key=natural_keys)
cg_no = xr.open_mfdataset(cg_no_f, combine='nested', concat_dim='eta_rho', parallel=True)

#---> maps #mW/m2
cg_no_before_map = cg_no.pi_int.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time').compute()
cg_no_during_map = cg_no.pi_int.sel(ocean_time=slice(during_start, during_end)).mean(dim='ocean_time').compute()
cg_no_after_map = cg_no.pi_int.sel(ocean_time=slice(after_start, after_end)).mean(dim='ocean_time').compute()

####-----> Importing Alpha files <----#####
alpha_path = "/Users/piero/arian/data1/NO_WINDS_IT/alpha/"

alpha_f  = sorted(glob.glob(alpha_path + 'alpha_slice_*.nc'), key=natural_keys)
alpha = xr.open_mfdataset(alpha_f, combine='nested', concat_dim='eta_rho', parallel=True).isel(eta_rho=cg_no.eta_rho.values).rolling(ocean_time=8, center=True, min_periods=1).mean()
alpha_w = alpha.sel(ocean_time=cg_no.ocean_time, method='nearest')

lat_rho = alpha_w.lat_rho.compute()
lon_rho = alpha_w.lon_rho.compute()


######----> making some summations
import xroms
ds1 = xr.open_dataset('/Users/piero/arian/data1/nc_outs/avg_internal_tides_paper.nc', chunks={'ocean_time': 1, 's_rho': -1, 'eta_rho': 'auto', 'xi_rho': 'auto'})
ds, xgrid = xroms.roms_dataset(ds1)

ds_fix = ds.isel(eta_rho=slice(0, 1280), xi_rho=slice(40, None))

h = ds_fix.h.compute()
lat_h = h.lat_rho
lon_h = h.lon_rho
dA = ds_fix.dA  # Extract area array for cleaner code

# --- 2. Define Region Masks ---
# Shelf: depth less than 200m
mask_shelf = h < 250

# Slope: depth between 200m and 2500m
#mask_slope = (h >= 250) & (h <= 3500)
mask_slope = h>=250


# --- Total CG ---
cg_no_bf_val = float((cg_no_before_map * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_no_dr_val = float((cg_no_during_map * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_no_af_val = float((cg_no_after_map * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)

# --- Shelf CG (h < 200m) ---
cg_no_bf_shelf = float((cg_no_before_map.where(mask_shelf) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_no_dr_shelf = float((cg_no_during_map.where(mask_shelf) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_no_af_shelf = float((cg_no_after_map.where(mask_shelf) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)

# --- Slope CG (200m <= h <= 2500m) ---
cg_no_bf_slope = float((cg_no_before_map.where(mask_slope) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_no_dr_slope = float((cg_no_during_map.where(mask_slope) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_no_af_slope = float((cg_no_after_map.where(mask_slope) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)


print(f"{'Scenario':<10} | {'Total CG':<10} | {'Shelf CG':<10} | {'Slope CG':<10}")
print(f"Before     | {cg_no_bf_val:.2f} | {cg_no_bf_shelf:.2f} | {cg_no_bf_slope:.2f}")
print(f"During     | {cg_no_dr_val:.2f} | {cg_no_dr_shelf:.2f} | {cg_no_dr_slope:.2f}")
print(f"After      | {cg_no_af_val:.2f} | {cg_no_af_shelf:.2f} | {cg_no_af_slope:.2f}")

#---> per time per region
dA_south = ds_fix.dA.isel(eta_rho=south_slice)
total_area_south = dA_south.sum(dim=['eta_rho', 'xi_rho']).compute()
dA_central = ds_fix.dA.isel(eta_rho=central_slice)
total_area_central = dA_central.sum(dim=['eta_rho', 'xi_rho']).compute()
dA_north = ds_fix.dA.isel(eta_rho=north_slice)
total_area_north = dA_north.sum(dim=['eta_rho', 'xi_rho']).compute()

cg_no_time = cg_no.pi_int


cg_no_south = ((cg_no_time.isel(eta_rho=south_slice) * dA_south).sum(dim=['eta_rho', 'xi_rho'])).compute()/total_area_south

cg_no_central = ((cg_no_time.isel(eta_rho=central_slice) * dA_central).sum(dim=['eta_rho', 'xi_rho'])).compute() /total_area_central

cg_no_north = ((cg_no_time.isel(eta_rho=north_slice) * dA_north).sum(dim=['eta_rho', 'xi_rho'])).compute() /total_area_north



##############################################################################
###########################-----> WITH WINDS <----#############################
##############################################################################


def remove_lat(ds):
	return ds.drop_vars('lat_rho', errors='ignore')

cg_with = xr.open_mfdataset(
	cg_with_f, 
	combine='nested', 
	concat_dim='eta_rho', 
	preprocess=remove_lat,
	parallel=True
).isel(eta_rho=slice(0, 1280))

#---> maps #mW/m2
cg_with_before_map = cg_with.pi_int.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time').compute()
cg_with_during_map = cg_with.pi_int.sel(ocean_time=slice(during_start, during_end)).mean(dim='ocean_time').compute()
cg_with_after_map = cg_with.pi_int.sel(ocean_time=slice(after_start, after_end)).mean(dim='ocean_time').compute()

# --- Total CG ---
cg_with_bf_val = float((cg_with_before_map * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_with_dr_val = float((cg_with_during_map * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_with_af_val = float((cg_with_after_map * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)

# --- Shelf CG (h < 200m) ---
cg_with_bf_shelf = float((cg_with_before_map.where(mask_shelf) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_with_dr_shelf = float((cg_with_during_map.where(mask_shelf) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_with_af_shelf = float((cg_with_after_map.where(mask_shelf) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)

# --- Slope CG (200m <= h <= 2500m) ---
cg_with_bf_slope = float((cg_with_before_map.where(mask_slope) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_with_dr_slope = float((cg_with_during_map.where(mask_slope) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
cg_with_af_slope = float((cg_with_after_map.where(mask_slope) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)


print(f"{'Scenario':<10} | {'Total CG':<10} | {'Shelf CG':<10} | {'Slope CG':<10}")
print(f"Before     | {cg_with_bf_val:.2f} | {cg_with_bf_shelf:.2f} | {cg_with_bf_slope:.2f}")
print(f"During     | {cg_with_dr_val:.2f} | {cg_with_dr_shelf:.2f} | {cg_with_dr_slope:.2f}")
print(f"After      | {cg_with_af_val:.2f} | {cg_with_af_shelf:.2f} | {cg_with_af_slope:.2f}")

#---> per time per region
dA_south = ds_fix.dA.isel(eta_rho=south_slice)
total_area_south = dA_south.sum(dim=['eta_rho', 'xi_rho']).compute()
dA_central = ds_fix.dA.isel(eta_rho=central_slice)
total_area_central = dA_central.sum(dim=['eta_rho', 'xi_rho']).compute()
dA_withrth = ds_fix.dA.isel(eta_rho=north_slice)
total_area_withrth = dA_withrth.sum(dim=['eta_rho', 'xi_rho']).compute()

cg_with_time = cg_with.pi_int


cg_with_south = ((cg_with_time.isel(eta_rho=south_slice) * dA_south).sum(dim=['eta_rho', 'xi_rho'])).compute()/total_area_south

cg_with_central = ((cg_with_time.isel(eta_rho=central_slice) * dA_central).sum(dim=['eta_rho', 'xi_rho'])).compute() /total_area_central

cg_with_north = ((cg_with_time.isel(eta_rho=north_slice) * dA_withrth).sum(dim=['eta_rho', 'xi_rho'])).compute() /total_area_withrth





####-----> Neap and spring points <----#####

spring_dates = pd.to_datetime([
	"2004-03-24",   
	"2004-04-07"    
])

neap_dates = pd.to_datetime([
	"2004-04-01"    
])



def add_tidal_markers(ax):

	# small secondary axis
	ax_top = ax.twiny()

	ax_top.set_xlim(ax.get_xlim())

	# move slightly upward
	ax_top.spines['top'].set_position(('axes', 1.01))

	# remove frame
	for spine in ax_top.spines.values():
		spine.set_visible(False)

	ax_top.set_xticks([])
	#ax_top.set_yticks([])

	# ---- SPRING ----
	for d in spring_dates:

		ax_top.plot(
			d, 1,
			marker='o',
			markersize=7,
			color='crimson',
			clip_on=False,
			transform=ax_top.get_xaxis_transform()
		)

		ax_top.text(
			d, 1.05,
			'Spring',
			color='crimson',
			fontsize=8,
			ha='center',
			va='bottom',
			transform=ax_top.get_xaxis_transform()
		)

	# ---- NEAP ----
	for d in neap_dates:

		ax_top.plot(
			d, 1,
			marker='s',
			markersize=6,
			color='royalblue',
			clip_on=False,
			transform=ax_top.get_xaxis_transform()
		)

		ax_top.text(
			d, 1.05,
			'Neap',
			color='royalblue',
			fontsize=8,
			ha='center',
			va='bottom',
			transform=ax_top.get_xaxis_transform()
		)

# =============================================================================
# Plotting Helper Functions
# =============================================================================


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







#######################################################################
############-------> Plotting <---------#######################


fig = plt.figure(figsize=(14, 16)) #2 5 inches per columns
gs = gridspec.GridSpec(nrows=4, ncols=8, height_ratios=[1,1,1,1])
gs.update(left=0.05, right=0.99, wspace=0.4, hspace=0.2, top=0.98, bottom=0.05)

## isobaths
levels_1 = [50]
levels_2 = [200]
levels_3 = [1000]
levels_4 = [2000]

levels1 = np.asarray(levels_1)
levels2 = np.asarray(levels_2)
levels3 = np.asarray(levels_3)
levels4 = np.asarray(levels_4)

#colorbar
unit_map = r'$\Pi$ $\times 10^{-3}$ W m$^{-2}$'


cmap_map = plt.cm.bwr

vmin = -1
vmax = 1
norm_map = mpl.colors.Normalize(vmin=vmin, vmax=vmax)


#####################-----> NO WINDS
# [0, 0] Before Map
a_bf_m = plt.subplot(gs[0, 0:4],projection=ccrs.PlateCarree())
a_bf_m.set_aspect('auto')
a_bf_m.text(1.03, 0.5, 'T1',
			transform=a_bf_m.transAxes,
			rotation=90,
			fontsize=20,
			fontweight='bold',
			va='center',
			ha='center')

a_bf_m.text(0.5, 1.05, 'No Winds',
			transform=a_bf_m.transAxes,
			rotation=0,
			fontsize=15,
			fontweight='bold',
			va='center',
			ha='center')

a_bf_m.text(0.88, 0.9, '(a)', transform=a_bf_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
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


a_bf_m.contourf(lon_rho, lat_rho, cg_no_before_map*1e3, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

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

#a_bf_m.scatter(lon_rho[A[1],A[0]], lat_rho[A[1],A[0]], zorder=5, s=40, marker='^', color='k', label='P1')
#a_bf_m.scatter(lon_rho[B[1],B[0]], lat_rho[B[1],B[0]], zorder=5, s=40, marker='v', color='green', label='P2')
#a_bf_m.scatter(lon_rho[C[1],C[0]], lat_rho[C[1],C[0]], zorder=5, s=40, marker='s', color='cyan', label='P3')


#-->isobaths
c1 = a_bf_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = a_bf_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = a_bf_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = a_bf_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)


# [1, 0] during Map
a_dr_m = plt.subplot(gs[1, 0:4],projection=ccrs.PlateCarree())
a_dr_m.set_aspect('auto')
a_dr_m.text(1.03, 0.5, 'T2',
			transform=a_dr_m.transAxes,
			rotation=90,
			fontsize=20,
			fontweight='bold',
			va='center',
			ha='center')
a_dr_m.text(0.88, 0.9, '(b)', transform=a_dr_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
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
hc_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_map, norm=norm_map, extend='both', orientation='horizontal')
hc_map.set_label(unit_map, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


a_dr_m.contourf(lon_rho, lat_rho, cg_no_during_map*1e3, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

a_dr_m.coastlines()
a_dr_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=5)
a_dr_m.patch.set_edgecolor('black')
gl = a_dr_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
c1 = a_dr_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = a_dr_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = a_dr_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = a_dr_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

# Plot the specific sub-regions

a_dr_m.scatter(lon_rho[central_top,40:], lat_rho[central_top,40:],c='green',s=0.05,marker='.')
a_dr_m.scatter(lon_rho[south_top,40:], lat_rho[south_top,40:],c='cyan',s=0.05,marker='.')

a_dr_m.text(lon_rho[central_top, 609]+0.2, lat_rho[central_top, 609]-1, 'North', color='green',zorder=5, fontsize='x-small', verticalalignment='bottom')
a_dr_m.text(lon_rho[south_top, 609]+0.2, lat_rho[south_top, 609]-1, 'South', color='cyan', zorder=5,fontsize='x-small', verticalalignment='bottom')


# [2, 0] after Map
a_af_m = plt.subplot(gs[2, 0:4],projection=ccrs.PlateCarree())
a_af_m.set_aspect('auto')
a_af_m.text(1.03, 0.5, 'T3',
			transform=a_af_m.transAxes,
			rotation=90,
			fontsize=20,
			fontweight='bold',
			va='center',
			ha='center')

a_af_m.text(0.88, 0.9, '(c)', transform=a_af_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
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
hc_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_map, norm=norm_map, extend='both', orientation='horizontal')
hc_map.set_label(unit_map, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


a_af_m.contourf(lon_rho, lat_rho, cg_no_after_map*1e3, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

a_af_m.coastlines()
a_af_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=5)
a_af_m.patch.set_edgecolor('black')
gl = a_af_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
c1 = a_af_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = a_af_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = a_af_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = a_af_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

# Use proxy artists to create legend entries
legend_lines = [Line2D([0], [0], linestyle='dotted', linewidth=1, color='brown'),
				Line2D([0], [0], linestyle='dotted', linewidth=1, color='grey'),
				Line2D([0], [0], linestyle='dashed', linewidth=1, color='k'),
				Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')]

labels = ['50 m', '200 m', '1000 m', '2000 m']

fig.legend(
	legend_lines,
	labels,
	title='Isobaths',
	fontsize='small',
	title_fontsize='small',
	loc='center',
	bbox_to_anchor=(0.08 , 0.4)
)



#####################-----> WITH WINDS
# [0, 0] Before Map
n_bf_m = plt.subplot(gs[0, 4:8],projection=ccrs.PlateCarree())
n_bf_m.set_aspect('auto')
n_bf_m.text(0.5, 1.05, 'With Winds',
			transform=n_bf_m.transAxes,
			rotation=0,
			fontsize=15,
			fontweight='bold',
			va='center',
			ha='center')

n_bf_m.text(0.88, 0.9, '(d)', transform=n_bf_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
n_bf_m.set_ylim(bottom=-33, top=-20)
n_bf_m.set_xlim(left=-52, right=-39.2)

ax_map_cb = inset_axes(n_bf_m, 
					   width="60%", 
					   height="3%", 
					   loc='upper left',
					   bbox_to_anchor=(0.08, -0.03, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=n_bf_m.transAxes,
					   borderpad=0)

ax_map_cb.set_facecolor('lightgray')

cb_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_map, norm=norm_map, extend='both', orientation='horizontal')
cb_map.set_label(unit_map, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


n_bf_m.contourf(lon_rho, lat_rho, cg_with_before_map*1e3, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

n_bf_m.coastlines()
n_bf_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=5)
n_bf_m.patch.set_edgecolor('black')
gl = n_bf_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
c1 = n_bf_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = n_bf_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = n_bf_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = n_bf_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)


# [1, 0] during Map
n_dr_m = plt.subplot(gs[1, 4:8],projection=ccrs.PlateCarree())
n_dr_m.set_aspect('auto')

n_dr_m.text(0.88, 0.9, '(e)', transform=n_dr_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
n_dr_m.set_ylim(bottom=-33, top=-20)
n_dr_m.set_xlim(left=-52, right=-39.2)

ax_map_cb = inset_axes(n_dr_m, 
					   width="60%", 
					   height="3%", 
					   loc='upper left',
					   bbox_to_anchor=(0.08, -0.03, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=n_dr_m.transAxes,
					   borderpad=0)

ax_map_cb.set_facecolor('lightgray')
hc_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_map, norm=norm_map, extend='both', orientation='horizontal')
hc_map.set_label(unit_map, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


n_dr_m.contourf(lon_rho, lat_rho, cg_with_during_map*1e3, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

n_dr_m.coastlines()
n_dr_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=5)
n_dr_m.patch.set_edgecolor('black')
gl = n_dr_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
c1 = n_dr_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = n_dr_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = n_dr_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = n_dr_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

# Plot the specific sub-regions

n_dr_m.scatter(lon_rho[central_top,40:], lat_rho[central_top,40:],c='green',s=0.05,marker='.')
n_dr_m.scatter(lon_rho[south_top,40:], lat_rho[south_top,40:],c='cyan',s=0.05,marker='.')

n_dr_m.text(lon_rho[central_top, 609]+0.2, lat_rho[central_top, 609]-1, 'North', color='green',zorder=5, fontsize='x-small', verticalalignment='bottom')
n_dr_m.text(lon_rho[south_top, 609]+0.2, lat_rho[south_top, 609]-1, 'South', color='cyan', zorder=5,fontsize='x-small', verticalalignment='bottom')


# [2, 0] after Map
n_af_m = plt.subplot(gs[2, 4:8],projection=ccrs.PlateCarree())
n_af_m.set_aspect('auto')

n_af_m.text(0.88, 0.9, '(f)', transform=n_af_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
n_af_m.set_ylim(bottom=-33, top=-20)
n_af_m.set_xlim(left=-52, right=-39.2)

ax_map_cb = inset_axes(n_af_m, 
					   width="60%", 
					   height="3%", 
					   loc='upper left',
					   bbox_to_anchor=(0.08, -0.03, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=n_af_m.transAxes,
					   borderpad=0)

ax_map_cb.set_facecolor('lightgray')
hc_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_map, norm=norm_map, extend='both', orientation='horizontal')
hc_map.set_label(unit_map, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


n_af_m.contourf(lon_rho, lat_rho, cg_with_after_map*1e3, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

n_af_m.coastlines()
n_af_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=5)
n_af_m.patch.set_edgecolor('black')
gl = n_af_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
c1 = n_af_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = n_af_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = n_af_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = n_af_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

# Use proxy artists to create legend entries
legend_lines = [Line2D([0], [0], linestyle='dotted', linewidth=1, color='brown'),
				Line2D([0], [0], linestyle='dotted', linewidth=1, color='grey'),
				Line2D([0], [0], linestyle='dashed', linewidth=1, color='k'),
				Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')]

labels = ['50 m', '200 m', '1000 m', '2000 m']

fig.legend(
	legend_lines,
	labels,
	title='Isobaths',
	fontsize='small',
	title_fontsize='small',
	loc='center',
	bbox_to_anchor=(0.08 , 0.4)
)


###############---> CG points time
sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3, 0:3], hspace=0.15)

ax_top = plt.subplot(sub_gs[0, 0])    # Top slice: No Wind
ax_bottom = plt.subplot(sub_gs[1, 0]) # Bottom slice: With Wind

# --- Define common layout parameters ---
t1_start = pd.to_datetime("2004-03-16")
t2_start = pd.to_datetime("2004-03-24")
t3_start = pd.to_datetime("2004-03-31")
t3_end   = pd.to_datetime("2004-04-06")

mid_t1 = t1_start + (t2_start - t1_start) / 2
mid_t2 = t2_start + (t3_start - t2_start) / 2
mid_t3 = t3_start + (t3_end - t3_start) / 2
label_y_pos = -1.8

# ==========================================
# PANEL 1: TOP (No Wind)
# ==========================================
ax_top.text(0.01, 0.95, '(g) No Winds', transform=ax_top.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

ax_top.plot(cg_no_north['ocean_time'][3:], cg_no_north[3:]*1e4, label='North', linestyle=':', color='green')
ax_top.plot(cg_no_central['ocean_time'][3:], cg_no_central[3:]*1e4, label='Central', linestyle='--', color='orange')
ax_top.plot(cg_no_south['ocean_time'][3:], cg_no_south[3:]*1e4, label='South', linestyle='-', color='cyan')

# Structural lines & T-labels for Top
ax_top.axvline(t2_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)
ax_top.axvline(t3_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)
ax_top.text(mid_t1, label_y_pos, 'T1', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
ax_top.text(mid_t2, label_y_pos, 'T2', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
ax_top.text(mid_t3, label_y_pos, 'T3', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')

ax_top.legend(loc=1, fontsize=9)
add_tidal_markers(ax_top)
ax_top.set_ylim(-2, 2)
ax_top.set_ylabel(r'$\Pi$ $\times 10^{-4}$ W m$^{-2}$', fontsize=10)

# Hide X-axis dates for the top plot so they don't overlap with the bottom plot
ax_top.tick_params(labelbottom=False)


# ==========================================
# PANEL 2: BOTTOM (With Wind)
# ==========================================
ax_bottom.text(0.01, 0.95, '(h) With Winds', transform=ax_bottom.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')

# (Replace "cg_with_..." with your actual variable names for the wind data)
ax_bottom.plot(cg_with_north['ocean_time'][3:51], cg_with_north[3:51]*1e4, linestyle=':', color='green')
ax_bottom.plot(cg_with_central['ocean_time'][3:51], cg_with_central[3:51]*1e4, linestyle='--', color='orange')
ax_bottom.plot(cg_with_south['ocean_time'][3:51], cg_with_south[3:51]*1e4, linestyle='-', color='cyan')

# Structural lines & T-labels for Bottom
ax_bottom.axvline(t2_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)
ax_bottom.axvline(t3_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)
ax_bottom.text(mid_t1, label_y_pos, 'T1', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
ax_bottom.text(mid_t2, label_y_pos, 'T2', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
ax_bottom.text(mid_t3, label_y_pos, 'T3', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')

add_tidal_markers(ax_bottom)
ax_bottom.set_ylim(-2, 2)
ax_bottom.set_ylabel(r'$\Pi$ $\times 10^{-4}$ W m$^{-2}$', fontsize=10)

# Apply time formatting ONLY to the bottom axis
format_time_axis(ax_bottom)
ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=4))
ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))


###############---> Budget

# Populating lists with your variables
cg_no_shelf_data = [cg_no_bf_shelf, cg_no_dr_shelf, cg_no_af_shelf]
cg_no_slope_data = [cg_no_bf_slope, cg_no_dr_slope, cg_no_af_slope]

cg_with_shelf_data = [cg_with_bf_shelf, cg_with_dr_shelf, cg_with_af_shelf]
cg_with_slope_data = [cg_with_bf_slope, cg_with_dr_slope, cg_with_af_slope]

# --- Data Setup ---
scenarios = ['T3', 'T2', 'T1']  

y = np.arange(len(scenarios))
# --- Controlled Bar Spacing ---
height = 0.18      # Thickness of each individual bar
inner_gap = 0.02   # Micro-distance between bars inside the same scenario group

# Calculating exact offsets from the center coordinate (y)
offset_top = height + inner_gap
offset_bottom = -(height + inner_gap)

colors = {
	'NO_shelf': 'slateblue',    
	'NO_slope': 'orchid',  
	'WITH_shelf': 'k', 
	'WITH_slope': 'grey',

}


# =====================================================================
# --- GRAPH (h): SHELF REGION ---
# =====================================================================
# Allocates the left half of the bottom row (columns 0, 1, 2)
ax_shelf = plt.subplot(gs[3, 3:5])

# Added hatch='////' to give the No Wind bars a distinct physical texture
b1 = ax_shelf.barh(y + offset_top, cg_no_shelf_data, height, 
				   color=colors['NO_shelf'], edgecolor='k', hatch='////', alpha=0.9)
b2 = ax_shelf.barh(y, cg_with_shelf_data, height, 
				   color=colors['WITH_shelf'], edgecolor='k', alpha=0.9)

ax_shelf.set_title('Shelf ($h < 250$m)', fontweight='bold', fontsize=10, pad=8)
ax_shelf.set_yticks(y)
ax_shelf.set_yticklabels(scenarios, fontweight='bold', fontsize=10, rotation=90, va='center')
ax_shelf.set_xlabel('MW', fontweight='bold', fontsize=10)
ax_shelf.grid(axis='x', linestyle=':', alpha=0.4)
ax_shelf.axvline(0, color='black', linewidth=0.8, alpha=0.5)

# Added a clean legend to explicitly define the textured vs solid bars
ax_shelf.legend([b1, b2], [r'No Winds', r'With Winds'], loc=1, fontsize=9, framealpha=0.9)

# Panel label (h)
ax_shelf.text(0.92, 0.5, '(i)', transform=ax_shelf.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')


# =====================================================================
# --- GRAPH (i): DEEP REGION ---
# =====================================================================
# Allocates the right half of the bottom row (columns 3, 4, 5)
ax_slope = plt.subplot(gs[3, 5:8])

# Added hatch='////' to give the No Wind bars a distinct physical texture
b3 = ax_slope.barh(y + offset_top, cg_no_slope_data, height, 
				   color=colors['NO_slope'], edgecolor='k', hatch='////', alpha=0.9)
b4 = ax_slope.barh(y, cg_with_slope_data, height, 
				   color=colors['WITH_slope'], edgecolor='k', alpha=0.9)

ax_slope.set_title('Deep ($250$m–$3500$m)', fontweight='bold', fontsize=11, pad=8)
ax_slope.set_yticks(y)
ax_slope.set_yticklabels(scenarios, fontweight='bold', fontsize=10, rotation=90, va='center')
ax_slope.set_xlabel('MW', fontweight='bold', fontsize=10)
ax_slope.grid(axis='x', linestyle=':', alpha=0.4)
ax_slope.axvline(0, color='black', linewidth=0.8, alpha=0.5)

# Added a clean legend here as well
ax_slope.legend([b3, b4], [r'No Winds', r'With Winds'], loc=1, fontsize=9, framealpha=0.9)

# Panel label (i)
ax_slope.text(0.95, 0.1, '(j)', transform=ax_slope.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')


# =====================================================================
# --- DYNAMIC HORIZONTAL BAR LABELING FUNCTION ---
# =====================================================================
def label_bars_horizontal(rects, ax, label_template):
	for rect in rects:
		width = rect.get_width()
		
		if width >= 0:
			ha_dir = 'left'
			x_offset = 5
		else:
			ha_dir = 'right'
			x_offset = -5
			
		# The output format is now clean: " [Symbol]: [Value]"
		text_str = f" {label_template}: {width:.1f}"
		
		ax.annotate(text_str,
					xy=(width, rect.get_y() + rect.get_height() / 2),
					xytext=(x_offset, 0),
					textcoords="offset points",
					ha=ha_dir, va='center', fontsize=8, fontweight='bold')

# --- Updated Calls: Stripped out "Winds" text, leaving only the Pi variables ---
label_bars_horizontal(b1, ax_shelf, r'$\Pi$')
label_bars_horizontal(b2, ax_shelf, r'$\Pi$')

label_bars_horizontal(b3, ax_slope, r'$\Pi$')
label_bars_horizontal(b4, ax_slope, r'$\Pi$')


# Expand X-limits dynamically to protect label text boundary clipping
for ax in [ax_shelf, ax_slope]:
	xmin, xmax = ax.get_xlim()
	ax.set_xlim(xmin * 1.30 if xmin < 0 else xmin, xmax * 1.20 if xmax > 0 else xmax)

plt.savefig('cg_periods.png', dpi = 300)


###### s









