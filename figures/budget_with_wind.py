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

####-----> Importing TEC files <----#####
tec_path = "/Users/piero/arian/data1/IT_outs/tec/"

tec_f  = sorted(glob.glob(tec_path + 'tec_slice_*.nc'), key=natural_keys)
tec = xr.open_mfdataset(tec_f, combine='nested', concat_dim='eta_rho', parallel=True).sel(eta_rho=slice(0,1280))

#---> map
tec_before_map = -1*tec.topographic_energy_conversion.sel(ocean_time=slice(before_start, before_end)).sum(dim='mode').mean(dim='ocean_time').compute()
tec_during_map = -1*tec.topographic_energy_conversion.sel(ocean_time=slice(during_start, during_end)).sum(dim='mode').mean(dim='ocean_time').compute()
tec_after_map = -1*tec.topographic_energy_conversion.sel(ocean_time=slice(after_start, after_end)).sum(dim='mode').mean(dim='ocean_time').compute()




####-----> Importing EF files <----#####
ef_path = "/Users/piero/arian/data1/IT_outs/flux/"

ef_f  = sorted(glob.glob(ef_path + 'fbc_slice_*.nc'), key=natural_keys)
ef = xr.open_mfdataset(ef_f, combine='nested', concat_dim='eta_rho', parallel=True).sel(eta_rho=slice(0,1280))

div_roll =  ef.div_Fbc.rolling(ocean_time=25, center=True, min_periods=1).mean().sel(ocean_time=tec.ocean_time, method='nearest')
efx_roll = ef.Fx.rolling(ocean_time=25, center=True, min_periods=1).mean().sel(ocean_time=tec.ocean_time, method='nearest')
efy_roll = ef.Fy.rolling(ocean_time=25, center=True, min_periods=1).mean().sel(ocean_time=tec.ocean_time, method='nearest')

# --- Before Scenario ---
div_bf_lazy = div_roll.sel(ocean_time=slice(before_start, before_end)).isel(mode=0).mean(dim='ocean_time')
Fx_bf_lazy  = efx_roll.sel(ocean_time=slice(before_start, before_end)).isel(mode=0).sum(dim='s_rho').mean(dim='ocean_time')
Fy_bf_lazy  = efy_roll.sel(ocean_time=slice(before_start, before_end)).isel(mode=0).sum(dim='s_rho').mean(dim='ocean_time')

# --- During Scenario ---
div_dr_lazy = div_roll.sel(ocean_time=slice(during_start, during_end)).isel(mode=0).mean(dim='ocean_time')
Fx_dr_lazy  = efx_roll.sel(ocean_time=slice(during_start, during_end)).isel(mode=0).sum(dim='s_rho').mean(dim='ocean_time')
Fy_dr_lazy  = efy_roll.sel(ocean_time=slice(during_start, during_end)).isel(mode=0).sum(dim='s_rho').mean(dim='ocean_time')

# --- After Scenario ---
div_af_lazy = div_roll.sel(ocean_time=slice(after_start, after_end)).isel(mode=0).mean(dim='ocean_time')
Fx_af_lazy  = efx_roll.sel(ocean_time=slice(after_start, after_end)).isel(mode=0).sum(dim='s_rho').mean(dim='ocean_time')
Fy_af_lazy  = efy_roll.sel(ocean_time=slice(after_start, after_end)).isel(mode=0).sum(dim='s_rho').mean(dim='ocean_time')

print("Executing all maps in parallel sharing Dask optimizations...")

(
	div_before_map, Fx_before_map, Fy_before_map,
	div_during_map, Fx_during_map, Fy_during_map,
	div_after_map,  Fx_after_map,  Fy_after_map
) = dask.compute(
	div_bf_lazy, Fx_bf_lazy, Fy_bf_lazy,
	div_dr_lazy, Fx_dr_lazy, Fy_dr_lazy,
	div_af_lazy, Fx_af_lazy, Fy_af_lazy
)

print("All maps successfully computed!")


lon_div = div_after_map.lon_rho.compute()
lat_div = div_after_map.lat_rho.compute()



threshold = 5e-2

# xr.where(condition, x, y) keeps x where condition is true, otherwise replaces with y (np.nan)
div_before_masked =  xr.where(np.abs(div_before_map) <= threshold, div_before_map, np.nan)
div_during_masked =  xr.where(np.abs(div_during_map) <= threshold, div_during_map, np.nan)
div_after_masked  =  xr.where(np.abs(div_after_map) <= threshold, div_after_map, np.nan)

# =====================================================================
# 2. OPTIONAL: MASK FLUXES WHERE DIVERGENCE ERRS OUT
# =====================================================================
# If the divergence has a gradient error at a specific (eta, xi) pixel, 
# the underlying Fx and Fy fluxes at that spot are likely corrupted too.
# You can reuse the divergence mask condition on them:

mask_before = np.abs(div_before_map) <= threshold
mask_during = np.abs(div_during_map) <= threshold
mask_after  = np.abs(div_after_map)  <= threshold

Fx_before_masked = xr.where(mask_before, Fx_before_map, np.nan)
Fx_during_masked = xr.where(mask_during, Fx_during_map, np.nan)
Fx_after_masked  =xr.where(mask_after,  Fx_after_map,  np.nan)

Fy_before_masked = xr.where(mask_before, Fy_before_map, np.nan)
Fy_during_masked = xr.where(mask_during, Fy_during_map, np.nan)
Fy_after_masked  = xr.where(mask_after,  Fy_after_map,  np.nan)


skip_x = 25  
skip_y = 45  

# Create the 2D subsampled slices for coordinates and vectors
# (Using the lon/lat matching the dimension shapes of your flux matrices)
lon_q = lon_div[::skip_y, ::skip_x]
lat_q = lat_div[::skip_y, ::skip_x]

# Define quiver scaling properties (tune 'scale' up if arrows are too long)
q_scale = 25000  
q_color = 'grey'
q_width = 0.003


EF_before = np.sqrt(Fx_before_masked**2 + Fy_before_masked**2)/1e3 #kw/m
EF_during =np.sqrt(Fx_during_masked**2 + Fy_during_masked**2)/1e3 #kw/m
EF_after = np.sqrt(Fx_after_masked**2 + Fy_after_masked**2)/1e3 #kw/m


##############################################
####-----> Importing Alpha files <----#####
alpha_path = "/Users/piero/arian/data1/IT_outs/alpha/"

alpha_f  = sorted(glob.glob(alpha_path + 'alpha_slice_*.nc'), key=natural_keys)
alpha = xr.open_mfdataset(alpha_f, combine='nested', concat_dim='eta_rho', parallel=True).isel(eta_rho=tec.eta_rho.values).rolling(ocean_time=25, center=True, min_periods=1).mean()
alpha_w = alpha.sel(ocean_time=tec.ocean_time, method='nearest')

lat_rho = alpha_w.lat_rho.compute()
lon_rho = alpha_w.lon_rho.compute()

######----> making some summations
import xroms
ds1 = xr.open_dataset('/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc', chunks={'ocean_time': 1, 's_rho': -1, 'eta_rho': 'auto', 'xi_rho': 'auto'})
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

# =====================================================================
# --- 3. C (TEC) Calculations ---
# =====================================================================

# Ensure we don't have negative TEC values
tec_bf_clean = tec_before_map.where(tec_before_map > 0, 0)
tec_dr_clean = tec_during_map.where(tec_during_map > 0, 0)
tec_af_clean = tec_after_map.where(tec_after_map > 0, 0)

# --- Total C ---
c_bf_val = float((tec_bf_clean * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
c_dr_val = float((tec_dr_clean * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
c_af_val = float((tec_af_clean * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)

# --- Shelf C (h < 200m) ---
c_bf_shelf = float((tec_bf_clean.where(mask_shelf) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
c_dr_shelf = float((tec_dr_clean.where(mask_shelf) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
c_af_shelf = float((tec_af_clean.where(mask_shelf) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)

# --- Slope C (200m <= h <= 2500m) ---
c_bf_slope = float((tec_bf_clean.where(mask_slope) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
c_dr_slope = float((tec_dr_clean.where(mask_slope) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)
c_af_slope = float((tec_af_clean.where(mask_slope) * dA).sum(dim=['eta_rho', 'xi_rho']).compute() / 1e6)


print(f"{'Scenario':<10} | {'Total C':<10} | {'Shelf C':<10} | {'Slope C':<10}")
print(f"Before     | {c_bf_val:.2f} | {c_bf_shelf:.2f} | {c_bf_slope:.2f}")
print(f"During     | {c_dr_val:.2f} | {c_dr_shelf:.2f} | {c_dr_slope:.2f}")
print(f"After      | {c_af_val:.2f} | {c_af_shelf:.2f} | {c_af_slope:.2f}")

# =====================================================================
# --- 4. DEF (Divergence/Flux) Calculations ---
# =====================================================================

# --- Total DEF ---
def_bf_val = float((np.abs(div_before_masked) * dA).sum().compute() / 1e6)
def_dr_val = float((np.abs(div_during_masked) * dA).sum().compute() / 1e6)
def_af_val = float((np.abs(div_after_masked) * dA).sum().compute() / 1e6)

# --- Shelf DEF (h < 200m) ---
def_bf_shelf = float((np.abs(div_before_masked).where(mask_shelf) * dA).sum().compute() / 1e6)
def_dr_shelf = float((np.abs(div_during_masked).where(mask_shelf) * dA).sum().compute() / 1e6)
def_af_shelf = float((np.abs(div_after_masked).where(mask_shelf) * dA).sum().compute() / 1e6)

# --- Slope DEF (200m <= h <= 2500m) ---
def_bf_slope = float((np.abs(div_before_masked).where(mask_slope) * dA).sum().compute() / 1e6)
def_dr_slope = float((np.abs(div_during_masked).where(mask_slope) * dA).sum().compute() / 1e6)
def_af_slope = float((np.abs(div_after_masked).where(mask_slope) * dA).sum().compute() / 1e6)




print(f"{'Scenario':<10} | {'Total D EF':<10} | {'Shelf DEF':<10} | {'Slope DEF':<10}")
print(f"Before     | {def_bf_val:.2f} | {def_bf_shelf:.2f} | {def_bf_slope:.2f}")
print(f"During     | {def_dr_val:.2f} | {def_dr_shelf:.2f} | {def_dr_slope:.2f}")
print(f"After      | {def_af_val:.2f} | {def_af_shelf:.2f} | {def_af_slope:.2f}")



# =====================================================================
# --- 5. Graphs per time ---
# =====================================================================
#---> per time per region
dA_south = ds_fix.dA.isel(eta_rho=south_slice)
total_area_south = dA_south.sum(dim=['eta_rho', 'xi_rho']).compute()
dA_central = ds_fix.dA.isel(eta_rho=central_slice)
total_area_central = dA_central.sum(dim=['eta_rho', 'xi_rho']).compute()
dA_north = ds_fix.dA.isel(eta_rho=north_slice)
total_area_north = dA_north.sum(dim=['eta_rho', 'xi_rho']).compute()

tec_time = (-1*tec['topographic_energy_conversion']).sum(dim='mode')
tec_pos = tec_time.where(tec_time > 0, 0)

C_south = ((tec_pos.isel(eta_rho=south_slice) * dA_south).sum(dim=['eta_rho', 'xi_rho'])).compute()/total_area_south

C_central = ((tec_pos.isel(eta_rho=central_slice) * dA_central).sum(dim=['eta_rho', 'xi_rho'])).compute() /total_area_central

C_north = ((tec_pos.isel(eta_rho=north_slice) * dA_north).sum(dim=['eta_rho', 'xi_rho'])).compute() /total_area_north


####-----> Calculating the dissipation <----#####


# --- Shelf C (h < 200m) ---
dbc_bf_shelf = np.abs(c_bf_shelf - def_bf_shelf)
dbc_dr_shelf = np.abs(c_dr_shelf - def_dr_shelf)
dbc_af_shelf = np.abs(c_af_shelf - def_af_shelf)

# --- Slope C (200m <= h <= 2500m) ---
dbc_bf_slope = np.abs(c_bf_slope - def_bf_slope)
dbc_dr_slope = np.abs(c_dr_slope - def_dr_slope)
dbc_af_slope = np.abs(c_af_slope - def_af_slope)




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
gs.update(left=0.05, right=0.99, wspace=0.4, hspace=0.2, top=0.99, bottom=0.05)

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
unit_map = r'C $\times 10^{-2}$ W m$^{-2}$'

unit_div = r'$Fbc\quad \mathrm{kW m^{-1}}$'

dv_map = r'Dbc $\times 10^{-2}$ W m$^{-2}$'

cmap_map = plt.cm.bwr

vmin = -2
vmax = 2
norm_map = mpl.colors.Normalize(vmin=vmin, vmax=vmax)


###-----> Tec
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


a_bf_m.contourf(lon_rho, lat_rho, tec_before_map*1e2, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

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


a_dr_m.contourf(lon_rho, lat_rho, tec_during_map*1e2, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

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


a_af_m.contourf(lon_rho, lat_rho, tec_after_map*1e2, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

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


###-----> Div
# [0, 0] Before Map
cmap_ef = plt.cm.cubehelix_r
vmin = 0
vmax = 5
norm_ef = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

d_bf_m = plt.subplot(gs[0, 4:8],projection=ccrs.PlateCarree())
d_bf_m.set_aspect('auto')

d_bf_m.text(0.88, 0.9, '(d)', transform=d_bf_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
d_bf_m.set_ylim(bottom=-33, top=-20)
d_bf_m.set_xlim(left=-52, right=-39.2)

ax_map_cb = inset_axes(d_bf_m, 
					   width="60%", 
					   height="3%", 
					   loc='upper left',
					   bbox_to_anchor=(0.08, -0.03, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=d_bf_m.transAxes,
					   borderpad=0)

ax_map_cb.set_facecolor('lightgray')

cb_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_ef, norm=norm_ef, extend='max', orientation='horizontal')
cb_map.set_label(unit_div, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


d_bf_m.contourf(lon_div, lat_div, EF_before, levels=200, cmap=cmap_ef, norm=norm_ef, extend='max', zorder=0,projection=ccrs.PlateCarree())


# --- NEW: QUIVER OVERLAY (BEFORE) ---
Fx_b_q = Fx_before_masked[::skip_y, ::skip_x]
Fy_b_q = Fy_before_masked[::skip_y, ::skip_x]

q1 = d_bf_m.quiver(lon_q, lat_q, Fx_b_q, Fy_b_q, 
				   scale=q_scale, color=q_color, width=q_width,
				   transform=ccrs.PlateCarree(), zorder=4)

# Quiver key/legend for scale reference
#d_bf_m.quiverkey(q1, X=0.85, Y=0.05, U=1.0, label=r'50 $ W m$^{-1}$', labelpos='E', coordinates='axes', fontproperties={'size': 8})


d_bf_m.coastlines()
d_bf_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=5)
d_bf_m.patch.set_edgecolor('black')
gl = d_bf_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
c1 = d_bf_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = d_bf_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = d_bf_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = d_bf_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)



# [1, 0] during Map
d_dr_m = plt.subplot(gs[1, 4:8],projection=ccrs.PlateCarree())
d_dr_m.set_aspect('auto')

d_dr_m.text(0.88, 0.9, '(e)', transform=d_dr_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
d_dr_m.set_ylim(bottom=-33, top=-20)
d_dr_m.set_xlim(left=-52, right=-39.2)

ax_map_cb = inset_axes(d_dr_m, 
					   width="60%", 
					   height="3%", 
					   loc='upper left',
					   bbox_to_anchor=(0.08, -0.03, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=d_dr_m.transAxes,
					   borderpad=0)

ax_map_cb.set_facecolor('lightgray')
hc_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_ef, norm=norm_ef, extend='max', orientation='horizontal')
hc_map.set_label(unit_div, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


d_dr_m.contourf(lon_div, lat_div, EF_during, levels=200, cmap=cmap_ef, norm=norm_ef, extend='max', zorder=0,projection=ccrs.PlateCarree())

Fx_d_q = Fx_during_masked[::skip_y, ::skip_x]
Fy_d_q = Fy_during_masked[::skip_y, ::skip_x]

q2 = d_dr_m.quiver(lon_q, lat_q, Fx_d_q, Fy_d_q, 
				   scale=q_scale, color=q_color, width=q_width,
				   transform=ccrs.PlateCarree(), zorder=4)

d_dr_m.quiverkey(q2, X=0.7, Y=0.05, U=1000.0, label=r'1 kW m$^{-1}$', labelpos='E', coordinates='axes', fontproperties={'size': 12})



d_dr_m.coastlines()
d_dr_m.add_feature(cartopy.feature.LAND, facecolor='lightgray' , zorder=5)
d_dr_m.patch.set_edgecolor('black')
gl = d_dr_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
c1 = d_dr_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = d_dr_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = d_dr_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = d_dr_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)




# [2, 0] after Map
d_af_m = plt.subplot(gs[2, 4:8],projection=ccrs.PlateCarree())
d_af_m.set_aspect('auto')

d_af_m.text(0.88, 0.9, '(f)', transform=d_af_m.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
d_af_m.set_ylim(bottom=-33, top=-20)
d_af_m.set_xlim(left=-52, right=-39.2)

ax_map_cb = inset_axes(d_af_m, 
					   width="60%", 
					   height="3%", 
					   loc='upper left',
					   bbox_to_anchor=(0.08, -0.03, 1, 1), # Adjust 0.08 to move right
					   bbox_transform=d_af_m.transAxes,
					   borderpad=0)

ax_map_cb.set_facecolor('lightgray')
hc_map = mpl.colorbar.ColorbarBase(ax_map_cb, cmap=cmap_ef, norm=norm_ef, extend='max', orientation='horizontal')
hc_map.set_label(unit_div, size=10)
ax_map_cb.xaxis.set_ticks_position('bottom')
ax_map_cb.tick_params(axis='x', labelsize='small', rotation=15)


d_af_m.contourf(lon_div, lat_div, EF_after, levels=200, cmap=cmap_ef, norm=norm_ef, extend='max', zorder=0,projection=ccrs.PlateCarree())

Fx_a_q = Fx_after_masked[::skip_y, ::skip_x]
Fy_a_q = Fy_after_masked[::skip_y, ::skip_x]

q3 = d_af_m.quiver(lon_q, lat_q, Fx_a_q, Fy_a_q, 
				   scale=q_scale, color=q_color, width=q_width,
				   transform=ccrs.PlateCarree(), zorder=4)





d_af_m.coastlines()
d_af_m.add_feature(cartopy.feature.LAND, facecolor='lightgray', zorder=5)
d_af_m.patch.set_edgecolor('black')
gl = d_af_m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.3, color='gray', alpha=0.7, linestyle='--', zorder=11)
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
c1 = d_af_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = d_af_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = d_af_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = d_af_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

# Use proxy artists to create legend entries
legend_lines = [Line2D([0], [0], linestyle='dotted', linewidth=1, color='brown'),
				Line2D([0], [0], linestyle='dotted', linewidth=1, color='grey'),
				Line2D([0], [0], linestyle='dashed', linewidth=1, color='k'),
				Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')]

labels = ['50 m', '200 m', '1000 m', '2000 m']





###############---> tec points time
pcx = plt.subplot(gs[3, 0:3])

pcx.text(0.01, 0.95, '(g)', transform=pcx.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
#pcx.set_ylim(bottom=-30, top=30)

pcx.plot(C_south['ocean_time'][2:], C_north[2:]*1e4, label='North', linestyle=':', color='green')
pcx.plot(C_south['ocean_time'][2:], C_central[2:]*1e4, label='Central', linestyle='--', color='orange')
pcx.plot(C_south['ocean_time'][2:], C_south[2:]*1e4, label='South', linestyle='-', color='cyan')

# --- Clean Bounds Layout (Vertical Lines & Bottom Labels) ---
# Define the date boundaries
t1_start = pd.to_datetime("2004-03-16")
t2_start = pd.to_datetime("2004-03-24")
t3_start = pd.to_datetime("2004-03-31")
t3_end   = pd.to_datetime("2004-04-06")

# Draw structural boundary lines between periods
pcx.axvline(t2_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)
pcx.axvline(t3_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)

# Calculate temporal midpoints for label placement
mid_t1 = t1_start + (t2_start - t1_start) / 2
mid_t2 = t2_start + (t3_start - t2_start) / 2
mid_t3 = t3_start + (t3_end - t3_start) / 2

# Place text blocks near the bottom x-axis (y = 0.5)
label_y_pos = 0.5 
pcx.text(mid_t1, label_y_pos, 'T1', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
pcx.text(mid_t2, label_y_pos, 'T2', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
pcx.text(mid_t3, label_y_pos, 'T3', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')

for i, (start, end) in enumerate(during):
	pcx.axvspan(pd.to_datetime(start),
				pd.to_datetime(end),
				color='red',
				alpha=0.2,
				label='Hurricane' if i == 0 else None)

pcx.legend(loc=1,fontsize=9)
add_tidal_markers(pcx)
pcx.set_ylim(0,10)
format_time_axis(pcx)
pcx.xaxis.set_major_locator(mdates.DayLocator(interval=4))
pcx.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
pcx.set_ylabel(r'C $\times 10^{-4}$ W m$^{-2}$',fontsize = 10)


###############---> Budget

# Populating lists with your variables
c_shelf_data = [c_af_shelf, c_dr_shelf, c_bf_shelf]
c_slope_data = [c_af_slope, c_dr_slope, c_bf_slope]

def_shelf_data = [def_af_shelf, def_dr_shelf, def_bf_shelf]
def_slope_data = [def_af_slope, def_dr_slope, def_bf_slope]

dbc_shelf_data = [dbc_af_shelf, dbc_dr_shelf, dbc_bf_shelf]
dbc_slope_data = [dbc_af_slope, dbc_dr_slope, dbc_bf_slope]
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
	'C_shelf': 'slateblue',    
	'C_slope': 'orchid',  
	'DEF_shelf': 'k', 
	'DEF_slope': 'grey',
	'DBC_shelf': 'tomato',  # Color for shelf dissipation
	'DBC_slope': 'crimson'  # Color for slope dissipation
}


# =====================================================================
# --- GRAPH (h): SHELF REGION ---
# =====================================================================
# Allocates the left half of the bottom row (columns 0, 1, 2)
ax_shelf = plt.subplot(gs[3, 3:5])

b1 = ax_shelf.barh(y + offset_top, c_shelf_data, height, 
				   color=colors['C_shelf'], edgecolor='k', alpha=0.9)
b2 = ax_shelf.barh(y, def_shelf_data, height, 
				   color=colors['DEF_shelf'], edgecolor='k', alpha=0.9)
b_dbc_shelf = ax_shelf.barh(y + offset_bottom, dbc_shelf_data, height, 
						   color=colors['DBC_shelf'], edgecolor='k', alpha=0.9)

ax_shelf.set_title('Shelf ($h < 250$m)', fontweight='bold', fontsize=10, pad=8)
ax_shelf.set_yticks(y)
ax_shelf.set_yticklabels(scenarios, fontweight='bold', fontsize=10, rotation=90, va='center')
ax_shelf.set_xlabel('MW', fontweight='bold', fontsize=10)
ax_shelf.grid(axis='x', linestyle=':', alpha=0.4)
ax_shelf.axvline(0, color='black', linewidth=0.8, alpha=0.5)

# Panel label (h)
ax_shelf.text(0.9, 0.1, '(h)', transform=ax_shelf.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')


# =====================================================================
# --- GRAPH (i): DEEP REGION ---
# =====================================================================
# Allocates the right half of the bottom row (columns 3, 4, 5)
ax_slope = plt.subplot(gs[3, 5:8])

b3 = ax_slope.barh(y + offset_top, c_slope_data, height, 
				   color=colors['C_slope'], edgecolor='k', alpha=0.9)
b4 = ax_slope.barh(y, def_slope_data, height, 
				   color=colors['DEF_slope'], edgecolor='k', alpha=0.9)
b_dbc_slope = ax_slope.barh(y + offset_bottom, dbc_slope_data, height, 
						   color=colors['DBC_slope'], edgecolor='k', alpha=0.9)
ax_slope.set_title('Deep ($250$m–$3500$m)', fontweight='bold', fontsize=11, pad=8)
ax_slope.set_yticks(y)
ax_slope.set_yticklabels(scenarios, fontweight='bold', fontsize=10, rotation=90, va='center')
ax_slope.set_xlabel('MW', fontweight='bold', fontsize=10)
ax_slope.grid(axis='x', linestyle=':', alpha=0.4)
ax_slope.axvline(0, color='black', linewidth=0.8, alpha=0.5)

# Panel label (i)
ax_slope.text(0.9, 0.95, '(i)', transform=ax_slope.transAxes, fontsize=11, fontweight='bold', va='top', ha='left')


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
			
		text_str = f" {label_template}: {width:.1f}"
		
		ax.annotate(text_str,
					xy=(width, rect.get_y() + rect.get_height() / 2),
					xytext=(x_offset, 0),
					textcoords="offset points",
					ha=ha_dir, va='center', fontsize=8, fontweight='bold')

# Apply labels across all three bars for both graphs
label_bars_horizontal(b1, ax_shelf, r'$\mathrm{C}$')
label_bars_horizontal(b2, ax_shelf, r'$\nabla \cdot \mathbf{F_{bc}}$')
label_bars_horizontal(b_dbc_shelf, ax_shelf, r'$\mathrm{D_{bc}}$')

label_bars_horizontal(b3, ax_slope, r'$\mathrm{C}$')
label_bars_horizontal(b4, ax_slope, r'$\nabla \cdot \mathbf{F_{bc}}$')
label_bars_horizontal(b_dbc_slope, ax_slope, r'$\mathrm{D_{bc}}$')

# Expand X-limits dynamically to protect label text boundary clipping
for ax in [ax_shelf, ax_slope]:
	xmin, xmax = ax.get_xlim()
	ax.set_xlim(xmin * 1.30 if xmin < 0 else xmin, xmax * 1.20 if xmax > 0 else xmax)

plt.savefig('fig_with_winds.png', dpi = 300)










