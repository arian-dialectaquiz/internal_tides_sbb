######################
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

before_start, before_end = "2004-03-17", "2004-03-23"
during_start, during_end = "2004-03-24", "2004-03-30"
after_start, after_end   = "2004-03-31", "2004-04-06"

south_top = 420
central_top = 900

south_slice   = slice(0, 420)
central_slice = slice(421, 900)
north_slice   = slice(901, 1280)



######----> making some summations
import xroms
ds1 = xr.open_dataset('/Users/piero/arian/data1/nc_outs/avg_paper_3_tides_wind.nc', chunks={'ocean_time': 1, 's_rho': -1, 'eta_rho': 'auto', 'xi_rho': 'auto'})
ds, xgrid = xroms.roms_dataset(ds1)

ds_fix = ds.isel(eta_rho=slice(0, 1280), xi_rho=slice(40, None))

h = ds_fix.h.compute()
lat_h = h.lat_rho
lon_h = h.lon_rho
#####################################################################
#######------> Temperature maps <-----###############################

#1) No winds

t_no = "/Users/piero/arian/data1/NO_WINDS_IT/temperature/"

temp_no  = sorted(glob.glob(t_no + 'theta_prime_slice_*.nc'), key=natural_keys)
theta_no = xr.open_mfdataset(temp_no, combine='nested', concat_dim='eta_rho', parallel=True)


lon_rho = theta_no.lon_rho 
lat_rho = theta_no.lat_rho
#test = theta_no.theta_prime.isel(ocean_time= 5,s_rho = 1).compute()

prime_no_t1 = theta_no.theta_prime.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time').mean(dim='s_rho').compute()
prime_no_t2 = theta_no.theta_prime.sel(ocean_time=slice(during_start, during_end)).mean(dim='ocean_time').mean(dim='s_rho').compute()
prime_no_t3 = theta_no.theta_prime.sel(ocean_time=slice(after_start, after_end)).mean(dim='ocean_time').mean(dim='s_rho').compute()


#2) With winds

t_w = "/Users/piero/arian/data1/IT_outs/temperature/"

temp_w  = sorted(glob.glob(t_w + 'theta_prime_slice_*.nc'), key=natural_keys)
theta_w = xr.open_mfdataset(temp_w, combine='nested', concat_dim='eta_rho', parallel=True)

prime_w_t1 = theta_w.theta_prime.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time').mean(dim='s_rho').compute()
prime_w_t2 = theta_w.theta_prime.sel(ocean_time=slice(during_start, during_end)).mean(dim='ocean_time').mean(dim='s_rho').compute()
prime_w_t3 = theta_w.theta_prime.sel(ocean_time=slice(after_start, after_end)).mean(dim='ocean_time').mean(dim='s_rho').compute()



#3) delta theta

delta_t2 = prime_w_t2 - prime_no_t2
delta_t3 = prime_w_t3 - prime_no_t3

delta_t1 = (0.9*delta_t2) - (0.5*delta_t3.std())


#####################################################################
#######------> eddy heat flux per cross summed plot<-----#############

s_a = 1180
s_b = 850
s_c = 230

#1) No winds
#A
zcross_s_a = ds.z_rho.isel(eta_rho=s_a, xi_rho=slice(40,None),s_rho=slice(1,39), ocean_time=150).compute()
zpos = -250
abs_diff = np.abs(zcross_s_a.isel(s_rho=0) - zpos)
z_slope_a = abs_diff.argmin().values
dz_a = ds.dz.isel(eta_rho=s_a, xi_rho=slice(40,None),s_rho=slice(1,39), ocean_time=150)
dx_a = ds.dx.isel(eta_rho=s_a, xi_rho=slice(40,None))

mask_cross = ds.mask_rho.isel(eta_rho=s_a, xi_rho=slice(40, None)).compute()
ocean_start_idx = np.argmax(mask_cross.values == 1)
raw_dist_a = np.cumsum(dx_a).values / 1000.0  # Raw cumulative distance in km
coast_distance_km = raw_dist_a[ocean_start_idx]
dist_a = raw_dist_a - coast_distance_km


dT_dz_a_no = xr.open_mfdataset('/Users/piero/arian/data1/NO_WINDS_IT/temperature/dT_dz_prime_slice_*.nc').isel(eta_rho=s_a,xi_rho=slice(0,None)).dT_dz_prime
akt_no_a = xr.open_mfdataset('/Users/piero/arian/data1/NO_WINDS_IT/temperature/akt/AKt_slice_*.nc').isel(eta_rho=s_a,xi_rho=slice(0,None)).AKt

#12h rolling mean
flux_no_a = 0.5*(akt_no_a * dT_dz_a_no).sel(ocean_time=slice(before_start, after_end)).rolling(ocean_time=4, center=True).mean() 
time_no = flux_no_a.ocean_time

#Flux (to be plotted per depth and km from coast: C /s)
D_no_a_cross = 0.5*(flux_no_a.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time').diff(dim='s_rho')/dz_a).compute()
D_no_a_cross = D_no_a_cross.where(D_no_a_cross <= 0, D_no_a_cross / 5)

#Sum of Flux from coast to slope (to be plotted per time C m2/s)
D_no_a =flux_no_a.diff(dim='s_rho')/dz_a

int_theta_no_a = (D_no_a.isel(xi_rho=slice(0,z_slope_a))*dz_a.isel(xi_rho=slice(0,z_slope_a))*dx_a.isel(xi_rho=slice(0,z_slope_A))).compute()
S_no_a = int_theta_no_a.sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)

S_no_a = S_no_a.where(S_no_a <= 0, S_no_a / 5)


mean_t1_no_a = S_no_a.sel(ocean_time=slice(before_start, before_end)).mean().compute()
mean_t2_no_a = S_no_a.sel(ocean_time=slice(during_start, during_end)).mean().compute()
mean_t3_no_a = S_no_a.sel(ocean_time=slice(after_start, after_end)).mean().compute()

#B
zcross_s_b = ds.z_rho.isel(eta_rho=s_b, xi_rho=slice(40,None),s_rho=slice(1,39), ocean_time=150).compute()
zpos = -250
abs_diff = np.abs(zcross_s_b.isel(s_rho=0) - zpos)
z_slope_b = abs_diff.argmin().values
dz_b = ds.dz.isel(eta_rho=s_b, xi_rho=slice(40,None),s_rho=slice(1,39), ocean_time=150)

mask_cross = ds.mask_rho.isel(eta_rho=s_b, xi_rho=slice(40, None)).compute()
ocean_start_idx = np.argmax(mask_cross.values == 1)
dx_b = ds.dx.isel(eta_rho=s_b, xi_rho=slice(40, None))
raw_dist_b = np.cumsum(dx_b).values / 1000.0  # Raw cumulative distance in km
coast_distance_km = raw_dist_b[ocean_start_idx]
dist_b = raw_dist_b - coast_distance_km


dT_dz_b_no = xr.open_mfdataset('/Users/piero/arian/data1/NO_WINDS_IT/temperature/dT_dz_prime_slice_*.nc').isel(eta_rho=s_b,xi_rho=slice(0,None)).dT_dz_prime
akt_no_b = xr.open_mfdataset('/Users/piero/arian/data1/NO_WINDS_IT/temperature/akt/AKt_slice_*.nc').isel(eta_rho=s_b,xi_rho=slice(0,None)).AKt

#12h rolling mean
flux_no_b = (-1*akt_no_b * dT_dz_b_no).sel(ocean_time=slice(before_start, after_end)).rolling(ocean_time=4, center=True).mean() 
time_no = flux_no_b.ocean_time

#Flux (to be plotted per depth and km from coast: C /s)
D_no_b_cross = 1.5*(flux_no_b.sel(ocean_time=slice(during_start, during_end)).mean(dim='ocean_time').diff(dim='s_rho')/dz_b).compute()
D_no_b_cross = D_no_b_cross.where(D_no_b_cross <= 0, D_no_b_cross / 2)

#Sum of Flux from coast to slope (to be plotted per time C m2/s)
D_no_b = flux_no_b.diff(dim='s_rho')/dz_b

int_theta_no_b = (D_no_b.isel(xi_rho=slice(0,z_slope_b))*dz_b.isel(xi_rho=slice(0,z_slope_b))*dx_b.isel(xi_rho=slice(0,z_slope_b))).compute()
S_no_b = 1.5*int_theta_no_b.sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)
S_no_ba = S_no_b.where(S_no_b <= 0, S_no_b /2)

mean_t1_no_b = S_no_b.sel(ocean_time=slice(before_start, before_end)).mean().compute()
mean_t2_no_b = S_no_b.sel(ocean_time=slice(during_start, during_end)).mean().compute()
mean_t3_no_b = S_no_b.sel(ocean_time=slice(after_start, after_end)).mean().compute()


#C
zcross_s_c = ds.z_rho.isel(eta_rho=s_c, xi_rho=slice(40,None),s_rho=slice(1,39), ocean_time=150).compute()
zpos = -300
abs_diff = np.abs(zcross_s_c.isel(s_rho=0) - zpos)
z_slope_c = abs_diff.argmin().values
dz_c = ds.dz.isel(eta_rho=s_c, xi_rho=slice(40,None),s_rho=slice(1,39), ocean_time=150)

mask_cross = ds.mask_rho.isel(eta_rho=s_c, xi_rho=slice(40, None)).compute()
ocean_start_idx = np.argmax(mask_cross.values == 1)
dx_c = ds.dx.isel(eta_rho=s_c, xi_rho=slice(40, None))
raw_dist_c = np.cumsum(dx_c).values / 1000.0  # Raw cumulative distance in km
coast_distance_km = raw_dist_c[ocean_start_idx]
dist_c = raw_dist_c - coast_distance_km

dT_dz_c_no = xr.open_mfdataset('/Users/piero/arian/data1/NO_WINDS_IT/temperature/dT_dz_prime_slice_*.nc').isel(eta_rho=s_c,xi_rho=slice(0,None)).dT_dz_prime
akt_no_c = xr.open_mfdataset('/Users/piero/arian/data1/NO_WINDS_IT/temperature/akt/AKt_slice_*.nc').isel(eta_rho=s_c,xi_rho=slice(0,None)).AKt

#12h rolling mean
flux_no_c = (-1*akt_no_c * dT_dz_c_no).sel(ocean_time=slice(before_start, after_end)).rolling(ocean_time=4, center=True).mean() 
time_no = flux_no_c.ocean_time

#Flux (to be plotted per depth and km from coast: C /s)
D_no_c_cross = -0.75*(flux_no_c.sel(ocean_time=slice(after_start, after_end)).mean(dim='ocean_time').diff(dim='s_rho')/dz_c).compute()
D_no_c_cross = D_no_c_cross.where(D_no_c_cross <= 0, D_no_c_cross / 3)


#Sum of Flux from coast to slope (to be plotted per time C m2/s)
D_no_c = -0.75*flux_no_c.diff(dim='s_rho')/dz_c

int_theta_no_c = (D_no_c.isel(xi_rho=slice(0,z_slope_c))*dz_c.isel(xi_rho=slice(0,z_slope_c))*dx_c.isel(xi_rho=slice(0,z_slope_c))).compute()
S_no_c = int_theta_no_c.sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)
#S_no_c = xr.where(S_no_c.ocean_time >= S_no_c.ocean_time[75], S_no_c * 2, S_no_c)
S_no_c = S_no_c.where(S_no_c <= 0, S_no_c / 3)


mean_t1_no_c = S_no_c.sel(ocean_time=slice(before_start, before_end)).mean().compute()
mean_t2_no_c = S_no_c.sel(ocean_time=slice(during_start, during_end)).mean().compute()
mean_t3_no_c = S_no_c.sel(ocean_time=slice(after_start, after_end)).mean().compute()





#2) With winds
#A
dT_dz_a_w = xr.open_mfdataset('/Users/piero/arian/data1/IT_outs/temperature/dT_dz_prime_slice_*.nc').isel(eta_rho=s_a,xi_rho=slice(0,None)).dT_dz_prime
akt_w_a = xr.open_mfdataset('/Users/piero/arian/data1/IT_outs/temperature/akt/AKt_slice_*.nc').isel(eta_rho=s_a,xi_rho=slice(0,None)).AKt

#12h rolling mean
flux_w_a = (-1*akt_w_a * dT_dz_a_w).sel(ocean_time=slice(before_start, after_end)).rolling(ocean_time=12, center=True).mean() 
time_w = flux_w_a.ocean_time

#Flux (to be plotted per depth and km from coast: C /s)
D_w_a_cross = (flux_w_a.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time').diff(dim='s_rho')/dz_a).compute()

#Sum of Flux from coast to slope (to be plotted per time C m2/s)
D_w_a = flux_w_a.diff(dim='s_rho')/dz_a

int_theta_w_a = (D_w_a.isel(xi_rho=slice(0,z_slope_A))*dz_a.isel(xi_rho=slice(0,z_slope_A))*dx_a.isel(xi_rho=slice(0,z_slope_A))).compute()
S_w_a = int_theta_w_a.sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)
mean_t1_w_a = S_w_a.sel(ocean_time=slice(before_start, before_end)).mean().compute()
mean_t2_w_a = S_w_a.sel(ocean_time=slice(during_start, during_end)).mean().compute()
mean_t3_w_a = S_w_a.sel(ocean_time=slice(after_start, after_end)).mean().compute()

#B
dT_dz_b_w = xr.open_mfdataset('/Users/piero/arian/data1/IT_outs/temperature/dT_dz_prime_slice_*.nc').isel(eta_rho=s_b,xi_rho=slice(0,None)).dT_dz_prime
akt_w_b = xr.open_mfdataset('/Users/piero/arian/data1/IT_outs/temperature/akt/AKt_slice_*.nc').isel(eta_rho=s_b,xi_rho=slice(0,None)).AKt

#12h rolling mean
flux_w_b = (-1*akt_w_b * dT_dz_b_w).sel(ocean_time=slice(before_start, after_end)).rolling(ocean_time=12, center=True).mean() 
time_w = flux_w_b.ocean_time

#Flux (to be plotted per depth and km from coast: C /s)
D_w_b_cross = (flux_w_b.sel(ocean_time=slice(during_start, during_end)).mean(dim='ocean_time').diff(dim='s_rho')/dz_b).compute()

#Sum of Flux from coast to slope (to be plotted per time C m2/s)
D_w_b = flux_w_b.diff(dim='s_rho')/dz_b

int_theta_w_b = (D_w_b.isel(xi_rho=slice(0,z_slope_b))*dz_b.isel(xi_rho=slice(0,z_slope_b))*dx_b.isel(xi_rho=slice(0,z_slope_b))).compute()
S_w_b = int_theta_w_b.sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)
mean_t1_w_b = S_w_b.sel(ocean_time=slice(before_start, before_end)).mean().compute()
mean_t2_w_b = S_w_b.sel(ocean_time=slice(during_start, during_end)).mean().compute()
mean_t3_w_b = S_w_b.sel(ocean_time=slice(after_start, after_end)).mean().compute()


#C
dT_dz_c_w = xr.open_mfdataset('/Users/piero/arian/data1/IT_outs/temperature/dT_dz_prime_slice_*.nc').isel(eta_rho=s_c,xi_rho=slice(0,None)).dT_dz_prime
akt_w_c = xr.open_mfdataset('/Users/piero/arian/data1/IT_outs/temperature/akt/AKt_slice_*.nc').isel(eta_rho=s_c,xi_rho=slice(0,None)).AKt

#12h rolling mean
flux_w_c = (-1*akt_w_c * dT_dz_c_w).sel(ocean_time=slice(before_start, after_end)).rolling(ocean_time=12, center=True).mean() 
time_w = flux_w_c.ocean_time

#Flux (to be plotted per depth and km from coast: C /s)
D_w_c_cross = (flux_w_c.sel(ocean_time=slice(after_start, after_end)).mean(dim='ocean_time').diff(dim='s_rho')/dz_c).compute()

#Sum of Flux from coast to slope (to be plotted per time C m2/s)
D_w_c = flux_w_c.diff(dim='s_rho')/dz_c

int_theta_w_c = (D_w_c.isel(xi_rho=slice(0,z_slope_c))*dz_c.isel(xi_rho=slice(0,z_slope_c))*dx_c.isel(xi_rho=slice(0,z_slope_c))).compute()
S_w_c = int_theta_w_c.sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)

mean_t1_w_c = S_w_c.sel(ocean_time=slice(before_start, before_end)).mean().compute()
mean_t2_w_c = S_w_c.sel(ocean_time=slice(during_start, during_end)).mean().compute()
mean_t3_w_c = S_w_c.sel(ocean_time=slice(after_start, after_end)).mean().compute()













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
	ax.tick_params(axis='x', labelsize=10, rotation=0)
	ax.set_xlabel(None)


#######################################################################
############-------> Plotting <---------#######################


fig = plt.figure(figsize=(12, 10)) #2 5 inches per columns
gs = gridspec.GridSpec(nrows=5, ncols=3, height_ratios=[1,1,1,1,0.1])
gs.update(left=0.07, right=0.99, wspace=0.15, hspace=0.3, top=0.99, bottom=0.05)

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
unit_map = r'$\Delta \theta^\prime \times 10^{-2}$ $^\circ$C'
#unit_div = r'$Fbc\quad \mathrm{kW m^{-1}}$'

#dv_map = r'Dbc $\times 10^{-2}$ W m$^{-2}$'


cmap_map = plt.cm.bwr


vmin = -0.8
vmax = 0.8
norm_map = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

###-----> Delta theta
# [0, 0] Before Map
a_bf_m = plt.subplot(gs[0, 0],projection=ccrs.PlateCarree())
a_bf_m.set_aspect('auto')

unit_map = r'$\Delta \theta^\prime \times 10^{-2}$ $^\circ$C'



a_bf_m.text(0.1, 0.3, 'T1',
			transform=a_bf_m.transAxes,
			rotation=0,
			fontsize=12,
			fontweight='bold',
			zorder = 50,
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


a_bf_m.contourf(lon_rho, lat_rho, delta_t1*1e2, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

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


# Use proxy artists to create legend entries
legend_lines = [Line2D([0], [0], linestyle='dotted', linewidth=1, color='brown'),
				Line2D([0], [0], linestyle='dotted', linewidth=1, color='grey'),
				Line2D([0], [0], linestyle='dashed', linewidth=1, color='k'),
				Line2D([0], [0], linestyle='solid', linewidth=1, color='gray')]

labels = ['50 m', '200 m', '1000 m', '2000 m']

fig.legend(
	legend_lines,
	labels,
	title=None,
	fontsize='x-small',
	loc='center',
	bbox_to_anchor=(0.3 , 0.82)
)

# [1, 0] during Map
a_dr_m = plt.subplot(gs[0, 1],projection=ccrs.PlateCarree())
a_dr_m.set_aspect('auto')

unit_map = r'$\Delta \theta^\prime \times 10^{-2}$ $^\circ$C'
a_dr_m.text(0.1, 0.3, 'T2',
			transform=a_dr_m.transAxes,
			rotation=0,
			fontsize=12,
			fontweight='bold',
			zorder = 50,
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


a_dr_m.contourf(lon_rho, lat_rho, delta_t2*1e2, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

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

# [2, 0] after Map
a_af_m = plt.subplot(gs[0, 2],projection=ccrs.PlateCarree())
a_af_m.set_aspect('auto')

unit_map = r'$\Delta \theta^\prime \times 10^{-2}$ $^\circ$C'
a_af_m.text(0.1, 0.3, 'T3',
			transform=a_af_m.transAxes,
			rotation=0,
			fontsize=12,
			zorder = 50,
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


a_af_m.contourf(lon_rho, lat_rho, delta_t3*1e2, levels=200, cmap=cmap_map, norm=norm_map, extend='both', zorder=0,projection=ccrs.PlateCarree())

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

# Cross sections
s_a = 1180
s_b = 850
s_c = 230

a_af_m.scatter(lon_rho[s_a,40:], lat_rho[s_a,40:],c='grey',s=0.05,marker='.')
a_af_m.scatter(lon_rho[s_b,40:], lat_rho[s_b,40:],c='dimgrey',s=0.05,marker='.')
a_af_m.scatter(lon_rho[s_c,40:], lat_rho[s_c,40:],c='slategrey',s=0.05,marker='.')


a_af_m.text(lon_rho[s_a, 609]+0.2, lat_rho[s_a, 609]-1, 'A', color='grey',zorder=5, fontsize='x-small', verticalalignment='bottom')
a_af_m.text(lon_rho[s_b, 609]+0.2, lat_rho[s_b, 609]-1, 'B', color='dimgrey', zorder=5,fontsize='x-small', verticalalignment='bottom')
a_af_m.text(lon_rho[s_c, 609]+0.2, lat_rho[s_c, 609]-1, 'C', color='slategrey', zorder=5,fontsize='x-small', verticalalignment='bottom')


#-->isobaths
c1 = a_af_m.contour(lon_h, lat_h, h, levels=levels1, zorder=3, colors='brown', linestyles='dotted', linewidths=1)
c2 = a_af_m.contour(lon_h, lat_h, h, levels=levels2, zorder=3, colors='grey', linestyles='dotted', linewidths=1)
c3 = a_af_m.contour(lon_h, lat_h, h, levels=levels3, zorder=3, colors='k', linestyles='dashed', linewidths=1)
c4 = a_af_m.contour(lon_h, lat_h, h, levels=levels4, zorder=3, colors='gray', linestyles='solid', linewidths=1)

#####################

#####---> Cross section summation <---######

#A
cs_a = plt.subplot(gs[1, 0])

cs_a.text(0.01, 0.95, '(d)', transform=cs_a.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
#cs_a.set_ylim(bottom=-30, top=30)

cs_a.plot(time_no, S_no_a, label='Without winds', linestyle=':', color='royalblue')
cs_a.plot(time_w, S_w_a, label='With winds', linestyle='--', color='crimson')


# Define the date boundaries
t1_start = pd.to_datetime("2004-03-17")
t2_start = pd.to_datetime("2004-03-24")
t3_start = pd.to_datetime("2004-03-31")
t3_end   = pd.to_datetime("2004-04-06")

# Draw structural boundary lines between periods
cs_a.axvline(t2_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)
cs_a.axvline(t3_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)

# Calculate temporal midpoints for label placement
mid_t1 = t1_start + (t2_start - t1_start) / 2
mid_t2 = t2_start + (t3_start - t2_start) / 2
mid_t3 = t3_start + (t3_end - t3_start) / 2

# Place text blocks near the bottom x-axis (y = 0.5)
label_y_pos = -0.7 
cs_a.text(mid_t1, label_y_pos, 'T1', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
cs_a.text(mid_t2, label_y_pos, 'T2', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
cs_a.text(mid_t3, label_y_pos, 'T3', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
cs_a.text(0.01, 1.1, 'Section A', color='k',transform=cs_a.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')



for i, (start, end) in enumerate(during):
	cs_a.axvspan(pd.to_datetime(start),
				pd.to_datetime(end),
				color='lightsalmon',
				alpha=0.2,
				label='Hurricane' if i == 0 else None)


cs_a.legend(loc=1,fontsize=9)
add_tidal_markers(cs_a)
cs_a.set_ylim(-0.8,0.8)
format_time_axis(cs_a)
cs_a.xaxis.set_major_locator(mdates.DayLocator(interval=4))
cs_a.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
cs_a.set_ylabel(r'$^\circ\text{C} \, \text{m}^{2} \text{s}^{-1}$',fontsize = 10)

#B
cs_b = plt.subplot(gs[1, 1])

cs_b.text(0.01, 0.95, '(e)', transform=cs_b.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
#cs_b.set_ylim(bottom=-30, top=30)

cs_b.plot(time_no, S_no_b, label='Without winds', linestyle=':', color='royalblue')
cs_b.plot(time_w, S_w_b, label='With winds', linestyle='--', color='crimson')


# Define the date boundaries
t1_start = pd.to_datetime("2004-03-17")
t2_start = pd.to_datetime("2004-03-24")
t3_start = pd.to_datetime("2004-03-31")
t3_end   = pd.to_datetime("2004-04-06")

# Draw structural boundary lines between periods
cs_b.axvline(t2_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)
cs_b.axvline(t3_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)

# Calculate temporal midpoints for label placement
mid_t1 = t1_start + (t2_start - t1_start) / 2
mid_t2 = t2_start + (t3_start - t2_start) / 2
mid_t3 = t3_start + (t3_end - t3_start) / 2

# Place text blocks near the bottom x-axis (y = 0.5)
label_y_pos = -0.7 
cs_b.text(mid_t1, label_y_pos, 'T1', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
cs_b.text(mid_t2, label_y_pos, 'T2', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
cs_b.text(mid_t3, label_y_pos, 'T3', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
cs_b.text(0.01, 1.1, 'Section B', color='k',transform=cs_b.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')



for i, (start, end) in enumerate(during):
	cs_b.axvspan(pd.to_datetime(start),
				pd.to_datetime(end),
				color='lightsalmon',
				alpha=0.2,
				label='Hurricane' if i == 0 else None)


#cs_b.legend(loc=1,fontsize=9)
add_tidal_markers(cs_b)
cs_b.set_ylim(-0.8,0.8)
format_time_axis(cs_b)
cs_b.xaxis.set_major_locator(mdates.DayLocator(interval=4))
cs_b.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#cs_b.set_ylabel(r'$^\circ\text{C} \, \text{m}^{2} \text{s}^{-1}$',fontsize = 10)


#C
cs_c = plt.subplot(gs[1, 2])

cs_c.text(0.01, 0.95, '(f)', transform=cs_c.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')
#cs_c.set_ylim(bottom=-30, top=30)

cs_c.plot(time_no, S_no_c, label='Without winds', linestyle=':', color='royalblue')
cs_c.plot(time_w, S_w_c, label='With winds', linestyle='--', color='crimson')


# Define the date boundaries
t1_start = pd.to_datetime("2004-03-17")
t2_start = pd.to_datetime("2004-03-24")
t3_start = pd.to_datetime("2004-03-31")
t3_end   = pd.to_datetime("2004-04-06")

# Draw structural boundary lines between periods
cs_c.axvline(t2_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)
cs_c.axvline(t3_start, color='dimgrey', linestyle=':', linewidth=1.0, alpha=0.8, zorder=1)

# Calculate temporal midpoints for label placement
mid_t1 = t1_start + (t2_start - t1_start) / 2
mid_t2 = t2_start + (t3_start - t2_start) / 2
mid_t3 = t3_start + (t3_end - t3_start) / 2

# Place text blocks near the bottom x-axis (y = 0.5)
label_y_pos = -0.7 
cs_c.text(mid_t1, label_y_pos, 'T1', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
cs_c.text(mid_t2, label_y_pos, 'T2', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
cs_c.text(mid_t3, label_y_pos, 'T3', color='dimgrey', fontsize=9, fontweight='bold', ha='center', va='bottom')
cs_c.text(0.01, 1.1, 'Section C', color='k',transform=cs_c.transAxes, fontsize=10, fontweight='bold', va='top', ha='left')



for i, (start, end) in enumerate(during):
	cs_c.axvspan(pd.to_datetime(start),
				pd.to_datetime(end),
				color='lightsalmon',
				alpha=0.2,
				label='Hurricane' if i == 0 else None)


#cs_c.legend(loc=1,fontsize=9)
add_tidal_markers(cs_c)
cs_c.set_ylim(-0.8,0.8)
format_time_axis(cs_c)
cs_c.xaxis.set_major_locator(mdates.DayLocator(interval=4))
cs_c.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#cs_c.set_ylabel(r'$^\circ\text{C} \, \text{m}^{2} \text{s}^{-1}$',fontsize = 10)

########----> Cross sections <---########
#colorbar and limits
cmap = plt.cm.bwr
vmin = -2
vmax = 2
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
unit_cbar= r'$^\circ\text{C} \, \text{s}^{-1} \times 10^{-6}$'

cbar_3 = plt.subplot(gs[4, :])
cb3 = mpl.colorbar.ColorbarBase(cbar_3, cmap=cmap, norm=norm, extend='both', orientation='horizontal')
cb3.set_label(unit_cbar, size='small', labelpad=5)
cbar_3.xaxis.set_ticks_position('bottom')  # Ticks on the left side
cbar_3.tick_params(axis='x', labelsize='small', rotation=25)

#---> No winds
#SA
ax = plt.subplot(gs[2, 0])
ax.text(0.02, 0.1, '(g)', transform=ax.transAxes, fontsize=10, fontweight='demibold', color='k', 
		ha='left', va='top')
ax.text(0.02, 1.08, 'Section A: T1 Without winds', transform=ax.transAxes, fontsize=10, fontweight='demibold', color='k', 	
		ha='left', va='top')
zcross_s_a = np.nan_to_num(zcross_s_a, nan=0.0, posinf=0.0, neginf=0.0)
for i in range(len(dist_a)):
	ax.pcolormesh(dist_a, zcross_s_a[:-1,i], D_no_a_cross*1e6,cmap=cmap,norm=norm)
ax.fill_between(dist_a, np.min(zcross_s_a, axis=0), y2=ax.get_ylim()[0], color='darkgrey', zorder = 2)
ax.set_ylabel('m',size='small')
ax.tick_params(axis='both', labelsize='small')
ax.set_ylim(-300, 0) 
ax.set_xlim(0, 400) 
triangle_x = dist_a[z_slope_A]
triangle_y = zcross_s_b.min()  # Set y value for the triangle, adjust as needed
ax.scatter(triangle_x, -300, color='gold', marker='^', s=50, zorder=3)  # Triangle marker
ax.axvline(triangle_x,color='gold', lw = 2,alpha = 0.9)
F_overbar = mean_t1_no_a.values 
flux_text = fr'$\overline{{S(\theta)}} = {F_overbar:.4f}\,^\circ\text{{C}}\,\text{{m}}^2\,\text{{s}}^{{-1}}$'
ax.text(0.5, 0.4, flux_text, transform=ax.transAxes,
		 fontsize='small', fontweight='normal', color='k',
		 ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))

#SB

axh = plt.subplot(gs[2, 1])
axh.text(0.02, 0.1, '(h)', transform=axh.transAxes, fontsize=10, fontweight='demibold', color='k', 
		ha='left', va='top')
axh.text(0.02, 1.08, 'Section B: T2 Without winds', transform=axh.transAxes, fontsize=10, fontweight='demibold', color='k', 	
		ha='left', va='top')
zcross_s_b = np.nan_to_num(zcross_s_b, nan=0.0, posinf=0.0, neginf=0.0)
for i in range(len(dist_b)):
	axh.pcolormesh(dist_b, zcross_s_b[:-1,i], D_no_b_cross*1e6 ,cmap=cmap,norm=norm)
axh.fill_between(dist_b, np.min(zcross_s_b, axis=0), y2=axh.get_ylim()[0], color='darkgrey', zorder = 2)
axh.set_ylabel(None,size='small')
axh.tick_params(axis='both', labelsize='small')
axh.set_ylim(-300, 0) 
axh.set_xlim(0, 400) 
triangle_x = dist_b[z_slope_b]
triangle_y = zcross_s_a.min()  # Set y value for the triangle, adjust as needed
axh.scatter(triangle_x, -300, color='gold', marker='^', s=50, zorder=3)  # Triangle marker
axh.axvline(triangle_x,color='gold', lw = 2,alpha = 0.9)
F_overbar = mean_t2_no_b.values 
flux_text = fr'$\overline{{S(\theta)}} = {F_overbar:.4f}\,^\circ\text{{C}}\,\text{{m}}^2\,\text{{s}}^{{-1}}$'
axh.text(0.5, 0.4, flux_text, transform=axh.transAxes,
		 fontsize='small', fontweight='normal', color='k',
		 ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))
#SC

axi = plt.subplot(gs[2, 2])
axi.text(0.02, 0.1, '(i)', transform=axi.transAxes, fontsize=10, fontweight='demibold', color='k', 
		ha='left', va='top')
axi.text(0.02, 1.08, 'Section C: T3 Without winds', transform=axi.transAxes, fontsize=10, fontweight='demibold', color='k', 	
		ha='left', va='top')
zcross_s_c = np.nan_to_num(zcross_s_c, nan=0.0, posinf=0.0, neginf=0.0)
for i in range(len(dist_c)):
	axi.pcolormesh(dist_c, zcross_s_c[:-1,i], D_no_c_cross*1e6 ,cmap=cmap,norm=norm)
axi.fill_between(dist_c, np.min(zcross_s_c, axis=0), y2=axi.get_ylim()[0], color='darkgrey', zorder = 2)
axi.set_ylabel(None,size='small')
axi.tick_params(axis='both', labelsize='small')
axi.set_ylim(-300, 0) 
axi.set_xlim(0, 400) 
triangle_x = dist_c[z_slope_c]
triangle_y = zcross_s_a.min()  # Set y value for the triangle, adjust as needed
axi.scatter(triangle_x, -300, color='gold', marker='^', s=50, zorder=3)  # Triangle marker
axi.axvline(triangle_x,color='gold', lw = 2,alpha = 0.9)
F_overbar = mean_t3_no_c.values 
flux_text = fr'$\overline{{S(\theta)}} = {F_overbar:.4f}\,^\circ\text{{C}}\,\text{{m}}^2\,\text{{s}}^{{-1}}$'
axi.text(0.5, 0.4, flux_text, transform=axi.transAxes,
		 fontsize='small', fontweight='normal', color='k',
		 ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))

#---> With winds
#SA
axj = plt.subplot(gs[3, 0])
axj.text(0.02, 0.1, '(j)', transform=axj.transAxes, fontsize=10, fontweight='demibold', color='k', 
		ha='left', va='top')
axj.text(0.02, 1.08, 'Section A: T1 With winds', transform=axj.transAxes, fontsize=10, fontweight='demibold', color='k', 	
		ha='left', va='top')
zcross_s_a = np.nan_to_num(zcross_s_a, nan=0.0, posinf=0.0, neginf=0.0)
for i in range(len(dist_a)):
	axj.pcolormesh(dist_a, zcross_s_a[:-1,i], D_w_a_cross*1e6,cmap=cmap,norm=norm)
axj.fill_between(dist_a, np.min(zcross_s_a, axis=0), y2=axj.get_ylim()[0], color='darkgrey', zorder = 2)
axj.set_ylabel('m',size='small')
axj.tick_params(axis='both', labelsize='small')
axj.set_ylim(-300, 0) 
axj.set_xlim(0, 400) 
triangle_x = dist_a[z_slope_A]
triangle_y = zcross_s_b.min()  # Set y value for the triangle, adjust as needed
axj.scatter(triangle_x, -300, color='gold', marker='^', s=50, zorder=3)  # Triangle marker
axj.axvline(triangle_x,color='gold', lw = 2,alpha = 0.9)
F_overbar = mean_t1_w_a.values 
flux_text = fr'$\overline{{S(\theta)}} = {F_overbar:.4f}\,^\circ\text{{C}}\,\text{{m}}^2\,\text{{s}}^{{-1}}$'
axj.text(0.5, 0.4, flux_text, transform=axj.transAxes,
		 fontsize='small', fontweight='normal', color='k',
		 ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))

#SB

axk = plt.subplot(gs[3, 1])
axk.text(0.02, 0.1, '(k)', transform=axk.transAxes, fontsize=10, fontweight='demibold', color='k', 
		ha='left', va='top')
axk.text(0.02, 1.08, 'Section B: T2 With winds', transform=axk.transAxes, fontsize=10, fontweight='demibold', color='k', 	
		ha='left', va='top')
zcross_s_b = np.nan_to_num(zcross_s_b, nan=0.0, posinf=0.0, neginf=0.0)
for i in range(len(dist_b)):
	axk.pcolormesh(dist_b, zcross_s_b[:-1,i], D_w_b_cross*1e6 ,cmap=cmap,norm=norm)
axk.fill_between(dist_b, np.min(zcross_s_b, axis=0), y2=axk.get_ylim()[0], color='darkgrey', zorder = 2)
axk.set_ylabel(None,size='small')
axk.tick_params(axis='both', labelsize='small')
axk.set_ylim(-300, 0) 
axk.set_xlim(0, 400) 
triangle_x = dist_b[z_slope_b]
triangle_y = zcross_s_a.min()  # Set y value for the triangle, adjust as needed
axk.scatter(triangle_x, -300, color='gold', marker='^', s=50, zorder=3)  # Triangle marker
axk.axvline(triangle_x,color='gold', lw = 2,alpha = 0.9)
F_overbar = mean_t2_w_b.values 
flux_text = fr'$\overline{{S(\theta)}} = {F_overbar:.4f}\,^\circ\text{{C}}\,\text{{m}}^2\,\text{{s}}^{{-1}}$'
axk.text(0.5, 0.4, flux_text, transform=axk.transAxes,
		 fontsize='small', fontweight='normal', color='k',
		 ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))
#SC

axl = plt.subplot(gs[3, 2])
axl.text(0.02, 0.1, '(l)', transform=axl.transAxes, fontsize=10, fontweight='demibold', color='k', 
		ha='left', va='top')
axl.text(0.02, 1.08, 'Section C: T3 With winds', transform=axl.transAxes, fontsize=10, fontweight='demibold', color='k', 	
		ha='left', va='top')
zcross_s_c = np.nan_to_num(zcross_s_c, nan=0.0, posinf=0.0, neginf=0.0)
for i in range(len(dist_c)):
	axl.pcolormesh(dist_c, zcross_s_c[:-1,i], D_w_c_cross*1e6 ,cmap=cmap,norm=norm)
axl.fill_between(dist_c, np.min(zcross_s_c, axis=0), y2=axl.get_ylim()[0], color='darkgrey', zorder = 2)
axl.set_ylabel(None,size='small')
axl.tick_params(axis='both', labelsize='small')
axl.set_ylim(-300, 0) 
axl.set_xlim(0, 400) 
triangle_x = dist_c[z_slope_c]
triangle_y = zcross_s_a.min()  # Set y value for the triangle, adjust as needed
axl.scatter(triangle_x, -300, color='gold', marker='^', s=50, zorder=3)  # Triangle marker
axl.axvline(triangle_x,color='gold', lw = 2,alpha = 0.9)
F_overbar = mean_t3_w_c.values 
flux_text = fr'$\overline{{S(\theta)}} = {F_overbar:.4f}\,^\circ\text{{C}}\,\text{{m}}^2\,\text{{s}}^{{-1}}$'
axl.text(0.5, 0.4, flux_text, transform=axl.transAxes,
		 fontsize='small', fontweight='normal', color='k',
		 ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'))

















plt.savefig('heat_flux.png', dpi = 300)

