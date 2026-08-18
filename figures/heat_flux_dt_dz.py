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

#dt/dz (to be plotted per depth and km from coast: C/m)
D_no_a_cross = dT_dz_a_no.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time').compute()

#Sum of Flux from coast to slope (to be plotted per time C m2/s)
D_no_a =flux_no_a.diff(dim='s_rho')/dz_a

int_theta_no_a = (D_no_a.isel(xi_rho=slice(0,z_slope_a))*dz_a.isel(xi_rho=slice(0,z_slope_a))*dx_a.isel(xi_rho=slice(0,z_slope_a))).compute()
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


#dt/dz (to be plotted per depth and km from coast: C/m)
D_no_b_cross = dT_dz_b_no.sel(ocean_time=slice(during_start, during_end)).mean(dim='ocean_time').compute()

#Sum of Flux from coast to slope (to be plotted per time C m2/s)
D_no_b = flux_no_b.diff(dim='s_rho')/dz_b

int_theta_no_b = (D_no_b.isel(xi_rho=slice(0,z_slope_b))*dz_b.isel(xi_rho=slice(0,z_slope_b))*dx_b.isel(xi_rho=slice(0,z_slope_b))).compute()
S_no_b = int_theta_no_b.sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)


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
flux_no_c = (akt_no_c * dT_dz_c_no).sel(ocean_time=slice(before_start, after_end)).rolling(ocean_time=4, center=True).mean() 
time_no = flux_no_c.ocean_time

#dt/dz (to be plotted per depth and km from coast: C/m)
D_no_c_cross = dT_dz_c_no.sel(ocean_time=slice(after_start, after_end)).mean(dim='ocean_time').compute()


#Sum of Flux from coast to slope (to be plotted per time C m2/s)
D_no_c = flux_no_c.diff(dim='s_rho')/dz_c

int_theta_no_c = (D_no_c.isel(xi_rho=slice(0,z_slope_c))*dz_c.isel(xi_rho=slice(0,z_slope_c))*dx_c.isel(xi_rho=slice(0,z_slope_c))).compute()
S_no_c = int_theta_no_c.sum(dim='s_rho').sum(dim='xi_rho').compute().fillna(0)
#S_no_c = xr.where(S_no_c.ocean_time >= S_no_c.ocean_time[75], S_no_c * 2, S_no_c)

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

#dt/dz (to be plotted per depth and km from coast: C/m)
D_w_a_cross = dT_dz_a_w.sel(ocean_time=slice(before_start, before_end)).mean(dim='ocean_time').compute()

#Sum of Flux from coast to slope (to be plotted per time C m2/s)
D_w_a = flux_w_a.diff(dim='s_rho')/dz_a

int_theta_w_a = (D_w_a.isel(xi_rho=slice(0,z_slope_a))*dz_a.isel(xi_rho=slice(0,z_slope_a))*dx_a.isel(xi_rho=slice(0,z_slope_a))).compute()
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


#dt/dz (to be plotted per depth and km from coast: C/m)
D_w_b_cross = dT_dz_b_w.sel(ocean_time=slice(during_start, during_end)).mean(dim='ocean_time').compute()

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

#dt/dz (to be plotted per depth and km from coast: C/m)
D_w_c_cross = dT_dz_c_w.sel(ocean_time=slice(after_start, after_end)).mean(dim='ocean_time').compute()

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

