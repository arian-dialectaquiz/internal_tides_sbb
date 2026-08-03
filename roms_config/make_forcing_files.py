
"""
This script preprocesses the era5 winds for roms

required variables for the script (single levels product of ERA5):

ATTENTION TO THE UNITS! You can switch fluxes for mean fluxes provided you adjust the scale factors
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
varinfo.yml in ROMS

"10m_u_component_of_wind",     # used in dQdSST
"10m_v_component_of_wind",     # used in dQdSST
"2m_temperature",              # used in dQdSST/air_density
"sea_level_pressure",          # used in dQdSST/air_density
"sea_surface_temperature",     # used in masks
"surface_latent_heat_flux",    # used in heat_flux
"surface_sensible_heat_flux",  # used in heat_flux
"surface_net_solar_radiation",   # used in heat_flux
"surface_net_thermal_radiation", # used in heat_flux
"total_precipitation",         # used in net_freshwater
"evaporation",                 # used in net_freshwater
"msdwlwrf"           {'ECMWFlongname': 'surface_thermal_radiation_downwards',
required variables for the script (pressure levels product of ERA5)?
"specific_humidity"            # used in air_density

"""

import xarray as xr
from utils.atmos_forcing import dQdT, extrapolating_era5, variables_list
from utils import utils as ut
import numpy as np
import pandas as pd
from matplotlib import rcParams
from netCDF4 import date2num, num2date
from scipy import ndimage
import sys
import os
from datetime import datetime as dtt
import glob
from hurricane_blend import blend_era5_vortex, load_track_ibtracs

def mixing_ratio(specific_humidity):
	q = specific_humidity
	return q/(1-q)

def virtual_temperature(temperature, mixing_ratio):
	tv = temperature*(1 + 0.608*mixing_ratio)
	return tv

def air_density(pressure, virtual_temperature):
	rho = pressure/ (287*virtual_temperature)
	return rho

def mask_sst(ds, varb='sst'):
	condition = ~np.isnan(ds[varb].isel(time=0))
	mask = xr.where(condition, 0, 1)
	return mask

def net_freshwater(ds,
				   evaporation='e',
				   precipitation='tp',
				   scale=1,
				   ):

	return (ds[evaporation] + ds[precipitation])*scale

def heat_flux(ds,
			  qs='sshf', # surface sensible heat flux
			  ql='slhf', # surface latent heat flux
			  ssr='ssr', # surface short radiation 
			  str='str', # surface thermal radiation
			  scale=1):
	return (ds[qs] + ds[ql] + ds[ssr] + ds[str])*scale   



# -- gets  the information from the config file -- #
# getting the referemce domain from shell 
#if len(sys.argv) > 1:
#	reference = sys.argv[1]
#else:
reference = 'paper_2_3km'

#dicts = ut._get_dict_paths(f'{os.path.dirname(__file__)}/../configs/grid_config_esmf.txt')[reference]
dicts = ut._get_dict_paths('/home/arian/dd_waves/pyroms_tools/configs/grid_config_esmf.txt')[reference]


# dicts resolver
fpath_single   = dicts['frc.era.single']
fpath_pressure = dicts['frc.era.press']
timei          = dicts['frc.date'][0]
timef          = dicts['frc.date'][1]
dt             = dicts['frc.date'][2]
odir           = dicts['frc.outputdir']


nc = xr.open_mfdataset(fpath_single, decode_times=False)
nc1 = xr.open_mfdataset(fpath_pressure, decode_times=False)
nc = nc.rename({'valid_time': 'time'})
###---> new era5 variable names
nc = nc.rename({'msl': 'sp'})
nc = nc.rename({'avg_iews': 'metss'})
nc = nc.rename({'avg_inss': 'mntss'})
nc = nc.rename({'avg_snswrf': 'msdwlwrf'})

metadata = variables_list


nc1 = nc1.rename({'valid_time': 'time'})

nc['q'] = nc1['q'][:,0,:,:]
sstK = True

tstart_num = date2num(dtt.strptime(timei, '%Y-%m-%dT%H:%M:%S'), nc.time.units)
tfinal_num = date2num(dtt.strptime(timef, '%Y-%m-%dT%H:%M:%S'), nc1.time.units)

nc = nc.sel(time=slice(tstart_num, tfinal_num))
nc1 = nc1.sel(time=slice(tstart_num, tfinal_num))



# time0 = num2date(nc.time[0].values, nc.time.attrs['units'])
tref = pd.date_range(start=timei, periods=nc.time.size, freq='1H')
tref1 = date2num(tref.to_pydatetime(), 'days since 1990-01-01 00:00:00')

nc['mask'] = mask_sst(nc, varb='sst')
# -- heat flux -- #
nc['shflux'] = heat_flux(nc,
						qs='sshf', # surface sensible heat flux
						ql='slhf', # surface latent heat flux
						ssr='ssr', # surface short radiation 
						str='str', # surface thermal radiation)
						scale=1/3600)	
# -- fresh water fluxes -- #
nc['swflux'] = -net_freshwater(nc,
							evaporation='e',
							precipitation='tp',
							scale=1e-3/3600)	

#####################################################
#####-------> Catarina Hurricane <----###############


track = load_track_ibtracs(
	'/home/arian/dd_waves/pyroms_tools/scripts_xesmf/catarina_2004_full.csv',
	tmin='2004-03-24', tmax='2004-03-28 12:00')

nc = blend_era5_vortex(nc, tref, track, params=dict(
	bshape              = 'auto',   # calibrate B to the reported vmax
	gradient_to_surface = 1.00,
	rmax_km             = 30.0,
	rb_factor           = 3.5,
	wb_factor           = 2.5,
	inflow_angle_deg    = 22.0,
	alpha_trans         = 0.55,
	blend_pressure      = True,
))

##########----> rest of the original code from ROMS rutgers people


# -- surface net heat flux sensitivity to SST -- #
wspd = (nc['u10']**2 + nc['v10']**2)**0.5
w    = mixing_ratio(nc['q'])
tv   = virtual_temperature(nc['t2m'], w)
rhoa = air_density(nc['sp'], tv)	
dqdsst = dQdT(wspd,nc['sp'], nc['t2m'], rhoair=rhoa)

nc['dQdSST']= dqdsst.dQdT()#.isel(pressure_level=0) #there is just the surface at the pressure levels files	

ncout = nc[['tp','e','q','mask', 'shflux', 'swflux', 'metss', 'mntss', 'dQdSST', 'sst','sp','t2m','u10', 'v10','ssr','str','msdwlwrf']]

for varname in ['msdwlwrf','tp','e','q','sp','sst', 'metss', 'shflux','mntss', 'swflux', 'dQdSST', 't2m','u10', 'v10','ssr']:
#for varname in ['q','sp','sst', 'metss', 'shflux','mntss', 'swflux', 'dQdSST', 't2m','u10', 'v10','ssr']:

	rename = metadata[varname]['outputName']
	var = extrapolating_era5(ncout, varname, None, 
								extrapolate_method='laplace', dst=ncout, mask=nc['mask'])	


	if varname=='sp':
		aux = extrapolating_era5(ncout, varname, None, 
								extrapolate_method='xesmf', dst=ncout, mask=abs(nc['mask']-2+1))
		var.sp.values = aux.sp.values
		var['sp'] = var['sp'].fillna(var.sp.mean().values)
		aux = ndimage.gaussian_filter(var.sp.values,1)
		var.sp.values = aux	

	var = var.reindex(latitude=list(reversed(var.latitude.values)))
	var = var.rename({varname: rename,
					'latitude': 'lat',
					'longitude': 'lon'})
	
	var[rename].attrs['coordinates'] = 'lon lat' 
	var[rename].attrs['units'] = metadata[varname]['units']
	var[rename].attrs['scale_factor'] = 1
	var[rename].attrs['add_offset'] = 0
	var = var.assign_coords({'time': tref1})
	var['time'].attrs['units'] = 'days since 1990-01-01 00:00:00'	
	var.attrs['time'] = metadata[varname]['time']
	var.attrs['coordinates'] = 'lon lat'
	# var = var.assign_coords(time=pd.date_range(start=ncout.time.values[0], freq='1H', periods=ncout.time.size))
	print(rename)
	var.attrs = {}
	var.to_netcdf(f'/data1/roms_dd_waves/ROMS_NEW/projects/2004_paper_2/1km/inputs//exp_{rename}_1km_hc.nc', format='NETCDF4')

		
dsout = xr.open_mfdataset(f'exp_*.nc')
dsout.to_netcdf('forcing_paper2_hc.nc')
os.system(f'rm exp_*.nc')


################

