import os
import glob
import re
import xarray as xr
import dask
import xroms
from dask.diagnostics import ProgressBar
import numpy as np
import gc

# --- Paths ---
# Update this filename path to point to your M2 velocity dataset if it's separate
filename = '/Users/piero/arian/data1/nc_outs/avg_internal_tides_paper.nc' 
out_path = '/Users/piero/arian/data1/NO_WINDS_IT/ke_modes/'
os.makedirs(out_path, exist_ok=True)

# --- Configuration ---
NUM_MODES = 5      # Modes 0, 1, 2, 3, 4
WINDOW_SIZE = 8    # SVD window chunk size
ETA_STEP = 40      
eta_steps = np.arange(0, 1360, ETA_STEP) 
xis = slice(40, None)

# Target T3 Time Scenario Range
t3_start, t3_end = "2004-03-31", "2004-04-06"

# 1. Open Dataset Lazily and immediately isolate the T3 period
ds1 = xr.open_dataset(filename, chunks={'ocean_time': WINDOW_SIZE, 's_rho': -1, 'eta_rho': "auto", 'xi_rho': "auto"})
ds1_t3 = ds1.sel(ocean_time=slice(t3_start, t3_end))
ds, xgrid = xroms.roms_dataset(ds1_t3)

# 2. Interpolate M2 velocities to the central Rho grid lazily
# Note: change 'u_m2'/'v_m2' to 'u'/'v' if your variables are named differently in this file
u_all = xroms.to_rho(ds.u_m2, xgrid)
v_all = xroms.to_rho(ds.v_m2, xgrid)

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================
for i in range(len(eta_steps) - 1):
    print(f"--- Processing KE Modal Slice {i} (Rows {eta_steps[i]} to {eta_steps[i+1]}) ---")
    eta_slice = slice(eta_steps[i], eta_steps[i+1])
    ds_sub = ds.isel(eta_rho=eta_slice, xi_rho=xis)
    
    nt = len(ds_sub.ocean_time)
    n_windows = nt // WINDOW_SIZE
    n_eta = eta_steps[i+1] - eta_steps[i]  # Exactly 10 rows
    n_xi = len(ds_sub.xi_rho)
    
    # Accumulators for the time-averaged 2D maps across the T3 window
    ke_total_accum = np.zeros((n_eta, n_xi), dtype=np.float32)
    ke_modes_accum = np.zeros((NUM_MODES, n_eta, n_xi), dtype=np.float32)
    
    # Loop over Tidal Windows
    for w in range(n_windows):
        t_idx = slice(w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE)
        
        # Load small 3D chunks into RAM
        u_win = u_all.isel(eta_rho=eta_slice, xi_rho=xis, ocean_time=t_idx).values   # (Time, Depth, Eta, Xi)
        v_win = v_all.isel(eta_rho=eta_slice, xi_rho=xis, ocean_time=t_idx).values   # (Time, Depth, Eta, Xi)
        dz_win = ds_sub.dz.isel(ocean_time=t_idx).values                             # (Time, Depth, Eta, Xi)
        
        # A. Calculate Depth-Integrated Total M2 Kinetic Energy for this window
        ke_total_3d = 1025 * 0.5 * (u_win**2 + v_win**2)                             # J/m3
        ke_total_depth = np.sum(ke_total_3d * dz_win, axis=1)                        # J/m2 (Time, Eta, Xi)
        ke_total_accum += np.mean(ke_total_depth, axis=0) / n_windows                # Time average map contribution
        
        # B. Set up SVD matrices over depth
        dz_m = np.mean(dz_win, axis=0)  # (Depth, Eta, Xi)
        weight = np.sqrt(dz_m)          # Square-root depth weighting for energy conservation
        
        # Transpose to (Eta, Xi, Time, Depth) to isolate (Time, Depth) as SVD target space
        u_win_t = u_win.transpose(2, 3, 0, 1)
        v_win_t = v_win.transpose(2, 3, 0, 1)
        weight_t = weight.transpose(1, 2, 0)
        
        u_weighted = np.nan_to_num(u_win_t * weight_t[:, :, None, :], nan=0.0)
        v_weighted = np.nan_to_num(v_win_t * weight_t[:, :, None, :], nan=0.0)
        
        # De-mean to look at baroclinic velocity variances
        u_anom = u_weighted - np.mean(u_weighted, axis=2, keepdims=True)
        v_anom = v_weighted - np.mean(v_weighted, axis=2, keepdims=True)
        
        # Vectorized Stacked SVD execution
        U_u, S_u, _ = np.linalg.svd(u_anom, full_matrices=False)
        U_v, S_v, _ = np.linalg.svd(v_anom, full_matrices=False)
        
        # C. Calculate Depth-Integrated Kinetic Energy per Mode
        # Thanks to orthogonal SVD properties, the depth-integral of a weighted mode 
        # simplifies perfectly to: S_n**2 * U_n(t)**2
        for n in range(NUM_MODES):
            amp_u_sq = (U_u[..., :, n] * S_u[..., n, None])**2  # (Eta, Xi, Time)
            amp_v_sq = (U_v[..., :, n] * S_v[..., n, None])**2  # (Eta, Xi, Time)
            
            ke_mode_depth = 1025 * 0.5 * (amp_u_sq + amp_v_sq)   # J/m2 (Eta, Xi, Time)
            ke_modes_accum[n, :, :] += np.mean(ke_mode_depth, axis=2) / n_windows
            
        del u_win, v_win, dz_win, u_anom, v_anom, U_u, S_u, U_v, S_v
    
    # 3. Export clean 2D spatial maps for this slice
    print(f"--- Saving compiled KE maps for slice {i} ---")
    output_ds = xr.Dataset(
        {
            'ke_total_t3_avg': (['eta_rho', 'xi_rho'], ke_total_accum),
            'ke_modes_t3_avg': (['mode', 'eta_rho', 'xi_rho'], ke_modes_accum)
        },
        coords={
            'mode': np.arange(NUM_MODES),
            'eta_rho': ds_sub.eta_rho,
            'xi_rho': ds_sub.xi_rho
        }
    )
    
    save_file = os.path.join(out_path, f'ke_t3_slice_{i}.nc')
    output_ds.to_netcdf(save_file)
    
    del ke_total_accum, ke_modes_accum, output_ds
    gc.collect()

ds1.close()
print("Processing complete! You can now merge the lightweight slice files to plot.")