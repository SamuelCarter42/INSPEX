import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
from astropy.time import Time, TimeDelta

import spiceypy
from sunpy.coordinates import spice as spice

from sunpy.data import cache
from sunpy.time import parse_time
import sunpy.data.sample
import sunpy.map
from sunpy.coordinates import frames, sun

import sunpy_soar
import datetime as dt
import os
import pandas as pd
import warnings
from solo_epd_loader import epd_load
from sunpy.net import attrs as a
#from sunpy.net import FidoS
from sunpy.net import Fido
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit, Bounds as bd
import scipy.stats as stats
import lmfit
from scipy.special import erf, erfc
from tqdm import tqdm
import sys
from spacepy import pycdf

sys.path.append('C:/Users/w23014130/OneDrive - Northumbria University - Production Azure AD/Documents/PhD/In situ spectrum gui')
sys.path.append('C:/Users/w23014130/OneDrive - Northumbria University - Production Azure AD/Documents/PhD/In situ spectrum gui/Dependencies')






def EAS_data_load(date_for_spec,tstart,tend,epd_xyz_sectors,low_e_cutoff=0.8):

    tstart_set=tstart
    tend_set=date_for_spec+' 23:59:59'

    #must load day before and after too, to avoid data gaps
    
    date_obj = dt.datetime.strptime(date_for_spec, "%Y/%m/%d")
    
    # Add one day
    next_day = date_obj + dt.timedelta(days=1)
    
    # Convert back to string in the same format
    next_day= next_day.strftime("%Y/%m/%d")
    
    # minus one day
    prev_day = date_obj - dt.timedelta(days=1)
    
    # Convert back to string in the same format
    prev_day= prev_day.strftime("%Y/%m/%d")
    
    tstart = f'{prev_day} 00:00:00'
    tend = f'{next_day} 23:59:59'
    #%%load eas flux data
    date_range = a.Time(tstart, tend)
    instrument=a.Instrument("SWA")
    level = a.Level(2)

    
    product = a.soar.Product("swa-eas1-nm3d-dnf")
    result = Fido.search(date_range, instrument,product)
    EAS1_files = Fido.fetch(result)
    print(EAS1_files)
    
    #

    flux_files = [pycdf.CDF(str(i)) for i in EAS1_files]

    
    
    
    
    
    
    #%%load eas count data

    
    product = a.soar.Product("swa-eas1-nm3d")
    result = Fido.search(date_range, instrument,product)
    EAS1_files = Fido.fetch(result)
    #print(EAS1_files)
    
    if len(EAS1_files)==0:
        print('No EAS files found for this time period')
        sys.exit()

    files = [pycdf.CDF(str(i)) for i in EAS1_files]
    #%% times

    times=[]
    [times.extend(list(i['EPOCH'])) for i in files]#append each individual cdf together 
    
    
    #breakpoint()

    
    energies=np.array(files[0]['SWA_EAS1_ENERGY'])#energy bins same at all times for all files
    #breakpoint()
    energies=energies/1000 #convert ev to kev

    
    elevs_raw=np.array(files[0]['SWA_EAS_ELEVATION'])
    azs_raw=np.array(files[0]['SWA_EAS_AZIMUTH'])
    
    tops=files[0]['SWA_EAS1_ENERGY_delta_upper']
    bots=files[0]['SWA_EAS1_ENERGY_delta_lower']
    
    
    #convert the intrument angle coordinates to heliocentric
    
    #%%process look directions-only EAS1 ovelaps
    EPD_XYZ=epd_xyz_sectors 
    #retrieve step xyz
    EPD_X=EPD_XYZ[:,0]
    EPD_Y=EPD_XYZ[:,1]
    EPD_Z=EPD_XYZ[:,2]
    
    
    
    
    
    ### constants #
    rad=np.pi/180.
    au=149597870.7
    mass=9.11*10.**(-31.)
    joule=1.602*10.**(-19.)
    m0=1.2566*10.**(-6.)
    kb=1.3806*10.**(-23.)
    ###############
    #These are the STEP FOV values in the RTN frame
    elevation_min = -27
    elevation_max = 27
    azimuth_min = 21
    azimuth_max = 49
    
    EN1,EL1,AZ1=np.meshgrid(energies,elevs_raw,azs_raw)
    V1i=np.sqrt(2.*EN1*joule/mass)
    VX1i=V1i*np.cos(EL1*rad)*np.cos(AZ1*rad)
    VY1i=V1i*np.cos(EL1*rad)*np.sin(AZ1*rad)
    VZ1i=-V1i*np.sin(EL1*rad)

    V1_inst=1.*V1i
    VX1_inst=VY1i*np.cos(45.*rad)-VZ1i*np.cos(45.*rad)
    VY1_inst=-VY1i*np.cos(45.*rad)-VZ1i*np.cos(45.*rad)
    VZ1_inst=-VX1i

    EAS1_X=-VX1_inst/V1_inst
    EAS1_Y=-VY1_inst/V1_inst
    EAS1_Z=VZ1_inst/V1_inst
        
    ##########################################################################
    EAS1_fov_sel=np.where((EAS1_X>=np.min(EPD_X)) & (EAS1_X<=np.max(EPD_X)) & (EAS1_Y>=np.min(EPD_Y)) & (EAS1_Y<=np.max(EPD_Y))  & (EAS1_Z>=np.min(EPD_Z)) & (EAS1_Z<=np.max(EPD_Z)) )

    
    EAS1_Xf=EAS1_X.reshape(-1)
    EAS1_Yf=EAS1_Y.reshape(-1)
    EAS1_Zf=EAS1_Z.reshape(-1)
    
    EAS1_X=1.*EAS1_X[EAS1_fov_sel]
    EAS1_Y=1.*EAS1_Y[EAS1_fov_sel]
    EAS1_Z=1.*EAS1_Z[EAS1_fov_sel]
    
    
    
    EAS1_Xs=EAS1_X.reshape(-1)
    EAS1_Ys=EAS1_Y.reshape(-1)
    EAS1_Zs=EAS1_Z.reshape(-1)
    
    
    ###############
    ax = plt.axes(projection='3d')
    ax.scatter3D(EAS1_Xf,EAS1_Yf,EAS1_Zf,facecolors='none', edgecolors='blue',rasterized=True)
    ax.scatter3D(1.1*EPD_X,1.1*EPD_Y,1.1*EPD_Z,color='black',rasterized=True)
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.2,1.2])
    ax.set_zlim([-1.2,1.2])
    ax.set_xlabel('Vx_SRF')
    ax.set_ylabel('Vy_SRF')
    ax.set_zlabel('Vz_SRF')
    plt.show()


    ax = plt.axes(projection='3d')
    ax.scatter3D(EAS1_Xs,EAS1_Ys,EAS1_Zs,facecolors='none', edgecolors='blue',rasterized=True)
    ax.scatter3D(EPD_X,EPD_Y,EPD_Z,color='black',rasterized=True)
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.2,1.2])
    ax.set_zlim([-1.2,1.2])
    ax.set_xlabel('Vx_SRF')
    ax.set_ylabel('Vy_SRF')
    ax.set_zlabel('Vz_SRF')
    plt.show()
    
    #convert the xyz back to angles to allow easier filtering
    Els=np.arcsin((EAS1_Xs+EAS1_Ys)/(-2*np.cos(45.*rad)))
    
    Azs=np.arccos(EAS1_Zs/(-np.cos(Els)))
    
    
    Azs=np.unique(np.round(Azs,5))
    Els=np.unique(np.round(Els,5))
    
    eas_el_r=np.round(elevs_raw*rad,5)
    eas_az_r=np.round(azs_raw*rad,5)
    
    # Create boolean mask
    el_mask=np.isin(eas_el_r,Els)
    az_mask=np.isin(eas_az_r,Azs)
    
    #%% remove all the unwanted bits
    # Define the range of desired elevations and azimuths (e.g., min and max)
    
    #eas2 does not opverlap so not included   
    #breakpoint()
    #eas1
    combo_files=np.concatenate([i['SWA_EAS1_Data'] for i in files])#combine all files into a single time series
    
    # Deduplicate time array and apply the same to combo_files, as the loaded data can contain duplicates
    times = np.concatenate([i['EPOCH'][:] for i in files])
    times_unique, unique_indices = np.unique(times, return_index=True)
    
    combo_files = combo_files[unique_indices]

    #use the times from flux as these seem better
    times_flux = np.concatenate([i['EPOCH'][:] for i in flux_files])
    
    flux_files=np.concatenate([i['SWA_EAS1_NM3D_DNF_Data'] for i in flux_files])#combine all files into a single time series
    
    #breakpoint()
        # --- ALIGN TIMES --- in case there are bins not present in both
    # Find times common to both sources
    common_times = np.intersect1d(times_unique, times_flux)
    
    # Create masks for common times
    mask_combo = np.isin(times_unique, common_times)
    mask_flux = np.isin(times_flux, common_times)
    
    # Apply masks
    times_unique = times_unique[mask_combo]
    combo_files = combo_files[mask_combo]
    
    times_flux = times_flux[mask_flux]
    flux_files = flux_files[mask_flux]
    
    
    
    # Repeat for the second set
    #time,el,az,energy
    combo_files = np.transpose(combo_files, (0, 1, 3, 2))
    flux_files = np.transpose(flux_files, (0, 1, 3, 2))
    #print('EAS1 flux:')
    #print((flux_files2==0).any())    
    
        
    #breakpoint()
    #try:
    for tind, this_t in tqdm(enumerate(flux_files)):
       
       
       
       # Apply the mask across all energy bins for the second set
       combo_files[tind, ~el_mask][:, ~az_mask, :] = 0 #tilde is not
       flux_files[tind, ~el_mask][:, ~az_mask, :] = 0
        
    #except:
        #breakpoint()
    no_pix=np.count_nonzero(el_mask)*np.count_nonzero(az_mask)
    # Sum over azimuth (axis 2) and elevation (axis 1) for the second set
    curve = np.sum(np.sum(combo_files, axis=2), axis=1)/no_pix
    flux_curve = np.sum(np.sum(flux_files, axis=2), axis=1)/no_pix
    
    
    count_curve_raw=np.array(curve)
    flux_curve_raw=np.array(flux_curve)
    
    #print(count_curve_raw.shape)
    
    combo_energies=np.array(energies[0])#each time has same energy bins so use first
    
    #breakpoint()
    #%%sawtooth correction, only keep even index bins
    count_curve_valid=list()
    flux_curve_valid=list()
    valid_energies=list()
    valid_widths=list()
    energy_lims_eas=list()
    for count, energy in enumerate(combo_energies):
        #breakpoint()
        if count % 2 == 0: #if is even
            count_curve_valid.append([i[count] for i in count_curve_raw])
            flux_curve_valid.append([i[count] for i in flux_curve_raw])
            valid_energies.append(energy)
            
            width=tops[count]-bots[count]
            valid_widths.append(abs(width))
            energy_lims_eas.append((bots[count],tops[count]))
    count_curve=np.array(count_curve_valid).transpose()#swap into array then format to correct axes
    
    flux_curve=np.array(flux_curve_valid).transpose()#swap into array then format to correct axes
    flux_curve=(flux_curve)/10#keV conversion, cm-2 conversion)
    
    flux_curve=np.fliplr(flux_curve)#reverse order to ascending energy
    count_curve=np.fliplr(count_curve)
    valid_energies=np.array(valid_energies[::-1])
    valid_widths=np.array(valid_widths[::-1])
    energy_lims_eas=np.array(energy_lims_eas[::-1])
    
    #%%discard everything below low energy cutoff
    
    
    count_curve=count_curve[:,valid_energies>low_e_cutoff]
    flux_curve=flux_curve[:,valid_energies>low_e_cutoff]
    valid_widths=np.ma.masked_array(valid_widths, valid_energies<low_e_cutoff).compressed()
    
    
    mask = np.array(valid_energies) >= low_e_cutoff
    
    energy_lims_eas=energy_lims_eas[mask]
    
    
    #THIS MUST BE LAST SO OTHERS FILTER PROPERLY
    valid_energies=np.ma.masked_array(valid_energies, valid_energies<low_e_cutoff).compressed()

    #%%calculate flux uncertainty
    
    sqrt_curve=np.sqrt(count_curve)
    
    temp_curve=sqrt_curve/(0.92e-3*7.8e-6)#sqrt(c)/dtG
    
    uncert_curve=temp_curve/valid_widths[None,:]
    
    #%%cut back down to just day of interest
    
    
    mask=(times_flux>=dt.datetime.strptime(tstart_set,"%Y/%m/%d %H:%M:%S")) & (times_flux<dt.datetime.strptime(tend_set,"%Y/%m/%d %H:%M:%S") )
    times_flux=times_flux[mask]
    flux_curve=flux_curve[mask,:]
    uncert_curve=uncert_curve[mask,:]
    #Can outputbin limits too
    return times_flux,valid_energies,flux_curve,uncert_curve#,np.array(energy_lims_eas)/1000

