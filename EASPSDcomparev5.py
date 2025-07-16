
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

from EAS_data_load_v21 import EAS_data_load

import pickle #saves vars

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from dateutil.relativedelta import relativedelta

def normalize_series_list(series_list):
    # Find the global min and max across all series
    global_min = min(s.min() for s in series_list)
    global_max = max(s.max() for s in series_list)
    
    # Normalize each series using the global min/max
    normalized_list = [(s - global_min) / (global_max - global_min) for s in series_list]
    
    return normalized_list

#%%choose time
date_for_spec='2021/10/09'
#specify the time range to look for data over
tstart = f'{date_for_spec} 00:00:00'
tend = f'{date_for_spec} 23:59:59'
#tstart = f'{date_for_spec} 00:00:00'
#tend = f'{date_for_spec} 23:59:59'

#select period for background
#bg_start = f'{date_for_spec} 00:00:00'
bg_start = f'{date_for_spec} 00:01:00'
bg_end = f'{date_for_spec} 02:00:00'



#the number of second to integrate over
integ_start=f'{date_for_spec} 03:01:00'
integ_time=25000
integ_end=f'{date_for_spec} 14:00:00'     #dt.datetime.strptime(integ_start,"%Y/%m/%d %H:%M:%S")+dt.timedelta(0,integ_time) # days, seconds


#HEK event time range of interest
integ_period=(Time(dt.datetime.strptime(integ_start,"%Y/%m/%d %H:%M:%S")),Time(dt.datetime.strptime(integ_end,"%Y/%m/%d %H:%M:%S")))
start_time=integ_start
end_time=integ_end

low_e_cutoff=0.5

threshold=1
#%%load eas PSD data
date_range = a.Time(tstart, tend)
instrument=a.Instrument("SWA")
level = a.Level(2)


product = a.soar.Product("swa-eas2-nm3d-psd")
result = Fido.search(date_range, instrument,product)
EAS2_files = Fido.fetch(result)
print(EAS2_files)

#sys.exit()

psd_files2 = [pycdf.CDF(str(i)) for i in EAS2_files]


#%% times

times2=[]
[times2.extend(list(i['EPOCH'])) for i in psd_files2]#append each individual cdf together 





energies2=(psd_files2[0]['SWA_EAS2_ENERGY'][0])#energy bins same at all times for all files
energies2=energies2/1000 #convert ev to kev

elevs2_raw=np.array(psd_files2[0]['SWA_EAS_ELEVATION'])
azs2_raw=np.array(psd_files2[0]['SWA_EAS_AZIMUTH'])

tops2=psd_files2[0]['SWA_EAS2_ENERGY_delta_upper']
bots2=psd_files2[0]['SWA_EAS2_ENERGY_delta_lower']


#convert the intrument angle coordinates to heliocentric

#%%initialise spice kernel

#use selenium to find relevant kernel files for selected date
#selenium setup, with chrome opening headless to allow background running
options = webdriver.ChromeOptions()
options.add_argument("--headless")    
driver = webdriver.Chrome(options=options)

#format date
new_date=date_for_spec.replace("/", "") 

#open the webpage
driver.get("https://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/ck")

#cks for the current day
cks=driver.find_elements(By.PARTIAL_LINK_TEXT,new_date)

cks_list=list()
for ck in cks:
    cks_list.append(f'ck/{ck.text}')
    
#cks for previous day, to ensure correct day is caught

date_pre_spec=str(dt.datetime.strptime(date_for_spec,'%Y/%m/%d')- dt.timedelta(days=1))
new_date=date_pre_spec[:10].replace("-", "")
cks=driver.find_elements(By.PARTIAL_LINK_TEXT,new_date)

for ck in cks:
    cks_list.append(f'ck/{ck.text}')

#now need sclks
driver.get("https://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/sclk")

#format date
new_date=date_for_spec.replace("/", "")[:6]

#cks for the current day
sclks=driver.find_elements(By.PARTIAL_LINK_TEXT,new_date)

sclk_list=list()
for sclk in sclks:
    sclk_list.append(f'sclk/{sclk.text}')
#get previous months time fiels as well, as the sclk are not often
date_pre_spec=str(dt.datetime.strptime(date_for_spec,'%Y/%m/%d')- relativedelta(months=1))
new_date=date_pre_spec[:7].replace("-", "")

#cks for the current day
sclks=driver.find_elements(By.PARTIAL_LINK_TEXT,new_date)

for sclk in sclks:
    sclk_list.append(f'sclk/{sclk.text}')

# Close the driver
driver.quit()



kernel_urls = [
    "ck/solo_ANC_soc-sc-fof-ck_20180930-21000101_V03.bc",
    "ck/solo_ANC_soc-sc-iboom-ck_20180930-21000101_V01.bc",
    "ck/solo_ANC_soc-sc-iboom-ck_20180930-21000101_V02.bc",
    "ck/solo_ANC_soc-sc-oboom-ck_20180930-21000101_V01.bc",
    "ck/solo_ANC_soc-sc-oboom-ck_20180930-21000101_V02.bc",
    "ck/solo_ANC_soc-flown-att_20211007T115622-20211009T231752_V01.bc",
    "ck/solo_ANC_soc-flown-att_20211009T143507-20211012T230729_V01.bc",
    "ck/solo_ANC_soc-default-att-stp_20200210-20301120_280_V1_00288_V01.bc",
    "fk/solo_ANC_soc-sc-fk_V09.tf",
    "fk/solo_ANC_soc-sci-fk_V08.tf",
    "ik/solo_ANC_soc-swa-ik_V00.ti",
    "ik/solo_ANC_soc-swa-ik_V01.ti",
    "ik/solo_ANC_soc-swa-ik_V02.ti",
    "ik/solo_ANC_soc-swa-ik_V03.ti",
    "lsk/naif0012.tls",
    "pck/pck00010.tpc",
    "sclk/solo_ANC_soc-sclk_20210930_V01.tsc",
    "sclk/solo_ANC_soc-sclk_20211012_V01.tsc",
    "spk/de421.bsp",
    "spk/solo_ANC_soc-orbit-stp_20200210-20301120_280_V1_00288_V01.bsp",
    "spk/solo_ANC_soc-orbit_20200210-20301120_L015_V1_00024_V01.bsp"
    
]
kernel_urls.extend(cks_list)
kernel_urls.extend(sclk_list)
kernel_urls = [f"http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/{url}"
               for url in kernel_urls]
global kernel_files
kernel_files = [cache.download(url) for url in kernel_urls]

spice.initialize(kernel_files)

# Ensure kernel paths are strings
kernel_files = [str(Path(k)) for k in kernel_files]  # Convert Path objects to strings

# Load kernels
for kernel in kernel_files:
    spiceypy.furnsh(kernel)

spice.install_frame('IAU_SUN')
spice.install_frame("SOLO_SUN_RTN")
spice.install_frame("J2000")



#%% Prepare coordinate grids for EAS2,     #eas1 does not opverlap so not included   

azimuths2, elevations2 = np.meshgrid(azs2_raw, elevs2_raw)
azimuths2_flat = azimuths2.ravel()
elevations2_flat = elevations2.ravel()

# Create SkyCoord with all az/el pairs repeated over times for EAS2
swa_eas2_looks = SkyCoord(
    np.repeat(azimuths2_flat * u.deg, len(times2)),
    np.repeat(elevations2_flat * u.deg, len(times2)),
    frame='spice_SOLO_SWA_EAS2nSCI',
    obstime=np.repeat(times2, len(azimuths2_flat))
)

# Perform batch transformation for EAS2
swa_eas2_rtn = swa_eas2_looks.transform_to("spice_SOLO_SRF")

# Reshape results to (azimuth, elevation, time) for EAS2
lon2 = swa_eas2_rtn.lon.reshape(len(azs2_raw), len(elevs2_raw), len(times2))
lat2 = swa_eas2_rtn.lat.reshape(len(azs2_raw), len(elevs2_raw), len(times2))

# Combine results for EAS2
looks2 = np.stack((lon2, lat2), axis=-1)

print("looks2 processed")
#%% remove all the unwanted bits
# Define the range of desired elevations and azimuths (e.g., min and max)
#These are the STEP FOV values in the RTN frame
elevation_min = -27
elevation_max = 27
azimuth_min = 21
azimuth_max = 49

#eas1 does not opverlap so not included   



#eas2

psd_files2=np.concatenate([i['SWA_EAS2_Data'] for i in psd_files2])#combine all files into a single time series
# Repeat for the second set

psd_files2 = np.transpose(psd_files2, (0, 1, 3, 2))
#print('EAS2 flux:')
#print((flux_psd_files2==0).any())    

    
# Repeat the same for the second set
for tind, this_t in tqdm(enumerate(psd_files2)):
    this_look = looks2[:,:,tind,:]

    # Create boolean masks for the second set
    azimuth_mask = (this_look[..., 0].value >= azimuth_min) & (this_look[..., 0].value <= azimuth_max)
    elevation_mask = (this_look[..., 1].value >= elevation_min) & (this_look[..., 1].value <= elevation_max)
    combined_mask = azimuth_mask & elevation_mask
    reshaped_mask = combined_mask.T  # Transpose to match (16, 32)
    
    
    # Apply the mask across all energy bins for the second set

    psd_files2[tind, ~reshaped_mask, :] = 0
no_pix=np.count_nonzero(reshaped_mask)
# Sum over azimuth (axis 2) and elevation (axis 1) for the second set

psd_curve2 = np.sum(np.sum(psd_files2, axis=2), axis=1)/no_pix



psd_curve_raw=np.array(psd_curve2)

#print(count_curve_raw.shape)

combo_energies=np.array(energies2)

#co_tops=(np.array(tops1)+np.array(tops2))/2
#co_bots=(np.array(bots1)+np.array(bots2))/2
#print('presaw')
#print((flux_curve_raw==0).any())
#%%sawtooth correction, only keep even index bins

psd_curve_valid=list()
valid_energies=list()
valid_widths=list()
for count, energy in enumerate(combo_energies):
    if count % 2 == 0: #if is even

        psd_curve_valid.append([i[count] for i in psd_curve_raw])
        valid_energies.append(energy)
        
        width2=tops2[count]-bots2[count]
        valid_widths.append(abs(width2))
        

#%%units and flux
psd_curve=np.array(psd_curve_valid).transpose()#swap into array then format to correct axes


psd_curve=(psd_curve*(1e3)**-6)# m conversion)

m_e=9.11e-31*10**3
flux_curve=((2*np.array(valid_energies))/(m_e**2))*psd_curve*10**-30





flux_curve=np.fliplr(flux_curve)#reverse order to ascending energy

valid_energies=np.array(valid_energies[::-1])
valid_widths=np.array(valid_widths[::-1])


#%%discard everything below low energy cutoff

flux_curve=flux_curve[:,valid_energies>low_e_cutoff]
valid_widths=np.ma.masked_array(valid_widths, valid_energies<low_e_cutoff).compressed()
valid_energies=np.ma.masked_array(valid_energies, valid_energies<low_e_cutoff).compressed()

#%%threshold
psd_series_list=list()

es_len=len(valid_energies)
for channel in np.linspace(0, es_len-1, num=es_len).astype(int):
    this_series=pd.Series(flux_curve[:,channel],times2)
    thresh_series=this_series[this_series>threshold]

    psd_series_list.append(thresh_series)

#%% PSD TS plot
#n_psd_series_list=normalize_series_list(psd_series_list)

es_len=len(valid_energies)
for channel in np.linspace(0, es_len-1, num=es_len).astype(int):
    this_series=psd_series_list[channel]
    this_series.plot(label='%.2f'%valid_energies[channel])#

#
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Electron Energy (keV)')
plt.xticks(rotation=90)

title='Solar Orbiter EAS Electron Flux Series from PSD'
    
plt.title(title)
plt.yscale("log")
plt.ylabel("Electron Flux\n"+r"(cm$^2$ sr s keV)$^{-1}$")

plt.show()




#%%generate flux data, if not already there
my_file = Path(f'eas_ts_{date_for_spec.replace("/","_")}.pkl')
if not my_file.is_file():

    eas_times_master,energies_eas_master,eas_flux_curve_master,eas_uncert_array_master,psd_ld=EAS_data_load(date_for_spec,tstart,tend,low_e_cutoff)
    # Saving the objects:
        
    with open(f'eas_ts_{date_for_spec.replace("/","_")}.pkl', 'w+b') as f:
        pickle.dump([eas_times_master,energies_eas_master,eas_flux_curve_master,eas_uncert_array_master,psd_ld], f)

#%%load flux data


# Getting back the objects:
with open(f'eas_ts_{date_for_spec.replace("/","_")}.pkl', 'rb') as f:
    eas_times_master,energies_eas_master,eas_flux_curve_master,eas_uncert_array_master,psd_ld = pickle.load(f)

eas_times,energies_eas,eas_flux_curve,eas_uncert_array=eas_times_master,energies_eas_master,eas_flux_curve_master,eas_uncert_array_master

eas_series_list=list()
eas_uncert_series_list=list()
es_len=len(energies_eas)
for channel in np.linspace(0, es_len-1, num=es_len).astype(int):
    this_series=pd.Series(eas_flux_curve[:,channel],eas_times)
    thresh_series=this_series[this_series>threshold]
    this_uncert_series=pd.Series(eas_uncert_array[:,channel],eas_times)
    
    thresh_uncert_series=this_uncert_series[this_series>threshold]
    
    eas_uncert_series_list.append(thresh_uncert_series)
    eas_series_list.append(thresh_series)
    
    
    


#n_eas_series_list=normalize_series_list(eas_series_list)
#%% DNF TS plot

es_len=len(valid_energies)
for channel in np.linspace(0, es_len-1, num=es_len).astype(int):
    this_series=eas_series_list[channel]
    this_series.plot(label='%.2f'%valid_energies[channel])#

#
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Electron Energy (keV)')
plt.xticks(rotation=90)

title='Solar Orbiter EAS Electron Flux Series from DNF'
    
plt.title(title)
plt.yscale("log")
plt.ylabel("Electron Flux\n"+r"(cm$^2$ sr s keV)$^{-1}$")

plt.show()

#%% Difference TS plot




es_len=len(valid_energies)
for channel in np.linspace(0, es_len-1, num=es_len).astype(int):
    this_series=psd_series_list[channel]-(eas_series_list[channel])
    this_series.plot(label='%.2f'%valid_energies[channel])#

#
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Electron Energy (keV)')
plt.xticks(rotation=90)

title='Difference between PSD and DNF'
    
plt.title(title)

plt.ylabel("Electron Flux\n"+r"(cm$^2$ sr s keV)$^{-1}$")

plt.show()

#%%select DNF background
bg_period=(Time(dt.datetime.strptime(bg_start,"%Y/%m/%d %H:%M:%S")),Time(dt.datetime.strptime(bg_end,"%Y/%m/%d %H:%M:%S")))
es_len=len(energies_eas)

bgd_eas_series_list=list()

for channel in np.linspace(0, es_len-1, num=es_len).astype(int):
    this_chan_series=eas_series_list[channel]

    
    this_times=np.array(this_chan_series.index)
    this_flux=this_chan_series.values
    
    bg_maska=this_times >=bg_period[0]#0 is start
    bg_maskb=this_times<=bg_period[1]
    
    #combine masks into one
    bg_mask=np.logical_and(bg_maska, bg_maskb)
    
    #time masking
    eas_pos_not_bg=[pos for pos,val in enumerate(zip(bg_mask, this_times)) if not(val[0])]
    this_times_bg=[val[1] for pos,val in enumerate(zip(bg_mask, this_times)) if (val[0])]
           
    eas_flux_curve_bg=np.delete(this_flux,eas_pos_not_bg,0)
    
    this_bg=np.mean(eas_flux_curve_bg)

    
    #EAS bg subtraction and error propagation
    bgd_eas_series_list.append(pd.Series(this_flux-this_bg,this_times))

#%%select PSD background
bg_period=(Time(dt.datetime.strptime(bg_start,"%Y/%m/%d %H:%M:%S")),Time(dt.datetime.strptime(bg_end,"%Y/%m/%d %H:%M:%S")))
es_len=len(valid_energies)

bgd_psd_series_list=list()

for channel in np.linspace(0, es_len-1, num=es_len).astype(int):
    this_chan_series=psd_series_list[channel]

    
    this_times=np.array(this_chan_series.index)
    this_flux=this_chan_series.values
    
    bg_maska=this_times >=bg_period[0]#0 is start
    bg_maskb=this_times<=bg_period[1]
    
    #combine masks into one
    bg_mask=np.logical_and(bg_maska, bg_maskb)
    
    #time masking
    psd_pos_not_bg=[pos for pos,val in enumerate(zip(bg_mask, this_times)) if not(val[0])]
    this_times_bg=[val[1] for pos,val in enumerate(zip(bg_mask, this_times)) if (val[0])]
           
    psd_flux_curve_bg=np.delete(this_flux,psd_pos_not_bg,0)
    
    this_bg=np.mean(psd_flux_curve_bg)

    
    #psd bg subtraction and error propagation
    bgd_psd_series_list.append(pd.Series(this_flux-this_bg,this_times))

    

#%%generate PSD spectrum


peak_fluxes_psd=list()

es_len=len(valid_energies)
for channel in np.linspace(0, es_len-1, num=es_len).astype(int):
    this_chan_series=bgd_psd_series_list[channel]

    this_times=np.array(this_chan_series.index)
    this_flux=this_chan_series.values
    
    maska=this_times >=integ_period[0]#0 is start
    maskb=this_times<=integ_period[1]
    
    #combine masks into one
    mask=np.logical_and(maska, maskb)
    
    #time masking
    psd_pos_not=[pos for pos,val in enumerate(zip(mask, this_times)) if not(val[0])]
    this_times_sliced=[val[1] for pos,val in enumerate(zip(mask, this_times)) if (val[0])]
           

    this_flux=np.delete(this_flux,psd_pos_not,0)
    
    
    pf=pd.DataFrame(this_flux).max(axis=0,skipna=True).item()
    peak_fluxes_psd.append(pf)
    max_pos=list(this_flux).index(pf)


#%%generate dnf spectrum


peak_fluxes_eas=list()

es_len=len(valid_energies)
for channel in np.linspace(0, es_len-1, num=es_len).astype(int):
    this_chan_series=bgd_eas_series_list[channel]

    this_times=np.array(this_chan_series.index)
    this_flux=this_chan_series.values
    
    maska=this_times >=integ_period[0]#0 is start
    maskb=this_times<=integ_period[1]
    
    #combine masks into one
    mask=np.logical_and(maska, maskb)
    
    #time masking
    eas_pos_not=[pos for pos,val in enumerate(zip(mask, this_times)) if not(val[0])]
    this_times_sliced=[val[1] for pos,val in enumerate(zip(mask, this_times)) if (val[0])]
           

    this_flux=np.delete(this_flux,eas_pos_not,0)
    
    
    pf=pd.DataFrame(this_flux).max(axis=0,skipna=True).item()
    peak_fluxes_eas.append(pf)
    max_pos=list(this_flux).index(pf)

#%%plot PSD and DNF spectra

plt.scatter(energies_eas,np.array(peak_fluxes_eas),c='r',label='DNF')
plt.scatter(valid_energies,np.array(peak_fluxes_psd)*2.55,c='g',label='PSD times 2.55')

plt.title(f'{date_for_spec}')
plt.xlabel("Energy (keV)")
plt.ylabel("Electron flux\n"+r"(cm$^2$ sr s MeV)$^{-1}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.yscale("log")
plt.xscale("log")
plt.grid()

#plot bg
#plt.scatter(energies_eas,list(eas_bg_spectrum.values()))
#plt.scatter(energy_mids_step,list(step_bg_spectrum.values()))
plt.show()
plt.close()

plt.scatter(energies_eas,np.array(peak_fluxes_eas),c='r',label='DNF')
plt.scatter(valid_energies,np.array(peak_fluxes_psd),c='g',label='PSD')

plt.title(f'{date_for_spec}')
plt.xlabel("Energy (keV)")
plt.ylabel("Electron flux\n"+r"(cm$^2$ sr s MeV)$^{-1}$")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.yscale("log")
plt.xscale("log")
plt.grid()

#plot bg
#plt.scatter(energies_eas,list(eas_bg_spectrum.values()))
#plt.scatter(energy_mids_step,list(step_bg_spectrum.values()))
plt.show()
plt.close()