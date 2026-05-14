#%%Initial set up
import sys#for file path handling
import os#has general functions for file manipulation


import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk

import numpy as np #general mathematical operations

#%%average background spectrum calculator

def avg_bg_calc(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime):
    bg_maska=time_series_time >=bg_mintime#0 is start
    bg_maskb=time_series_time<=bg_maxtime

    #combine masks into one
    bg_mask=np.logical_and(bg_maska, bg_maskb)

    #time masking
    pos_not_bg=[pos for pos,val in enumerate(zip(bg_mask, time_series_time)) if not(val[0])]
    times_bg=[val[1] for pos,val in enumerate(zip(bg_mask, time_series_time)) if (val[0])]
           
    array_bg=np.delete(time_series_data,pos_not_bg,0)

    uncert_array_sliced_bg=np.delete(time_series_uncert,pos_not_bg,0)


    bg_spectrum=dict()#mean of every channel over selected background period
    bg_spectrum_uncert=dict()
    for channel in np.linspace(0, len(time_series_energies)-1, num=len(time_series_energies)).astype(int):#loop through the time series of each energy channel
        bg_spectrum[channel]=np.mean(array_bg[:,channel])
        bg_spectrum_uncert[channel]=np.sqrt(sum([err**2 for err in uncert_array_sliced_bg[:,channel]]))
    return bg_spectrum, bg_spectrum_uncert


#
#%% spectrum calculator for peak flux
def peak_flux_spec_gen(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime,spec_mintime,spec_maxtime):
    bg_spec, bg_spec_uncert=avg_bg_calc(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime)
    
    maska=time_series_time >=spec_mintime# is start
    maskb=time_series_time<=spec_maxtime

    #combine masks into one
    mask=np.logical_and(maska, maskb)

    #time masking
    TS_pos_not=[pos for pos,val in enumerate(zip(mask, time_series_time)) if not(val[0])]
    TS_times_sliced=[val[1] for pos,val in enumerate(zip(mask, time_series_time)) if (val[0])]
           
    TS_array_sliced=np.delete(time_series_data,TS_pos_not,0)

    TS_uncert_array_sliced=np.delete(time_series_uncert,TS_pos_not,0)
    
    #background subtraction
        
    TS_array_subtracted=TS_array_sliced.copy()
    TS_uncert_array_subtracted=TS_uncert_array_sliced.copy()
    for channel in np.linspace(0, len(time_series_energies)-1, num=len(time_series_energies)).astype(int):#loop through the time series of each energy channel
        this_chan_bg=bg_spec[channel]
        TS_array_subtracted[:,channel]=TS_array_sliced[:,channel]-this_chan_bg
        
        TS_uncert_array_subtracted[:,channel]=np.sqrt(np.array([err**2 for err in TS_uncert_array_sliced[:,channel]])+ bg_spec_uncert[channel]**2)
        
    
    
    
    #calculate the peak flux spectra
    TS_flux=dict()
    TS_flux_uncert=dict()
    for pos,channel in enumerate(time_series_energies):
        if channel>100:continue #need to limit energy range to 100 kev
        this_chan=list(TS_array_subtracted[:,pos])
        #breakpoint()
        this_e=channel
#        print(this_chan)
        TS_flux[this_e]=max(this_chan)
        
        max_pos=list(this_chan).index(max(this_chan))
        this_uncert=TS_uncert_array_subtracted[max_pos,pos]
        
        TS_flux_uncert[this_e]=this_uncert
    
    return list(TS_flux.values()),list(TS_flux_uncert.values())

#%%spectrum calculator for Fluence


def fluence_spec_gen(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime,spec_mintime,spec_maxtime):
    bg_spec, bg_spec_uncert=avg_bg_calc(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime)#background generating function
    
    maska=time_series_time >=spec_mintime# is start
    maskb=time_series_time<=spec_maxtime

    #combine masks into one
    mask=np.logical_and(maska, maskb)

    #time masking
    TS_pos_not=[pos for pos,val in enumerate(zip(mask, time_series_time)) if not(val[0])]
    TS_times_sliced=[val[1] for pos,val in enumerate(zip(mask, time_series_time)) if (val[0])]
           
    TS_array_sliced=np.delete(time_series_data,TS_pos_not,0)

    TS_uncert_array_sliced=np.delete(time_series_uncert,TS_pos_not,0)
    
    #background subtraction
        
    TS_array_subtracted=TS_array_sliced.copy()
    TS_uncert_array_subtracted=TS_uncert_array_sliced.copy()
    for channel in np.linspace(0, len(time_series_energies)-1, num=len(time_series_energies)).astype(int):#loop through the time series of each energy channel
        this_chan_bg=bg_spec[channel]
        TS_array_subtracted[:,channel]=TS_array_sliced[:,channel]-this_chan_bg
        
        TS_uncert_array_subtracted[:,channel]=np.sqrt(np.array([err**2 for err in TS_uncert_array_sliced[:,channel]])+ bg_spec_uncert[channel]**2)
        
        
        
        
        
    #calculate the fluence spectra
    TS_fluence=dict()
    TS_fluence_uncert=dict()
    for pos,channel in enumerate(time_series_energies):
        if channel>100:continue #need to limit energy range to 100 kev
        this_chan=TS_array_subtracted[:,pos]
        this_e=channel
        TS_fluence[this_e]=sum(this_chan)
        this_uncert_list=TS_uncert_array_subtracted[:,pos]
        this_uncert=np.sqrt(sum([i**2 for i in this_uncert_list]))
        TS_fluence_uncert[this_e]=this_uncert
        
        
        
    return list(TS_fluence.values()),list(TS_fluence_uncert.values())