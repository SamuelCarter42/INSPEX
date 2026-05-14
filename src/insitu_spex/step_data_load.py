# -*- coding: utf-8 -*-
import sys#for file path handling
import os#has general functions for file manipulation

import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk
import numpy as np #general mathematical operations
from solo_epd_loader import epd_load#module for loading SolO EPD data
import datetime as dt#handles general datetime operations
import pandas as pd #module for dataframe and time series handling




def load_early_step_data(start_time, end_time):
    #convert astropy time formats from hek to datetime formats for epd_load
    stringstart=str(start_time)
    stringend=str(end_time)
    
    # define start and end date of data to load (year, month, day):
    startdate = dt.datetime.strptime(stringstart,"%Y/%m/%d %H:%M:%S")
    enddate =  dt.datetime.strptime(stringend,"%Y/%m/%d %H:%M:%S")

    # load step data
    # sets your current wd path as where you want to save the data files:
    path = f"{os.getcwd()}/data/"

    # whether missing data files should automatically downloaded from SOAR:
    autodownload = True
    
    #import the data
    df_step, energies_step = epd_load(sensor='step', level='l2', startdate=startdate, enddate=enddate, path=path, autodownload=autodownload)

    data_step=[df_step, energies_step]#save data we need
    epd_xyz_sectors=energies_step['XYZ_Sectors']
    
    ##turning bin text to numerical values
    energy_texts=list(energies_step["Electron_Avg_Bins_Text"])

    energy_lims=list()
    energy_mids_step=list()
    energy_lims=list()
    #convert the energy bin labels, which are as text, to floats in the middle of the bin, already in KeV
    for i in energy_texts:
        if str(type(i))=="<class 'numpy.str_'>":
            string=str(i)
        else: 
            string=str(i[0])
        low=float(string.split(' - ')[0])
        high=float(string.split(' - ')[1][:-4])
        diff=(high-low)/2
        mid=low+(diff)

        #values times 1000 for MeV to keV
        energy_lims.append((low,high))
        energy_mids_step.append(mid)
       
    #correction values for the electron rates (see !!!!!!!!!! for why these corrections are neccesary)
    early_correction_table=[0.6,0.61,0.63,0.68,0.76,0.81,1.06,1.32,1.35,1.35,1.35,1.34,1.34,1.35,1.38,1.36,1.32,1.32,1.28,1.26,1.15,1.15,1.15,1.15,1.16,1.16,1.16,1.17,1.17,1.16,1.18,1.17,1.17,1.16,1.17,1.15,1.16,1.17,1.18,1.17,1.17,1.17,1.18,1.18,1.19,1.18,1.19,1.2]
         

    integ_chan_names=[f'Integral_Avg_Flux_{channel}'for channel in np.linspace(0, len(energies_step["Bins_Text"])-1, num=len(energies_step["Bins_Text"])).astype(int)]
    integ_flux_step=data_step[0][integ_chan_names]

    integ_uncert_chan_names=[f'Integral_Avg_Uncertainty_{channel}' for channel in np.linspace(0, len(energies_step["Bins_Text"])-1, num=len(energies_step["Bins_Text"])).astype(int)]
    integ_uncert_step=data_step[0][integ_uncert_chan_names]


    mag_chan_names=[f'Magnet_Avg_Flux_{channel}'for channel in np.linspace(0, len(energies_step["Bins_Text"])-1, num=len(energies_step["Bins_Text"])).astype(int)]
    mag_flux_step=data_step[0][mag_chan_names]


    mag_uncert_chan_names=[f'Magnet_Avg_Uncertainty_{channel}' for channel in np.linspace(0, len(energies_step["Bins_Text"])-1, num=len(energies_step["Bins_Text"])).astype(int)]
    mag_uncert_step=data_step[0][mag_uncert_chan_names]


    step_times_64=data_step[0].index


    step_times=step_times_64.to_pydatetime()



    integ_flux_step=integ_flux_step.to_numpy()
    mag_flux_step=mag_flux_step.to_numpy()


    step_array_raw=integ_flux_step-mag_flux_step
    #corrections
    step_array=step_array_raw.copy()

    for channel in np.linspace(0, 47, num=48).astype(int):
        #correction factor for electrons varies depending on when the data is from        
        if startdate<dt.datetime.strptime("2021/10/22 00:00:00","%Y/%m/%d %H:%M:%S"):
            correction_factor=early_correction_table[channel]
        else:
            print("Error, data should be loaded from the later data function")
        step_array[:,channel]=(step_array_raw[:,channel]*correction_factor)/1000#correction from raw unmodified counts including per keV conversion
        



    #error propagation

    integ_uncert_step=integ_uncert_step.to_numpy()
    mag_uncert_step=mag_uncert_step.to_numpy()


    integ_uncert_step_sq=integ_uncert_step**2
    mag_uncert_step_sq=mag_uncert_step**2

    step_uncert_array_raw=np.sqrt(integ_uncert_step_sq+mag_uncert_step_sq)
    
    #uncert must be corrected too
    step_uncert_array=step_uncert_array_raw.copy()
    for channel in np.linspace(0, 47, num=48).astype(int):
        #correction factor for electrons varies depending on when the data is from        
        if startdate<dt.datetime.strptime("2021/10/22 00:00:00","%Y/%m/%d %H:%M:%S"):
            correction_factor=early_correction_table[channel]
        else:
            print("Error, data should be loaded from the later data function")


        step_uncert_array[:,channel]=(step_uncert_array_raw[:,channel]*correction_factor)/1000#conversion to per keV#correction from raw unmodified counts 


    
    #slice data down to range requested from full days
    mask=(step_times>=startdate) & (step_times<enddate )
    step_times=step_times[mask]
    step_array=step_array[mask,:]
    step_uncert_array=step_uncert_array[mask,:]
    
    
    
    return step_times,energy_mids_step,step_array,step_uncert_array,epd_xyz_sectors,np.array(energy_lims)




def load_late_step_data(start_time, end_time):
    #convert astropy time formats from hek to datetime formats for epd_load
    stringstart=str(start_time)
    stringend=str(end_time)
    
    # define start and end date of data to load (year, month, day):
    startdate = dt.datetime.strptime(stringstart,"%Y/%m/%d %H:%M:%S")
    enddate =  dt.datetime.strptime(stringend,"%Y/%m/%d %H:%M:%S")

    # load step data
    # sets your current wd path as where you want to save the data files:
    path = f"{os.getcwd()}/data/"

    # whether missing data files should automatically downloaded from SOAR:
    autodownload = True
    
    #import the data
    df_step, energies_step = epd_load(sensor='step', level='l2', 
                                      startdate=startdate, enddate=enddate,
                                      path=path, autodownload=autodownload)
    
    data_step=[df_step, energies_step]#save data we need
    
    ##turning bin text to numerical values
    energy_texts=list(energies_step["Electron_Bins_Text"])
    epd_xyz_sectors=energies_step['XYZ_Pixels']

    energy_mids_step=list()
    energy_lims=list()
    #convert the energy bin labels, which are as text, to floats in the middle of the bin, in KeV
    for i in energy_texts:
        if str(type(i))=="<class 'numpy.str_'>":
            string=str(i)
        else: 
            string=str(i[0])
        low=float(string.split(' - ')[0])*1000
        high=float(string.split(' - ')[1][:-4])*1000
        diff=(high-low)/2
        mid=low+(diff)
        #values times 1000 for MeV to keV
       
        energy_mids_step.append(mid)
        energy_lims.append((low,high))
    #read correction table from the energy file
    correction_table=energies_step['Electron_Flux_Mult']['Electron_Avg_Flux_Mult']

    integ_chan_names=[f'Integral_Avg_Flux_{channel}'for channel in np.linspace(0, len(energies_step["Electron_Bins_Text"])-1, num=len(energies_step["Electron_Bins_Text"])).astype(int)]
    integ_flux_step=data_step[0][integ_chan_names]

    integ_uncert_chan_names=[f'Integral_Avg_Uncertainty_{channel}' for channel in np.linspace(0, len(energies_step["Electron_Bins_Text"])-1, num=len(energies_step["Electron_Bins_Text"])).astype(int)]
    integ_uncert_step=data_step[0][integ_uncert_chan_names]


    mag_chan_names=[f'Magnet_Avg_Flux_{channel}'for channel in np.linspace(0, len(energies_step["Electron_Bins_Text"])-1, num=len(energies_step["Electron_Bins_Text"])).astype(int)]
    mag_flux_step=data_step[0][mag_chan_names]


    mag_uncert_chan_names=[f'Magnet_Avg_Uncertainty_{channel}' for channel in np.linspace(0, len(energies_step["Electron_Bins_Text"])-1, num=len(energies_step["Electron_Bins_Text"])).astype(int)]
    mag_uncert_step=data_step[0][mag_uncert_chan_names]


    step_times_64=data_step[0].index


    step_times=step_times_64.to_pydatetime()



    integ_flux_step=integ_flux_step.to_numpy()
    mag_flux_step=mag_flux_step.to_numpy()


    step_array_raw=integ_flux_step-mag_flux_step

    #corrections
    step_array=step_array_raw.copy()

    for channel in np.linspace(0, 31, num=32).astype(int):
        correction_factor=correction_table[channel]
        step_array[:,channel]=(step_array_raw[:,channel]*correction_factor)/1000#correction from raw unmodified counts including per keV conversion
    
    
    #error propagation

    integ_uncert_step=integ_uncert_step.to_numpy()
    mag_uncert_step=mag_uncert_step.to_numpy()


    integ_uncert_step_sq=integ_uncert_step**2
    mag_uncert_step_sq=mag_uncert_step**2

    step_uncert_array_raw=np.sqrt(integ_uncert_step_sq+mag_uncert_step_sq)
    
    #uncert must be corrected too
    step_uncert_array=step_uncert_array_raw.copy()
    for channel in np.linspace(0, 31, num=32).astype(int):
        #correction factor for electrons varies depending on when the data is from        
        correction_factor=correction_table[channel]
        step_uncert_array[:,channel]=(step_uncert_array_raw[:,channel]*correction_factor)/1000#conversion to per keV#correction from raw unmodified counts

    #slice data down to range requested from full days
    mask=(step_times>=startdate) & (step_times<enddate )
    step_times=step_times[mask]
    step_array=step_array[mask,:]
    step_uncert_array=step_uncert_array[mask,:]
    
    
    return step_times,energy_mids_step,step_array,step_uncert_array,epd_xyz_sectors,np.array(energy_lims)
