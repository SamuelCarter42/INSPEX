
import sys#for file path handling
import os#has general functions for file manipulation

from . import stereo_data_load_calib

import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk

import numpy as np #general mathematical operations


import datetime as dt#handles general datetime operations
import pandas as pd #module for dataframe and time series handling
import scipy #for reading in idl saves and other various functions

#for calling the IDL code required to calibrate the STEREO data and convert it to flux




#%%stereo data load function


def stereo_data_load(start_time, end_time):
    
    date= dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S").strftime("%d-%m-%Y")
    
    this_folder=os.path.realpath(os.getcwd())
    this_folder=str(this_folder).replace('\\', '/')
    #run the function that calls the IDL load and calibration function
    #for calling the IDL code required to calibrate the STEREO data and convert it to flux
    stereo_data_load_calib(start_time,end_time,date,this_folder)
    
    
    
    
    
    
    #load in data from the IDl save files
    #read D0 data
    path=os.path.join(fr'{this_folder}/data/Stereo/Processed_data/STA_STE_D0_{date}_range.sav')
    STE_D0_sav=scipy.io.readsav(path)
    
    STE_D0=STE_D0_sav['structure0']#pulls array from .sav structure
    STE_D0_times_unix=STE_D0[0][0]
    
    #read D1 data
    path=os.path.join(fr'{this_folder}/data/Stereo/Processed_data/STA_STE_D1_{date}_range.sav')
    STE_D1_sav=scipy.io.readsav(path)
    
    STE_D1=STE_D1_sav['structure1']#pulls array from .sav structure
    STE_D1_times_unix=STE_D1[0][0]
    
    
    #read D2 data
    path=os.path.join(fr'{this_folder}/data/Stereo/Processed_data/STA_STE_D2_{date}_range.sav')
    STE_D2_sav=scipy.io.readsav(path)
    
    STE_D2=STE_D2_sav['structure2']#pulls array from .sav structure
    STE_D2_times_unix=STE_D2[0][0]
    
    #read D3 data
    path=os.path.join(fr'{this_folder}/data/Stereo/Processed_data/STA_STE_D3_{date}_range.sav')
    STE_D3_sav=scipy.io.readsav(path)
    
    STE_D3=STE_D3_sav['structure3']#pulls array from .sav structure
    STE_D3_times_unix=STE_D3[0][0]
    
    
    #for D0
    #converts times to gregorian
    STE_D0_times=list()
    for i in STE_D0_times_unix: STE_D0_times.append(dt.datetime.fromtimestamp(i))

    #retreive energies-all the same at all times, so will just take first value
    STE_D0_energies=STE_D0[0][2][:,0]
  
    STE_D0_flux=STE_D0[0][1]

    STE_D0_flux=STE_D0_flux.byteswap().view(STE_D0_flux.dtype.newbyteorder())# force native byteorder so that numpy array works with pandas

    #for D1
    #converts times to gregorian
    STE_D1_times=list()
    for i in STE_D1_times_unix: STE_D1_times.append(dt.datetime.fromtimestamp(i))

    #retreive energies-all the same at all times, so will just take first value- or will we just pull for the time?
    STE_D1_energies=STE_D1[0][2][:,0]

    STE_D1_flux=STE_D1[0][1]

    STE_D1_flux=STE_D1_flux.byteswap().view(STE_D1_flux.dtype.newbyteorder())# force native byteorder so that numpy array works with pandas

    #for D2
    #converts times to gregorian
    STE_D2_times=list()
    for i in STE_D2_times_unix: STE_D2_times.append(dt.datetime.fromtimestamp(i))

    #retreive energies-all the same at all times, so will just take first value- or will we just pull for the time?
    STE_D2_energies=STE_D2[0][2][:,0]

    STE_D2_flux=STE_D2[0][1]

    STE_D2_flux=STE_D2_flux.byteswap().view(STE_D2_flux.dtype.newbyteorder())# force native byteorder so that numpy array works with pandas


    #for D3
    #converts times to gregorian
    STE_D3_times=list()
    for i in STE_D3_times_unix: STE_D3_times.append(dt.datetime.fromtimestamp(i))

    #retreive energies-all the same at all times, so will just take first value- or will we just pull for the time?
    STE_D3_energies=STE_D3[0][2][:,0]

    STE_D3_flux=STE_D3[0][1]

    STE_D3_flux=STE_D3_flux.byteswap().view(STE_D3_flux.dtype.newbyteorder())# force native byteorder so that numpy array works with pandas

    #take average of the energy bins for each detector
    STE_Combo_Energies=np.array(list(STE_D0_energies))+np.array(list(STE_D1_energies))+np.array(list(STE_D2_energies))+np.array(list(STE_D3_energies))
    STE_Combo_Energies=STE_Combo_Energies/4#average
    STE_Combo_Energies=STE_Combo_Energies/1000 #convert to keV
    #sum the count arrays
    STE_combo_flux=STE_D0_flux+STE_D1_flux+STE_D2_flux+STE_D3_flux
    STE_combo_flux=(STE_combo_flux.transpose())*1000
    
    #propagate the errors
    STE_D0_Flux_Errors=STE_D0[0][6]*1000#convert per ev to per kev
    STE_D1_Flux_Errors=STE_D1[0][6]*1000#convert per ev to per kev
    STE_D2_Flux_Errors=STE_D2[0][6]*1000#convert per ev to per kev
    STE_D3_Flux_Errors=STE_D3[0][6]*1000#convert per ev to per kev
    STE_Combo_Errors=np.sqrt((np.array(list(STE_D0_Flux_Errors))**2)+(np.array(list(STE_D1_Flux_Errors))**2)+(np.array(list(STE_D2_Flux_Errors))**2)+(np.array(list(STE_D3_Flux_Errors))**2))
    STE_Combo_Errors=STE_Combo_Errors.transpose()

    #we will use the D0 times for all
    return(np.array(STE_D0_times),STE_Combo_Energies,STE_combo_flux,STE_Combo_Errors)