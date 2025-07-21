#%%Initial set up
import sys#for file path handling
import os#has general functions for file manipulation

sys.path.append(f"{os.getcwd()}/Dependencies")#ensures that dependencies folder is available at point that modules are loaded

import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk
import lmfit #this module contains the functions for the curve fitting
import numpy as np #general mathematical operations
from scipy.special import erf #imports an erf function for use in some of the fitting operations
from matplotlib import pyplot as plt #general plotting operations
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)#allows plotting to a tkinter window
from solo_epd_loader import epd_load#module for loading SolO EPD data
import datetime as dt#handles general datetime operations
import pandas as pd #module for dataframe and time series handling
import scipy #for reading in idl saves and other various functions
import subprocess #for running IDL codes
import stereo_idl_caller#for calling the IDL code required to calibrate the STEREO data and convert it to flux
import re#for handling regexs to validate inputs
import random as rn#for random number and choice utility, particularly in uncertainty estimation
from tqdm import tqdm #for tracking progress of long iterables
from bootstrap_for_uncert_v7 import uncert_bootstrap
import math
import numdifftools
import random#needed for random numbers
import pickle #saves vars
import threading  #required to allow gui updates during code

plt.rcParams['figure.dpi'] = 200 #set the dpi for the plots. this can be tuned to improve figure quality

bootstrap_n=30000 #this controls the number of iterations used for the bootstrap to get uncerts, need min 30 for central limit theorem

#initialise some windows as nones so they exist
global fit_window
global preview_window
global resid_window
fit_window=None
preview_window=None
resid_window=None

#setting the fit methods
method_1='basinhopping'#'dual_annealing'#
method_2='lbfgsb'

global redchi
redchi=0
global bic
bic=0
#%% early step data load function

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
    
    ##turning bin text to numerical values
    energy_texts=list(energies_step["Electron_Avg_Bins_Text"])


    energy_mids_step=list()
    #convert the energy bin labels, which are as text, to floats in the middle of the bin, already in KeV
    for i in energy_texts:
        string=i[0]
        low=float(string.split(' - ')[0])
        high=float(string.split(' - ')[1][:-4])
        diff=(high-low)/2
        mid=low+(diff)
        #values times 1000 for MeV to keV
       
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
        step_array[:][channel]=(step_array_raw[:][channel]*correction_factor)/1000#correction from raw unmodified counts including per keV conversion
        



    #error propagation

    integ_uncert_step=integ_uncert_step.to_numpy()
    mag_uncert_step=mag_uncert_step.to_numpy()


    integ_uncert_step_sq=integ_uncert_step**2
    mag_uncert_step_sq=mag_uncert_step**2

    step_uncert_array_raw=np.sqrt(integ_uncert_step_sq+mag_uncert_step_sq)/1000#conversion to per keV
    
    #uncert must be corrected too
    step_uncert_array=step_uncert_array_raw.copy()
    for channel in np.linspace(0, 47, num=48).astype(int):
        #correction factor for electrons varies depending on when the data is from        
        if startdate<dt.datetime.strptime("2021/10/22 00:00:00","%Y/%m/%d %H:%M:%S"):
            correction_factor=early_correction_table[channel]
        else:
            print("Error, data should be loaded from the later data function")
        step_uncert_array[:][channel]=(step_uncert_array_raw[:][channel]*correction_factor)#correction from raw unmodified counts 
        
    
    
    
    
    
    return step_times,energy_mids_step,step_array,step_uncert_array
#%% load post-recalibration step data
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


    energy_mids_step=list()
    #convert the energy bin labels, which are as text, to floats in the middle of the bin, in KeV
    for i in energy_texts:
        string=i[0]
        low=float(string.split(' - ')[0])*1000
        high=float(string.split(' - ')[1][:-4])*1000
        diff=(high-low)/2
        mid=low+(diff)
        #values times 1000 for MeV to keV
       
        energy_mids_step.append(mid)

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
        step_array[:][channel]=(step_array_raw[:][channel]*correction_factor)/1000#correction from raw unmodified counts including per keV conversion
    
    
    #error propagation

    integ_uncert_step=integ_uncert_step.to_numpy()
    mag_uncert_step=mag_uncert_step.to_numpy()


    integ_uncert_step_sq=integ_uncert_step**2
    mag_uncert_step_sq=mag_uncert_step**2

    step_uncert_array_raw=np.sqrt(integ_uncert_step_sq+mag_uncert_step_sq)/1000#conversion to per keV
    
    #uncert must be corrected too
    step_uncert_array=step_uncert_array_raw.copy()
    for channel in np.linspace(0, 31, num=32).astype(int):
        #correction factor for electrons varies depending on when the data is from        
        correction_factor=correction_table[channel]
        step_uncert_array[:][channel]=(step_uncert_array_raw[:][channel]*correction_factor)#correction from raw unmodified counts
        
    
    
    return step_times,energy_mids_step,step_array,step_uncert_array
#%%stereo data load function


def stereo_data_load(start_time, end_time):
    
    date= dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S").strftime("%d-%m-%Y")
    
    this_folder=os.path.realpath(os.getcwd())
    this_folder=str(this_folder).replace('\\', '/')
    #run the function that calls the IDL load and calibration function
    stereo_idl_caller.stereo_data_load_calib(start_time,end_time,date,this_folder)
    
    
    
    
    
    
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

#%%functions for fitting
k_B=8.617333262*(10**-8) # Boltzmann constant in keV per kelvin
G=6.67430e-11#in N m^2 kg^-2
m_sun=1.989e30 #solar mass in kg
r_sun=6.957e8#solar radius in m


def therm_func(x,therm_amp,T,alpha): #defines the thermal function's form
    x=np.array(x)
    #alpha=1#forces energy index to be 1
    y_therm=therm_amp*(x**alpha)*np.exp(-x/(k_B*T))
    return (y_therm)

def lin_func(x,A,B): #one of the power laws that makes up the broken power law
    y_lin_1=(A*(x**B))
    return y_lin_1

def lin_func2(x,A2,B2):#one of the power laws that makes up the broken power law
    y_lin_2=(A2*(x**B2))
    return y_lin_2


def broken_power_law(x,x1,A,B,A2,B2): #defines a broken power law to fit, like the thick target approx.
        
    if type(x)==int:
        xlo=1 if x<x1 else 0 #below x1
        xhi=1 if x>=x1 else 0#above x1  
    else:
        x=np.array(x)
        xlo=[ 1 if x_i<x1 else 0 for x_i in x] #below x1
        xhi=[ 1 if x_i>=x1 else 0 for x_i in x]#above x1    
    y_bpl=(xlo*lin_func(x,A,B))+(xhi*lin_func2(x,A2,B2))
    return y_bpl

def gauss_func(x,gauss_amp,gauss_centre,sigma): #defines a gaussian function that can be added
    x=np.array(x)
    y_gauss=gauss_amp*np.exp((-(x-gauss_centre)**2)/(2*sigma**2))
    return y_gauss

def power_func(x,A_sing,B_sing): #defines a simgle power law that can be added
    x=np.array(x)
    y_pow=(A_sing*(x**B_sing))
    return y_pow

def kappa_func(x, A_k, T_k, m_i, n_i, kappa):

    v_th=np.sqrt((2*x)/m_i)
    w=np.sqrt(((2*kappa-3)*k_B*T_k)/(kappa*m_i))
    term1=((v_th**2)/m_i)*(n_i/(2*np.pi*(kappa*w**2)**(3/2)))
    term2=math.gamma(kappa+1)/(math.gamma(kappa-1/2)*math.gamma(3/2))
    term3=(1+((v_th**2)/(kappa*(w**2))))**-(kappa+1)
    
    
    y_kappa=A_k*term1*term2*term3
    
    
    
    return y_kappa
    
#a combined bpl and thermal. parameters have _c to indicate combined
def bpl_and_therm_func(x,amp_c,T_c,alpha_c,x0_c,x1_c,B_c,B2_c):
    x=np.array(x)
    

    xmid=[ 1 if x_i<x1_c  else 0 for x_i in x] #below x1, above x0 'and x_i>=x0_c'
    xhi=[ 1 if x_i>=x1_c else 0 for x_i in x]#above x1    
    
    y_therm=(amp_c*(x**alpha_c)*np.exp(-x/(k_B*T_c)))
    
    
    
    A=amp_c*(x0_c**(alpha_c-B_c))*np.exp(-x0_c/(k_B*T_c))
    A2=A * x1_c**(B_c-B2_c)
    

    y_bpl=(xmid*lin_func(x,A,B_c))+(xhi*lin_func2(x,A2,B2_c))
    
    
    y_combined=y_therm+y_bpl
    
    return y_combined

# a double thermal curve
def double_therm_func(x,therm_amp,T,alpha,therm_amp2,T2,alpha2): #defines the thermal function's form
    x=np.array(x)
    #alpha=1#forces energy index to be 1
    y_therm=therm_amp*(x**alpha)*np.exp(-x/(k_B*T))
    y_therm2=therm_amp2*(x**alpha2)*np.exp(-x/(k_B*T2))
    
    return (y_therm+y_therm2)


#a triple power law

def triple_power_law(x,x1,x2,A,B,A2,B2,A3,B3):
    
    if type(x)==int:
        xlo=1 if x<x1 else 0 #below x1
        xmid=1  if (x>=x1 and x<=x2) else 0 #between x1 and x2
        xhi=1 if x>x2 else 0#above x2  
    else:
        x=np.array(x)
        xlo=[ 1 if x_i<x1 else 0 for x_i in x] #below x1
        xmid =[ 1 if (x_i>=x1 and x_i<=x2) else 0 for x_i in x] #between x1 and x2
        xhi=[ 1 if x_i>=x2 else 0 for x_i in x]#above x2    
    
    
    y_tpl=(xlo*lin_func(x,A,B))+(xmid*lin_func2(x,A2,B2))+(xhi*lin_func2(x,A3,B3))
    return y_tpl

def quad_power_law(x,x1,x2,x3,A,B,A2,B2,A3,B3,A4,B4):
    
    if type(x)==int:
        xlo=1 if x<x1 else 0 #below x1
        xmid1=1  if (x>=x1 and x<=x2) else 0 #between x1 and x2
        xmid2=1  if (x>x2 and x<=x3) else 0 #between x2 and x3
        xhi=1 if x>x3 else 0#above x3  
    else:
        x=np.array(x)
        xlo=[ 1 if x_i<x1 else 0 for x_i in x] #below x1
        xmid1 =[ 1 if (x_i>=x1 and x_i<=x2) else 0 for x_i in x] #between x1 and x2
        xmid2 =[ 1 if (x_i>x2 and x_i<=x3) else 0 for x_i in x] #between x2 and x3
        xhi=[ 1 if x_i>x3 else 0 for x_i in x]#above x3    
    
    
    y_qpl=(xlo*lin_func2(x,A,B))+(xmid1*lin_func2(x,A2,B2))+(xmid2*lin_func2(x,A3,B3))+(xhi*lin_func2(x,A4,B4))
    return y_qpl

def quint_power_law(x,x1,x2,x3,x4,A,B,A2,B2,A3,B3,A4,B4,A5,B5):
    
    if type(x)==int:
        xlo=1 if x<x1 else 0 #below x1
        xmid1=1  if (x>=x1 and x<=x2) else 0 #between x1 and x2
        xmid2=1  if (x>x2 and x<=x3) else 0 #between x2 and x3
        xmid3=1  if (x>x3 and x<=x4) else 0 #between x3 and x4
        xhi=1 if x>x4 else 0#above x4  
    else:
        x=np.array(x)
        xlo=[ 1 if x_i<x1 else 0 for x_i in x] #below x1
        xmid1 =[ 1 if (x_i>=x1 and x_i<=x2) else 0 for x_i in x] #between x1 and x2
        xmid2 =[ 1 if (x_i>x2 and x_i<=x3) else 0 for x_i in x] #between x2 and x3
        xmid3 =[ 1 if (x_i>x3 and x_i<=x4) else 0 for x_i in x] #between x3 and x4
        xhi=[ 1 if x_i>x4 else 0 for x_i in x]#above x4   
    
    
    y_5pl=(xlo*lin_func2(x,A,B))+(xmid1*lin_func2(x,A2,B2))+(xmid2*lin_func2(x,A3,B3))+(xmid3*lin_func2(x,A4,B4))+(xhi*lin_func2(x,A5,B5))
    return y_5pl

#%%  residuals and fitting
def resid_calc(pars,x_data,y_data,uncert,header): #defines the calculator for residuals that the fitting function needs to minimise
    #unpack params object
    parvals=pars.valuesdict() #converts the parameters to a dictionary form

    #calculate values
    calcd_vals=test_func(x_data,parvals,header)#uses the defined test function to get the calculated values
    #calc resids
    resids=(np.array(calcd_vals)-np.array(y_data))/(np.array(uncert)*1) #calculates the residuals
    return list(resids)

def fitting(header,init,vary,minval,maxval,x_data,y_data,uncert,fitmin,fitmax,spec_type): #defines our fitting process
    
    global fit_window
    global preview_window
    global resid_window
    if fit_window is not None:# and fit_window.winfo_exists():
        #close any open figues
        #fit_window.destroy()
        fit_window=None
        
    #open a window, around which the main program section is built
    
    plot_wind_size=(4,3.5)#define the window size for the plots
    
    fit_window=tk.Toplevel()
    fit_window.title('Fit window')
    
    
    
    if resid_window is not None:# and resid_window.winfo_exists():
        #close any open figues
        #resid_window.destroy()
        resid_window=None
    
    if preview_window is not None:# and preview_window.winfo_exists():
        #close any open figues
        #preview_window.destroy()
        preview_window=None
    
    #set range to user defined fitting limits
    x_data_sliced=list()
    y_data_sliced=list()
    uncert_sliced=list()
    for pos,E in enumerate(x_data):
      if E>=fitmin  and E<=fitmax:
          x_data_sliced.append(E)
          y_data_sliced.append(y_data[pos])
          uncert_sliced.append(uncert[pos])

    #build test function according to the user set options
    global test_func
    def test_func(x,parvals,header): # this function is the one we are trying to fit to the data
        #print('testtest')
        #if x data list, create y data as list too. else if x is array, use array for y
        if type(x)==list:
            y=np.zeros(len(x))
            x=np.array(x)
        else:
            y=0
        
         
        #defining what parameters to read in, depending on the header definiions of the function to be fitted
        if header[9]=='1':# ie if the broken power law is present
            
            x1=parvals["x1"]
            A=parvals["A"]
            B=parvals["B"]
            A2=parvals["A2"]
            B2=parvals["B2"]   
            y+=broken_power_law(x,x1,A,B,A2,B2)
        
        
        
        if header[28]=='1':#ie if the therm func is present 
            amp=parvals["amp"]
            T=parvals["T"]
            alpha=parvals["alpha"]
            y+=therm_func(x,amp,T,alpha)
        
        if header[42]=='1': #ie if gaussian is present
            gauss_amp=parvals["gauss_amp"]
            gauss_centre=parvals["gauss_centre"]
            sigma=parvals["sigma"]
            y+=gauss_func(x, gauss_amp, gauss_centre, sigma)
            
        if header[56]=='1': #ie if single power law is present
            A_sing=parvals["A_sing"]
            B_sing=parvals["B_sing"]
            y+=power_func(x, A_sing, B_sing)
            
        if header[70]=='1': #ie if kappa func is present
            A_k=parvals["A_k"]
            T_k=parvals["T_k"]
            m_i=parvals["m_i"]
            n_i=parvals["n_i"]
            kappa=parvals["kappa"]
            y+=kappa_func(x, A_k, T_k, m_i, n_i, kappa)
            
        if header[92]=='1':
            amp_c=parvals['amp_c']
            T_c=parvals['T_c']
            alpha_c=parvals['alpha_c']
            x0_c=parvals['x0_c']
            x1_c=parvals['x1_c']
            B_c=parvals['B_c']
            B2_c=parvals['B2_c']
            
            y+=bpl_and_therm_func(x,amp_c,T_c,alpha_c,x0_c,x1_c,B_c,B2_c)
        
        if header[118]=='1':#ie if the double therm func is present 
            amp_d_1=parvals["amp_d_1"]
            T_d_1=parvals["T_d_1"]
            alpha_d_1=parvals["alpha_d_1"]
            amp_d_2=parvals["amp_d_2"]
            T_d_2=parvals["T_d_2"]
            alpha_d_2=parvals["alpha_d_2"]
            
            y+=double_therm_func(x,amp_d_1,T_d_1,alpha_d_1,amp_d_2,T_d_2,alpha_d_2)
        
        if header[130]=='1':# ie if the triple power law is present
            
            x1_tpl=parvals["x1_tpl"]
            x2_tpl=parvals["x2_tpl"]
            A_tpl=parvals["A_tpl"]
            B_tpl=parvals["B_tpl"]
            A2_tpl=parvals["A2_tpl"]
            B2_tpl=parvals["B2_tpl"]   
            A3_tpl=parvals["A3_tpl"]
            B3_tpl=parvals["B3_tpl"] 
            y+=triple_power_law(x,x1_tpl,x2_tpl,A_tpl,B_tpl,A2_tpl,B2_tpl,A3_tpl,B3_tpl)
            
            
        if header[142]=='1':# ie if the quad power law is present
            
            x1_qpl=parvals["x1_qpl"]
            x2_qpl=parvals["x2_qpl"]
            x3_qpl=parvals["x3_qpl"]
            A_qpl=parvals["A_qpl"]
            B_qpl=parvals["B_qpl"]
            A2_qpl=parvals["A2_qpl"]
            B2_qpl=parvals["B2_qpl"]   
            A3_qpl=parvals["A3_qpl"]
            B3_qpl=parvals["B3_qpl"] 
            A4_qpl=parvals["A4_qpl"]
            B4_qpl=parvals["B4_qpl"]
            y+=quad_power_law(x,x1_qpl,x2_qpl,x3_qpl,A_qpl,B_qpl,A2_qpl,B2_qpl,A3_qpl,B3_qpl,A4_qpl,B4_qpl)
        
        if header[159]=='1':# ie if the 5 power law is present
                
                x1_5pl=parvals["x1_5pl"]
                x2_5pl=parvals["x2_5pl"]
                x3_5pl=parvals["x3_5pl"]
                x4_5pl=parvals["x4_5pl"]
                A_5pl=parvals["A_5pl"]
                B_5pl=parvals["B_5pl"]
                A2_5pl=parvals["A2_5pl"]
                B2_5pl=parvals["B2_5pl"]   
                A3_5pl=parvals["A3_5pl"]
                B3_5pl=parvals["B3_5pl"] 
                A4_5pl=parvals["A4_5pl"]
                B4_5pl=parvals["B4_5pl"]
                A5_5pl=parvals["A5_5pl"]
                B5_5pl=parvals["B5_5pl"]
                y+=quint_power_law(x,x1_5pl,x2_5pl,x3_5pl,x4_5pl,A_5pl,B_5pl,A2_5pl,B2_5pl,A3_5pl,B3_5pl,A4_5pl,B4_5pl,A5_5pl,B5_5pl)
    
        return y
    
    #define params with bounds and initial values
    params=lmfit.Parameters()
    
    #adding params depending on which functions user has selected
    
    #addwithtuples:(NAME VALUE VARY MIN MAX EXPR BRUTE_STEP) 
    
    
    if header[28]=='1':#ie if the therm func is present                   
        params.add_many(('amp',init['amp'],vary['amp'],minval['amp'],maxval['amp'],None,None),
                    ('T',init['T'] ,vary['T'],minval['T'],maxval['T'],None,None),
                    ('alpha',init['alpha'],vary['alpha'],minval['alpha'],maxval['alpha'],None,None))
    
    
    if header[9]=='1':# ie if the broken power law is present
    
        params.add_many(('x1',init['x1'],vary['x1'],minval['x1'],maxval['x1'],None,None),
                       ('B',init['B'] ,vary['B'],minval['B'],maxval['B'],None,None), 
                      ('B2',init['B2'],vary['B2'],minval['B2'],maxval['B2'],None,None),#this expression ensure continuity at spectral break
                  ('A',init['A'],vary['A'],minval['A'],maxval['A'],None,None),
                  ('A2',init['A2'] ,vary['A2'],minval['A2'],maxval['A2'],'A * x1**(B-B2)',None))#must add after A is defined
    
    
    if header[42]=='1': #ie if gaussian is present
        params.add_many(('gauss_amp',init['gauss_amp'],vary['gauss_amp'],minval['gauss_amp'],maxval['gauss_amp'],None,None),
                    ('gauss_centre',init['gauss_centre'] ,vary['gauss_centre'],minval['gauss_centre'],maxval['gauss_centre'],None,None),
                    ('sigma',init['sigma'],vary['sigma'],minval['sigma'],maxval['sigma'],None,None))
    
    if header[56]=='1':#ie if the power law is present
        params.add_many(('A_sing',init['A_sing'],vary['A_sing'],minval['A_sing'],maxval['A_sing'],None,None),
                    ('B_sing',init['B_sing'] ,vary['B_sing'],minval['B_sing'],maxval['B_sing'],None,None))

    if header[70]=='1':#ie if the kappa func is present
        params.add_many(
                    ('A_k',init['A_k'] ,vary['A_k'],minval['A_k'],maxval['A_k'],None,None),
                    ('T_k',init['T_k'] ,vary['T_k'],minval['T_k'],maxval['T_k'],None,None),
                    ('m_i',init['m_i'] ,vary['m_i'],minval['m_i'],maxval['m_i'],None,None),
                    ('n_i',init['n_i'],vary['n_i'],minval['n_i'],maxval['n_i'],None,None),
                    ('kappa',init['kappa'] ,vary['kappa'],minval['kappa'],maxval['kappa'],None,None))
    
    if header[92]=='1':#ie if the combined thermal and bpl is present
    
        params.add_many(('amp_c',init['amp_c'],vary['amp_c'],minval['amp_c'],maxval['amp_c'],None,None),
                    ('T_c',init['T_c'] ,vary['T_c'],minval['T_c'],maxval['T_c'],None,None),
                    ('alpha_c',init['alpha_c'],vary['alpha_c'],minval['alpha_c'],maxval['alpha_c'],None,None),
                    ('x0_c',init['x0_c'],vary['x0_c'],minval['x0_c'],maxval['x0_c'],None,None),
                    ('x1_c',init['x1_c'],vary['x1_c'],minval['x1_c'],maxval['x1_c'],None,None),
                     ('B_c',init['B_c'] ,vary['B_c'],minval['B_c'],maxval['B_c'],None,None), #this one should be shallower than B2, constrained as such
                    ('B2_c',init['B2_c'],vary['B2_c'],minval['B2_c'],maxval['B2_c'],None,None))
    
    if header[118]=='1':#ie if the double therm func is present
        
        params.add_many(('amp_d_1',init['amp_d_1'],vary['amp_d_1'],minval['amp_d_1'],maxval['amp_d_1'],None,None),
                    ('T_d_1',init['T_d_1'] ,vary['T_d_1'],minval['T_d_1'],maxval['T_d_1'],None,None),
                    ('alpha_d_1',init['alpha_d_1'],vary['alpha_d_1'],minval['alpha_d_1'],maxval['alpha_d_1'],None,None),
                    ('amp_d_2',init['amp_d_2'],vary['amp_d_2'],minval['amp_d_2'],maxval['amp_d_2'],None,None),
                    ('T_d_2',init['T_d_2'] ,vary['T_d_2'],minval['T_d_2'],maxval['T_d_2'],None,None),
                    ('alpha_d_2',init['alpha_d_2'],vary['alpha_d_2'],minval['alpha_d_2'],maxval['alpha_d_2'],None,None))
    
    if header[130]=='1':# ie if the triple power law is present
    
        params.add_many(('x1_tpl',init['x1_tpl'],vary['x1_tpl'],minval['x1_tpl'],maxval['x1_tpl'],None,None),
                        ('x2_tpl',init['x2_tpl'],vary['x2_tpl'],minval['x2_tpl'],maxval['x2_tpl'],None,None),
                       ('B_tpl',init['B_tpl'] ,vary['B_tpl'],minval['B_tpl'],maxval['B_tpl'],None,None), 
                      ('B2_tpl',init['B2_tpl'],vary['B2_tpl'],minval['B2_tpl'],maxval['B2_tpl'],None,None),
                      ('B3_tpl',init['B3_tpl'],vary['B3_tpl'],minval['B3_tpl'],maxval['B3_tpl'],None,None),
                  ('A_tpl',init['A_tpl'],vary['A_tpl'],minval['A_tpl'],maxval['A_tpl'],None,None),
                  ('A2_tpl',init['A2_tpl'] ,vary['A2_tpl'],minval['A2_tpl'],maxval['A2_tpl'],'A_tpl * x1_tpl**(B_tpl-B2_tpl)',None),
                  ('A3_tpl',init['A3_tpl'] ,vary['A3_tpl'],minval['A3_tpl'],maxval['A3_tpl'],'A2_tpl * x2_tpl**(B2_tpl-B3_tpl)',None))#must add after A_tpl is defined
        
    if header[142]=='1':# ie if the quad power law is present
    
        params.add_many(('x1_qpl',init['x1_qpl'],vary['x1_qpl'],minval['x1_qpl'],maxval['x1_qpl'],None,None),
                        ('x2_qpl',init['x2_qpl'],vary['x2_qpl'],minval['x2_qpl'],maxval['x2_qpl'],None,None),
                        ('x3_qpl',init['x3_qpl'],vary['x3_qpl'],minval['x3_qpl'],maxval['x3_qpl'],None,None),
                       ('B_qpl',init['B_qpl'] ,vary['B_qpl'],minval['B_qpl'],maxval['B_qpl'],None,None), 
                      ('B2_qpl',init['B2_qpl'],vary['B2_qpl'],minval['B2_qpl'],maxval['B2_qpl'],None,None),
                      ('B3_qpl',init['B3_qpl'],vary['B3_qpl'],minval['B3_qpl'],maxval['B3_qpl'],None,None),
                      ('B4_qpl',init['B4_qpl'],vary['B4_qpl'],minval['B4_qpl'],maxval['B4_qpl'],None,None),
                  ('A_qpl',init['A_qpl'],vary['A_qpl'],minval['A_qpl'],maxval['A_qpl'],None,None),
                  ('A2_qpl',init['A2_qpl'] ,vary['A2_qpl'],minval['A2_qpl'],maxval['A2_qpl'],'A_qpl * x1_qpl**(B_qpl-B2_qpl)',None),
                  ('A3_qpl',init['A3_qpl'] ,vary['A3_qpl'],minval['A3_qpl'],maxval['A3_qpl'],'A2_qpl * x2_qpl**(B2_qpl-B3_qpl)',None),
                  ('A4_qpl',init['A4_qpl'] ,vary['A4_qpl'],minval['A4_qpl'],maxval['A4_qpl'],'A3_qpl * x3_qpl**(B3_qpl-B4_qpl)',None))#must add after A_qpl is defined
        
              
    if header[159]=='1':# ie if the quint power law is present
    
        params.add_many(('x1_5pl',init['x1_5pl'],vary['x1_5pl'],minval['x1_5pl'],maxval['x1_5pl'],None,None),
                        ('x2_5pl',init['x2_5pl'],vary['x2_5pl'],minval['x2_5pl'],maxval['x2_5pl'],None,None),
                        ('x3_5pl',init['x3_5pl'],vary['x3_5pl'],minval['x3_5pl'],maxval['x3_5pl'],None,None),
                        ('x4_5pl',init['x4_5pl'],vary['x4_5pl'],minval['x4_5pl'],maxval['x4_5pl'],None,None),
                       ('B_5pl',init['B_5pl'] ,vary['B_5pl'],minval['B_5pl'],maxval['B_5pl'],None,None), 
                      ('B2_5pl',init['B2_5pl'],vary['B2_5pl'],minval['B2_5pl'],maxval['B2_5pl'],None,None),
                      ('B3_5pl',init['B3_5pl'],vary['B3_5pl'],minval['B3_5pl'],maxval['B3_5pl'],None,None),
                      ('B4_5pl',init['B4_5pl'],vary['B4_5pl'],minval['B4_5pl'],maxval['B4_5pl'],None,None),
                      ('B5_5pl',init['B5_5pl'],vary['B5_5pl'],minval['B5_5pl'],maxval['B5_5pl'],None,None),
                  ('A_5pl',init['A_5pl'],vary['A_5pl'],minval['A_5pl'],maxval['A_5pl'],None,None),
                  ('A2_5pl',init['A2_5pl'] ,vary['A2_5pl'],minval['A2_5pl'],maxval['A2_5pl'],'A_5pl * x1**(B-B2)',None),
                  ('A3_5pl',init['A3_5pl'] ,vary['A3_5pl'],minval['A3_5pl'],maxval['A3_5pl'],'A2_5pl * x2**(B2-B3)',None),
                  ('A4_5pl',init['A4_5pl'] ,vary['A4_5pl'],minval['A4_5pl'],maxval['A4_5pl'],'A3_5pl * x3**(B3-B4)',None),
                  ('A5_5pl',init['A5_5pl'] ,vary['A5_5pl'],minval['A5_5pl'],maxval['A5_5pl'],'A4_5pl * x4**(B4-B5_5pl)',None))#must add after A_5pl is defined
        
        
    
        
    #two stage fit, global and local minimisation# but only if not kappa
    #setup fitter, with resid func, param starts, and x+y data
    
    
    if header[70] != '1':
        fitter = lmfit.Minimizer(resid_calc, params, fcn_kws={
            'x_data': x_data_sliced,
            'y_data': y_data_sliced,
            'uncert': uncert_sliced,
            'header': header
        }, scale_covar=True)
    
        progress_win = tk.Toplevel()
        progress_win.title("Global Fit Progress")
    
        label = tk.Label(progress_win, text="Running fit...")
        label.pack(pady=10)
    
        pb = ttk.Progressbar(progress_win, orient='horizontal', length=300, mode='determinate', maximum=100)
        pb.pack(pady=20)
    
        def run_fit():
            global result, fitter_local
            best_result = None
            lowest_chisq = float("inf")
            n_seeds = 10
    
            for seed in range(n_seeds):
                try:
                    np.random.seed(seed)
                    random.seed(seed)
                    trial = fitter.minimize(
                        method='basinhopping',
                        stepsize=0.00001,
                        seed=seed,
                        minimizer_kwargs={'method': 'TNC', 'options': {'ftol': 1e-9, 'gtol': 1e-9, 'eps': 1e-7}}
                    )
                    if trial.chisqr < lowest_chisq:
                        best_result = trial
                        lowest_chisq = trial.chisqr
                except Exception as e:
                    print(f"Seed {seed} failed: {e}")
    
                pb['value'] += 100 / n_seeds
                pb.update_idletasks()
    
            if best_result is not None:
                params.update(best_result.params)
                fitter_local = lmfit.Minimizer(
                    resid_calc,
                    params,
                    fcn_kws={'x_data': x_data, 'y_data': y_data, 'uncert': uncert, 'header': header},
                    scale_covar=True
                )
                result = fitter_local.minimize(
                    method='lbfgsb',
                    options={'max_nfev': 50000, 'ftol': 1e-9, 'gtol': 1e-9, 'eps': 1e-7}
                )
            progress_win.destroy()  # Closes the window, allows wait_window to continue
            
    
        threading.Thread(target=run_fit, daemon=True).start()
        
        # Block until progress window is destroyed
        progress_win.wait_window()
        

    #write error report (optional, un comment for print to console)
    global fit_summary
    global result
    fit_summary=lmfit.fit_report(result)

    global bic
    bic=result.bic
    
    #unpack params object
    pars=result.params
    global parvals
    #convert params object to a dictionary
    parvals=pars.valuesdict()
    
    #read the parameters dictionary, according to the functions that should be present
    if header[9]=='1':# ie if the bpl is present
        x1=parvals["x1"]
        A=parvals["A"]
        B=parvals["B"]
        A2=parvals["A2"]
        B2=parvals["B2"]
    
    if header[28]=='1':#ie if the therm func is present   
        amp=parvals["amp"]
        T=parvals["T"]
        alpha=parvals["alpha"]
        
    if header[42]=='1': #ie if gaussian is present
        gauss_amp=parvals["gauss_amp"]
        gauss_centre=parvals["gauss_centre"]
        sigma=parvals["sigma"]
    
    if header[56]=='1': #ie if power law is present
        A_sing=parvals["A_sing"]
        B_sing=parvals["B_sing"]
    
    if header[70]=='1': #ie if kappa is present

        A_k=parvals["A_k"]
        T_k=parvals["T_k"]
        m_i=parvals["m_i"]
        n_i=parvals["n_i"]
        kappa=parvals["kappa"]
    
    if header[92]=='1':
        amp_c=parvals['amp_c']
        T_c=parvals['T_c']
        alpha_c=parvals['alpha_c']
        x0_c=parvals['x0_c']
        x1_c=parvals['x1_c']
        B_c=parvals['B_c']
        B2_c=parvals['B2_c']
    
    if header[118]=='1':#ie if the double therm func is present   
        amp_d_1=parvals["amp_d_1"]
        T_d_1=parvals["T_d_1"]
        alpha_d_1=parvals["alpha_d_1"]
        amp_d_2=parvals["amp_d_2"]
        T_d_2=parvals["T_d_2"]
        alpha_d_2=parvals["alpha_d_2"]
    
    
    if header[130]=='1':# ie if the triple power law is present
        
        x1_tpl=parvals["x1_tpl"]
        x2_tpl=parvals["x2_tpl"]
        A_tpl=parvals["A_tpl"]
        B_tpl=parvals["B_tpl"]
        A2_tpl=parvals["A2_tpl"]
        B2_tpl=parvals["B2_tpl"]   
        A3_tpl=parvals["A3_tpl"]
        B3_tpl=parvals["B3_tpl"] 
   
    if header[142]=='1':# ie if the quad power law is present
        
        x1_qpl=parvals["x1_qpl"]
        x2_qpl=parvals["x2_qpl"]
        x3_qpl=parvals["x3_qpl"]
        A_qpl=parvals["A_qpl"]
        B_qpl=parvals["B_qpl"]
        A2_qpl=parvals["A2_qpl"]
        B2_qpl=parvals["B2_qpl"]   
        A3_qpl=parvals["A3_qpl"]
        B3_qpl=parvals["B3_qpl"] 
        A4_qpl=parvals["A4_qpl"]
        B4_qpl=parvals["B4_qpl"]

    if header[159]=='1':# ie if the quint power law is present
        
        x1_5pl=parvals["x1_5pl"]
        x2_5pl=parvals["x2_5pl"]
        x3_5pl=parvals["x3_5pl"]
        x4_5pl=parvals["x4_5pl"]
        A_5pl=parvals["A_5pl"]
        B_5pl=parvals["B_5pl"]
        A2_5pl=parvals["A2_5pl"]
        B2_5pl=parvals["B2_5pl"]   
        A3_5pl=parvals["A3_5pl"]
        B3_5pl=parvals["B3_5pl"] 
        A4_5pl=parvals["A4_5pl"]
        B4_5pl=parvals["B4_5pl"]
        A5_5pl=parvals["A5_5pl"]
        B5_5pl=parvals["B5_5pl"]

    #print(B)
    
    #calculate the chi squared of the fit
    y_fit=test_func(x_data_sliced,parvals,header)#fitted y values
    chi_sq=sum(((y_fit-y_data_sliced)/uncert_sliced)**2)#chi squared

    #calc reduced chi sq
    dof=len(y_data_sliced)-len(parvals)#degrees of freedom
    global redchi
    redchi=chi_sq/dof


    
    #return the parameter uncertainties as well
    global param_uncert_calced
    native_uncert=result.errorbars#this determines if uncerts were generated natively in the fit
    if native_uncert:#if fitter generated uncerts, use those
        print('native uncerts use')
        param_uncert_calced={param_name: param.stderr for param_name, param in result.params.items()}#output requires some reprocessing to correct form by removing param values

    else:#if fitter has not generated uncerts, use bayesian posterior method
        print('bayes uncerts use')
        posterior = fitter_local.minimize( method='emcee',  burn=300, steps=5000, thin=20,
                              is_weighted=True)
        
        #locate MLE value in the chain to get sigmas-not neccessary for operations
        highest_prob = np.argmax(posterior.lnprob)
        hp_loc = np.unravel_index(highest_prob, posterior.lnprob.shape)
        mle_soln = posterior.chain[hp_loc]#chain item at location of highest prob
        param_uncert_calced=dict()#set up uncert starage object 
        for name, param in parvals.items():#go through each parameter. if varying get stderr, if not is 0
            if vary[name]:#chain only has varying params! must make sure they only use these
                param_uncert_calced[name]=posterior.params[name].stderr
            else:#for non-varying params
                param_uncert_calced[name] =0
            
            
    

    #%%result plotting
    x_model=np.linspace(min(x_data), max(x_data), 1000000)#set up an x-model for plotting the fitted line
    fit=test_func(x_model,parvals,header)# y-values for our new modeled fit
    

    


    
    
    
    
    #fit window declared global above


    
    if fit_window is not None:# and fit_window.winfo_exists():
            #close any open figues
        fit_window.destroy()
        fit_window=None
        

    
    
    
    
    
    
    
    fig_fit =plt.Figure()#figsize=plot_wind_size, dpi=200)
    ax_fit= fig_fit.add_subplot()#1, 1, 1)
    

    #plot data
    ax_fit.scatter(list(x_data),list(y_data),marker='1',s=30,c='r')
    ax_fit.set_xlabel("Energy (keV)", fontsize=12, labelpad=10)
    ax_fit.set_ylabel(f"Electron {spec_type} "+r"(cm$^2$ sr s keV)$^{-1}$", fontsize=12, labelpad=10)
    ax_fit.set_yscale("log")
    ax_fit.set_xscale("log")
    ax_fit.set_xticklabels(ax_fit.get_xticklabels(), rotation=45)
    #add error bars
    
    ax_fit.vlines(x=x_data, 
               ymin=np.array(y_data) - np.array(uncert), 
               ymax=np.array(y_data) + np.array(uncert), 
               color='blue', alpha=0.5, linewidth=1)

    #ax_fit.set_title(, fontsize=14, fontweight="bold", pad=15)

    ax_fit.plot(x_model,fit, 'k',zorder=100000)
    
    if header[28]=='1':#ie oif the therm func is present 
        fit2=therm_func(x_model,amp,T,alpha)
        ax_fit.plot(x_model,fit2, 'r', label='Thermal Law', linestyle='solid')
    
    if header[9]=='1':# ie if the bpl is present
        xlo=[ 1 if x_i<x1 else 0 for x_i in x_model] #below x0
        xhi=[ 1 if x_i>=x1 else 0 for x_i in x_model]#above x
        
        fit3=lin_func(x_model,A,B)*xlo
        fit4=lin_func2(x_model,A2,B2)*xhi

        ax_fit.plot(x_model,fit3, 'g', label='Broken Power Law',linestyle='dotted')
        ax_fit.plot(x_model,fit4, 'g')
        ax_fit.scatter(x1,test_func(int(x1),parvals,header),zorder=100000,c='black')
        
    if header[42]=='1': #ie if gaussian is present
        fit5=gauss_func(x_model, gauss_amp, gauss_centre, sigma)
        ax_fit.plot(x_model,fit5, 'b', label='Gaussian',linestyle='dashdot')
    
    
    if header[56]=='1': #ie if power law is present
        fit6=power_func(x_model, A_sing, B_sing)
        ax_fit.plot(x_model,fit6, 'm', label='Power Law',linestyle='dashed')
    
    if header[70]=='1': #ie if kappa function is present
        fit7=kappa_func(x_model, A_k, T_k, m_i,n_i,kappa)
        ax_fit.plot(x_model,fit7, 'c', label='Kappa Function',linestyle=(0, (3, 5, 1, 5, 1, 5)))
        
    if header[92]=='1': #ie if combined function is present
        #print('all works')
        fit8=bpl_and_therm_func(x_model,amp_c,T_c,alpha_c,x0_c,x1_c,B_c,B2_c)
        ax_fit.plot(x_model,fit8, 'g', label='BPL and Thermal Function',linestyle='dotted')

    if header[118]=='1':#ie if the double therm func is present 
        fit9=double_therm_func(x_model,amp_d_1,T_d_1,alpha_d_1,amp_d_2,T_d_2,alpha_d_2)
        fit10=therm_func(x_model,amp_d_1,T_d_1,alpha_d_1)
        fit11=therm_func(x_model,amp_d_2,T_d_2,alpha_d_2)
        ax_fit.plot(x_model,fit9, 'r', label='Double Thermal Law', linestyle='solid')
        ax_fit.plot(x_model,fit10, 'r', linestyle='dotted')
        ax_fit.plot(x_model,fit11, 'r',  linestyle='dotted')

    if header[130]=='1':# ie if the tpl is present
        xlo=[ 1 if x_i<x1_tpl else 0 for x_i in x_model] #below x1
        xmid =[ 1 if (x_i>=x1_tpl and x_i<=x2_tpl) else 0 for x_i in x_model] #between x1 and x2
        xhi=[ 1 if x_i>=x2_tpl else 0 for x_i in x_model]#above x2    
        
        fit12=lin_func(x_model,A_tpl,B_tpl)*xlo
        fit13=lin_func2(x_model,A2_tpl,B2_tpl)*xmid
        fit14=lin_func2(x_model,A3_tpl,B3_tpl)*xhi
        
        ax_fit.plot(x_model,fit12, 'g')
        ax_fit.plot(x_model,fit13, 'g', label='Triple Power Law',linestyle='dotted')
        ax_fit.plot(x_model,fit14, 'g')
        ax_fit.scatter(x1_tpl,test_func(int(x1_tpl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x2_tpl,test_func(int(x2_tpl),parvals,header),zorder=100000,c='black')
        
    if header[142]=='1':# ie if the qpl is present
        xlo=[ 1 if x_i<x1 else 0 for x_i in x_model] #below x1
        xmid1 =[ 1 if (x_i>=x1_qpl and x_i<=x2_qpl) else 0 for x_i in x_model] #between x1 and x2
        xmid2 =[ 1 if (x_i>x2_qpl and x_i<=x3_qpl) else 0 for x_i in x_model] #between x2 and x3
        xhi=[ 1 if x_i>x3_qpl else 0 for x_i in x_model]#above x3    
        
        fit15=lin_func2(x_model,A_qpl,B_qpl)*xlo
        fit16=lin_func2(x_model,A2_qpl,B2_qpl)*xmid1
        fit17=lin_func2(x_model,A3_qpl,B3_qpl)*xmid2
        fit18=lin_func2(x_model,A4_qpl,B4_qpl)*xhi
        
        ax_fit.plot(x_model,fit15, 'g')
        ax_fit.plot(x_model,fit16, 'g', label='Quadruple Power Law',linestyle='dotted')
        ax_fit.plot(x_model,fit17, 'g')
        ax_fit.plot(x_model,fit18, 'g')
        ax_fit.scatter(x1_qpl,test_func(int(x1_qpl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x2_qpl,test_func(int(x2_qpl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x3_qpl,test_func(int(x3_qpl),parvals,header),zorder=100000,c='black')

    if header[159]=='1':# ie if the 5pl is present
        xlo=[ 1 if x_i<x1_5pl else 0 for x_i in x_model] #below x1
        xmid1 =[ 1 if (x_i>=x1_5pl and x_i<=x2_5pl) else 0 for x_i in x_model] #between x1 and x2
        xmid2 =[ 1 if (x_i>x2_5pl and x_i<=x3_5pl) else 0 for x_i in x_model] #between x2 and x3
        xmid3 =[ 1 if (x_i>x3_5pl and x_i<=x4_5pl) else 0 for x_i in x_model] #between x3 and x4
        xhi=[ 1 if x_i>x4_5pl else 0 for x_i in x_model]#above x4    
        
        fit19=lin_func2(x_model,A_5pl,B_5pl)*xlo
        fit20=lin_func2(x_model,A2_5pl,B2_5pl)*xmid1
        fit21=lin_func2(x_model,A3_5pl,B3_5pl)*xmid2
        fit22=lin_func2(x_model,A4_5pl,B4_5pl)*xmid3
        fit23=lin_func2(x_model,A5_5pl,B5_5pl)*xhi
        
        ax_fit.plot(x_model,fit19, 'g')
        ax_fit.plot(x_model,fit20, 'g', label='Quintuple Power Law',linestyle='dotted')
        ax_fit.plot(x_model,fit21, 'g')
        ax_fit.plot(x_model,fit22, 'g')
        ax_fit.plot(x_model,fit23, 'g')
        ax_fit.scatter(x1_5pl,test_func(int(x1_5pl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x2_5pl,test_func(int(x2_5pl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x3_5pl,test_func(int(x3_5pl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x4_5pl,test_func(int(x4_5pl),parvals,header),zorder=100000,c='black')
        
    ax_fit.set_yscale("log")
    ax_fit.set_xscale("log")
    
    #set plot limits so that it is focussed on the data, to avoid scaling issues from fitted curve
    ax_fit.set_ylim(min(y_data)/2,max(y_data)*2) 
    ax_fit.set_xlim(min(x_model)-2,max(x_model)+20)
    
    #add legend to plot
    ax_fit.legend(title=f"Reduced Chi sq = {round(redchi,1)}")



    ax_fit.grid()
    canvas_fit = FigureCanvasTkAgg(fig_fit, master=fit_window) 
    canvas_fit.draw()  
    canvas_fit.get_tk_widget().pack()
    
    #add buttton to save figure
    def fig_save_hndl():
        file_obj=tk.filedialog.asksaveasfilename()
        fig_fit.savefig(file_obj,bbox_inches='tight')
    
    #create preview button
    fig_save_button=tk.Button(
    text="Save Plot",  width=25,  height=2,  bg="white",  fg="black",  command=fig_save_hndl,  master=fit_window)
    fig_save_button.pack(side=tk.BOTTOM) 
    
    #second plot showing the residuals of the fit
    
    resid_window=tk.Toplevel()
    fig_resids =plt.Figure(figsize=plot_wind_size, dpi=300)
    ax_resids= fig_resids.add_subplot(1, 1, 1)

    
    resids=resid_calc(pars,x_data_sliced,y_data_sliced,uncert_sliced,header)

    ax_resids.plot(list(x_data_sliced),resids,marker='o')
    ax_resids.set_ylabel('Residual')
    ax_resids.set_xscale('log')
    ax_resids.set_xlabel('Energy (keV)')
    ax_resids.grid()
    
    #enable this to save residuals-must change file name-make a button soon
    with open('C:/Users/w23014130/OneDrive - Northumbria University - Production Azure AD/Documents/PhD/1st year/Paper 1 maybe/final figs and fits/final figs v7/TS fitting eg/'+f'ts_eas_fitting_resids.pkl', 'w+b') as f:
        pickle.dump([resids], f)
     
     
    #fig_resids.savefig(fname=.png',bbox_inches='tight')
    canvas_resids = FigureCanvasTkAgg(fig_resids, master=resid_window) 
    canvas_resids.draw()  
    canvas_resids.get_tk_widget().pack()
    


    return(parvals,param_uncert_calced)




#%%preview initial parameter for fit

def param_preview(x_data,y_data,parvals,header):
    x_model=np.linspace(min(x_data), max(x_data), 1000000)#set up an x-model for plotting the fitted line

    
    #unpack parvals
    #defining what parameters to read in, depending on the header definiions of the function to be fitted
    if header[9]=='1':# ie if the broken power law is present

        x1=parvals["x1"]
        A=parvals["A"]
        B=parvals["B"]
        A2=parvals["A2"]
        B2=parvals["B2"]   

        
    
    
    
    if header[28]=='1':#ie if the therm func is present 
        amp=parvals["amp"]
        T=parvals["T"]
        alpha=parvals["alpha"]
        
    
    if header[42]=='1': #ie if gaussian is present
        gauss_amp=parvals["gauss_amp"]
        gauss_centre=parvals["gauss_centre"]
        sigma=parvals["sigma"]
        
        
    if header[56]=='1': #ie if single power law is present
        A_sing=parvals["A_sing"]
        B_sing=parvals["B_sing"]

        
        
    if header[70]=='1': #ie if kappa is present
        A_k=parvals["A_k"]
        T_k=parvals["T_k"]
        m_i=parvals["m_i"]
        n_i=parvals["n_i"]
        kappa=parvals["kappa"]   
    
    if header[92]=='1':#combined func
        amp_c=parvals['amp_c']
        T_c=parvals['T_c']
        alpha_c=parvals['alpha_c']
        x0_c=parvals['x0_c']
        x1_c=parvals['x1_c']
        B_c=parvals['B_c']
        B2_c=parvals['B2_c']
    
    if header[118]=='1':#ie if the double therm func is present 
        amp_d_1=parvals["amp_d_1"]
        T_d_1=parvals["T_d_1"]
        alpha_d_1=parvals["alpha_d_1"]
        amp_d_2=parvals["amp_d_2"]
        T_d_2=parvals["T_d_2"]
        alpha_d_2=parvals["alpha_d_2"]
        
    if header[130]=='1':# ie if the triple power law is present
        
        x1_tpl=parvals["x1_tpl"]
        x2_tpl=parvals["x2_tpl"]
        A_tpl=parvals["A_tpl"]
        B_tpl=parvals["B_tpl"]
        A2_tpl=parvals["A2_tpl"]
        B2_tpl=parvals["B2_tpl"]   
        A3_tpl=parvals["A3_tpl"]
        B3_tpl=parvals["B3_tpl"] 
 
    if header[142]=='1':# ie if the quad power law is present
        
        x1_qpl=parvals["x1_qpl"]
        x2_qpl=parvals["x2_qpl"]
        x3_qpl=parvals["x3_qpl"]
        A_qpl=parvals["A_qpl"]
        B_qpl=parvals["B_qpl"]
        A2_qpl=parvals["A2_qpl"]
        B2_qpl=parvals["B2_qpl"]   
        A3_qpl=parvals["A3_qpl"]
        B3_qpl=parvals["B3_qpl"] 
        A4_qpl=parvals["A4_qpl"]
        B4_qpl=parvals["B4_qpl"]
        
    if header[159]=='1':# ie if the quint power law is present
        
        x1_5pl=parvals["x1_5pl"]
        x2_5pl=parvals["x2_5pl"]
        x3_5pl=parvals["x3_5pl"]
        x4_5pl=parvals["x4_5pl"]
        A_5pl=parvals["A_5pl"]
        B_5pl=parvals["B_5pl"]
        A2_5pl=parvals["A2_5pl"]
        B2_5pl=parvals["B2_5pl"]   
        A3_5pl=parvals["A3_5pl"]
        B3_5pl=parvals["B3_5pl"] 
        A4_5pl=parvals["A4_5pl"]
        B4_5pl=parvals["B4_5pl"]
        A5_5pl=parvals["A5_5pl"]
        B5_5pl=parvals["B5_5pl"]
        
    global test_func
    def test_func(x,parvals,header): # this function is the one we are trying to fit to the data
        
    #if x data list, create y data as list too. else if x is array, use array for y
        if type(x)==list:
            y=np.zeros(len(x))
            x=np.array(x)
        else:
            y=0
        
        #print(header)
        #defining what parameters to read in, depending on the header definiions of the function to be fitted
        if header[9]=='1':# ie if the broken power law is present
            x1=parvals["x1"]
            A=parvals["A"]
            B=parvals["B"]
            A2=parvals["A2"]
            B2=parvals["B2"]   
            y+=broken_power_law(x,x1,A,B,A2,B2)
        
        
        
        if header[28]=='1':#ie if the therm func is present 
            amp=parvals["amp"]
            T=parvals["T"]
            alpha=parvals["alpha"]
            y+=therm_func(x,amp,T,alpha)
        
        if header[42]=='1': #ie if gaussian is present
            gauss_amp=parvals["gauss_amp"]
            gauss_centre=parvals["gauss_centre"]
            sigma=parvals["sigma"]
            y+=gauss_func(x, gauss_amp, gauss_centre, sigma)
            
        if header[56]=='1': #ie if single power law is present
            A_sing=parvals["A_sing"]
            B_sing=parvals["B_sing"]
            y+=power_func(x, A_sing, B_sing)
            
        if header[70]=='1': #ie if kappa is present
            A_k=parvals["A_k"]
            T_k=parvals["T_k"]
            m_i=parvals["m_i"]
            n_i=parvals["n_i"]
            kappa=parvals["kappa"]
            y+=kappa_func(x, A_k, T_k, m_i,n_i,kappa)
            
        if header[92]=='1':
            amp_c=parvals['amp_c']
            T_c=parvals['T_c']
            alpha_c=parvals['alpha_c']
            x0_c=parvals['x0_c']
            x1_c=parvals['x1_c']
            B_c=parvals['B_c']
            B2_c=parvals['B2_c']
            
            y+=bpl_and_therm_func(x,amp_c,T_c,alpha_c,x0_c,x1_c,B_c,B2_c)
            #print('test')
            #print(y)
        
        if header[118]=='1':#ie if the double therm func is present 
            amp_d_1=parvals["amp_d_1"]
            T_d_1=parvals["T_d_1"]
            alpha_d_1=parvals["alpha_d_1"]
            amp_d_2=parvals["amp_d_2"]
            T_d_2=parvals["T_d_2"]
            alpha_d_2=parvals["alpha_d_2"]
            y+=double_therm_func(x,amp_d_1,T_d_1,alpha_d_1,amp_d_2,T_d_2,alpha_d_2)
        
        
        if header[130]=='1':# ie if the triple power law is present
            
            x1_tpl=parvals["x1_tpl"]
            x2_tpl=parvals["x2_tpl"]
            A_tpl=parvals["A_tpl"]
            B_tpl=parvals["B_tpl"]
            A2_tpl=parvals["A2_tpl"]
            B2_tpl=parvals["B2_tpl"]   
            A3_tpl=parvals["A3_tpl"]
            B3_tpl=parvals["B3_tpl"] 
            y+=triple_power_law(x,x1_tpl,x2_tpl,A_tpl,B_tpl,A2_tpl,B2_tpl,A3_tpl,B3_tpl)
        
        
        if header[142]=='1':# ie if the quad power law is present
    
            x1_qpl=parvals["x1_qpl"]
            x2_qpl=parvals["x2_qpl"]
            x3_qpl=parvals["x3_qpl"]
            A_qpl=parvals["A_qpl"]
            B_qpl=parvals["B_qpl"]
            A2_qpl=parvals["A2_qpl"]
            B2_qpl=parvals["B2_qpl"]   
            A3_qpl=parvals["A3_qpl"]
            B3_qpl=parvals["B3_qpl"] 
            A4_qpl=parvals["A4_qpl"]
            B4_qpl=parvals["B4_qpl"]
            y+=quad_power_law(x,x1_qpl,x2_qpl,x3_qpl,A_qpl,B_qpl,A2_qpl,B2_qpl,A3_qpl,B3_qpl,A4_qpl,B4_qpl)
            
        if header[159]=='1':# ie if the quint power law is present
            
            x1_5pl=parvals["x1_5pl"]
            x2_5pl=parvals["x2_5pl"]
            x3_5pl=parvals["x3_5pl"]
            x4_5pl=parvals["x4_5pl"]
            A_5pl=parvals["A_5pl"]
            B_5pl=parvals["B_5pl"]
            A2_5pl=parvals["A2_5pl"]
            B2_5pl=parvals["B2_5pl"]   
            A3_5pl=parvals["A3_5pl"]
            B3_5pl=parvals["B3_5pl"] 
            A4_5pl=parvals["A4_5pl"]
            B4_5pl=parvals["B4_5pl"]
            A5_5pl=parvals["A5_5pl"]
            B5_5pl=parvals["B5_5pl"]
            y+=quint_power_law(x, x1_5pl, x2_5pl, x3_5pl, x4_5pl, A_5pl, B_5pl, A2_5pl, B2_5pl, A3_5pl, B3_5pl, A4_5pl, B4_5pl, A5_5pl, B5_5pl)
            
        return y
    #print(header)
    fit=test_func(x_model,parvals,header)# y-values for our new modeled fit

    #open a new figure in a new window
    
    plot_wind_size=(4,3.5)#define the window size for the plots
    
    global preview_window
    preview_window=tk.Toplevel()
    preview_window.title('Preview window')
    fig_fit =plt.Figure(figsize=plot_wind_size, dpi=300)
    ax_fit= fig_fit.add_subplot(1, 1, 1)
    

    #plot data
    ax_fit.scatter(list(x_data),list(y_data))
    ax_fit.set_xlabel("Energy (keV)")
    ax_fit.set_ylabel("Electron flux\n"+r"(cm$^2$ sr s keV)$^{-1}$")
    ax_fit.set_yscale("log")
    ax_fit.set_xscale("log")
    
    ax_fit.plot(x_model,fit, 'k',zorder=100000)
    
    if header[28]=='1':#ie if the therm func is present 
        fit2=therm_func(x_model,amp,T,alpha)
        ax_fit.plot(x_model,fit2, 'r', label='Thermal Law', linestyle='solid')
    
    if header[9]=='1':# ie if the bpl is present
        xlo=[ 1 if x_i<x1 else 0 for x_i in x_model] #below x0
        xhi=[ 1 if x_i>=x1 else 0 for x_i in x_model]#above x
        fit3=lin_func(x_model,A,B)*xlo
        fit4=lin_func2(x_model,A2,B2)*xhi

        ax_fit.plot(x_model,fit3, 'g', label='Broken Power Law',linestyle='dotted')
        ax_fit.plot(x_model,fit4, 'g')
        ax_fit.scatter(x1,test_func(int(x1),parvals,header),zorder=100000,c='black')

        
    if header[42]=='1': #ie if gaussian is present
        fit5=gauss_func(x_model, gauss_amp, gauss_centre, sigma)
        ax_fit.plot(x_model,fit5, 'b', label='Gaussian',linestyle='dashdot')
    
    
    if header[56]=='1': #ie if power law is present
        fit6=power_func(x_model, A_sing, B_sing)
        ax_fit.plot(x_model,fit6, 'm', label='Power Law',linestyle='dashed')
    
    if header[70]=='1': #ie if kappa function is present
        fit7=kappa_func(x_model, A_k, T_k, m_i,n_i,kappa)
        ax_fit.plot(x_model,fit7, 'c', label='Kappa Function',linestyle=(0, (3, 5, 1, 5, 1, 5)))

    if header[92]=='1': #ie if combined function is present
        fit8=bpl_and_therm_func(x_model,amp_c,T_c,alpha_c,x0_c,x1_c,B_c,B2_c)
        ax_fit.plot(x_model,fit8, 'g', label='BPL and Thermal Function',linestyle='dotted')
        
    if header[118]=='1':
        fit9=therm_func(x_model,amp_d_1,T_d_1,alpha_d_1)
        fit10=therm_func(x_model,amp_d_2,T_d_2,alpha_d_2)
        ax_fit.plot(x_model,fit9, 'r', label='Thermal Law 1', linestyle='solid')
        ax_fit.plot(x_model,fit10, 'r', label='Thermal Law 2', linestyle='solid')
        
    if header[130]=='1':# ie if the tpl is present
        xlo=[ 1 if x_i<x1_tpl else 0 for x_i in x_model] #below x1
        xmid =[ 1 if (x_i>=x1_tpl and x_i<=x2_tpl) else 0 for x_i in x_model] #between x1 and x2
        xhi=[ 1 if x_i>=x2_tpl else 0 for x_i in x_model]#above x2    
        
        fit12=lin_func(x_model,A_tpl,B_tpl)*xlo
        fit13=lin_func2(x_model,A2_tpl,B2_tpl)*xmid
        fit14=lin_func2(x_model,A3_tpl,B3_tpl)*xhi
        
        ax_fit.plot(x_model,fit12, 'g')
        ax_fit.plot(x_model,fit13, 'g', label='Broken Power Law',linestyle='dotted')
        ax_fit.plot(x_model,fit14, 'g')
        ax_fit.scatter(x1_tpl,test_func(int(x1_tpl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x2_tpl,test_func(int(x2_tpl),parvals,header),zorder=100000,c='black')
      
        
    if header[142]=='1':# ie if the qpl is present
        xlo=[ 1 if x_i<x1_qpl else 0 for x_i in x_model] #below x1
        xmid1 =[ 1 if (x_i>=x1_qpl and x_i<x2_qpl) else 0 for x_i in x_model] #between x1 and x2
        xmid2 =[ 1 if (x_i>=x2_qpl and x_i<x3_qpl) else 0 for x_i in x_model] #between x2 and x3
        xhi=[ 1 if x_i>=x3_qpl else 0 for x_i in x_model]#above x3    
        
        fit15=lin_func2(x_model,A_qpl,B_qpl)*xlo
        fit16=lin_func2(x_model,A2_qpl,B2_qpl)*xmid1
        fit17=lin_func2(x_model,A3_qpl,B3_qpl)*xmid2
        fit18=lin_func2(x_model,A4_qpl,B4_qpl)*xhi

        ax_fit.plot(x_model,fit15, 'g')
        ax_fit.plot(x_model,fit16, 'g', label='Quadruple Power Law',linestyle='dotted')
        ax_fit.plot(x_model,fit17, 'g')
        ax_fit.plot(x_model,fit18, 'g')
        ax_fit.scatter(x1_qpl,test_func(int(x1_qpl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x2_qpl,test_func(int(x2_qpl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x3_qpl,test_func(int(x3_qpl),parvals,header),zorder=100000,c='black')
        
        
    if header[159]=='1':# ie if the 5pl is present
        xlo=[ 1 if x_i<x1_5pl else 0 for x_i in x_model] #below x1
        xmid1 =[ 1 if (x_i>=x1_5pl and x_i<=x2_5pl) else 0 for x_i in x_model] #between x1 and x2
        xmid2 =[ 1 if (x_i>x2_5pl and x_i<=x3_5pl) else 0 for x_i in x_model] #between x2 and x3
        xmid3 =[ 1 if (x_i>x3_5pl and x_i<=x4_5pl) else 0 for x_i in x_model] #between x3 and x4
        xhi=[ 1 if x_i>x4_5pl else 0 for x_i in x_model]#above x4    
        
        fit19=lin_func2(x_model,A_5pl,B_5pl)*xlo
        fit20=lin_func2(x_model,A2_5pl,B2_5pl)*xmid1
        fit21=lin_func2(x_model,A3_5pl,B3_5pl)*xmid2
        fit22=lin_func2(x_model,A4_5pl,B4_5pl)*xmid3
        fit23=lin_func2(x_model,A5_5pl,B5_5pl)*xhi
        
        ax_fit.plot(x_model,fit19, 'g')
        ax_fit.plot(x_model,fit20, 'g', label='Quintuple Power Law',linestyle='dotted')
        ax_fit.plot(x_model,fit21, 'g')
        ax_fit.plot(x_model,fit22, 'g')
        ax_fit.plot(x_model,fit23, 'g')
        ax_fit.scatter(x1_5pl,test_func(int(x1_5pl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x2_5pl,test_func(int(x2_5pl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x3_5pl,test_func(int(x3_5pl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x4_5pl,test_func(int(x4_5pl),parvals,header),zorder=100000,c='black')
            
    ax_fit.set_yscale("log")
    ax_fit.set_xscale("log")
    
    #set plot limits so that it is focussed on the data, to avoid scaling issues from fitted curve
    ax_fit.set_ylim(min(y_data)/2,max(y_data)*2) 
    ax_fit.set_xlim(min(x_model),max(x_model))
    
    ax_fit.grid()
    canvas_fit = FigureCanvasTkAgg(fig_fit, master=preview_window) 
    canvas_fit.draw()  
    canvas_fit.get_tk_widget().pack()

#%%save load params from previous fits

def param_save(date,inst,spec_type, bpl_pres, therm_func_pres, gauss_pres,power_pres,kappa_pres,bpl_and_therm_pres, double_therm_func_pres, tpl_pres, qpl_pres, quint_pl_pres,redchi): #function that saves the parameters for later retrieval
    
    #open the file select dialogue to choose a save location
    files = [('Text Document','*.txt')]
    file_obj=tk.filedialog.asksaveasfile(filetypes = files, defaultextension=".txt")
    #file_obj=open(fileloc,'w')#define name of save file, opens it in write mode
    content=list()#set up empty list of contents
    global header #use/define header for all functions
    header=f"bpl_pres={bpl_pres}; therm_func_pres={therm_func_pres}; gauss_pres={gauss_pres}; power_pres={power_pres}; kappa_pres={kappa_pres}; bpl_and_therm_pres={bpl_and_therm_pres}; double_therm_func_pres={double_therm_func_pres}; tpl_pres={tpl_pres}; qpl_pres={qpl_pres}; quint_pl_pres={quint_pl_pres};\n"#define header. header defines which functions have been used to fit the data. ends with newline character
    content.append(header)#add header to top of file
    
    save_pars=dict()
    try:#try excpet statement is to validate inputs as integers
        if therm_func_pres==1:#if thermal function present, save parameter options from the gui for that function
            
            global frame_therm
            
            save_pars['T']=None if init_T_entry.get()=='None' else float(init_T_entry.get())            
            save_pars['amp']=None if init_amp_entry.get()=='None' else float(init_amp_entry.get())            
            save_pars['alpha']=None if init_alpha_entry.get()=='None' else float(init_alpha_entry.get())

            
        if bpl_pres==1:#if bpl function present, save parameter options from the gui for that function            
            
            save_pars['x1']=None if init_x1_entry.get()=='None' else float(init_x1_entry.get())
            save_pars['A2']=None if init_A2_entry.get()=='None' else float(init_A2_entry.get())
            save_pars['B2']=None if init_B2_entry.get()=='None' else float(init_B2_entry.get())
            save_pars['B']=None if init_B_entry.get()=='None' else float(init_B_entry.get())
            save_pars['A']=None if init_A_entry.get()=='None' else float(init_A_entry.get())

        if gauss_pres==1:#if gaussian function present, save parameter options from the gui for that function
            save_pars['gauss_centre']=None if init_gauss_centre_entry.get()=='None' else float(init_gauss_centre_entry.get())
            save_pars['gauss_amp']=None if init_gauss_amp_entry.get()=='None' else float(init_gauss_amp_entry.get())
            save_pars['sigma']=None if init_sigma_entry.get()=='None' else float(init_sigma_entry.get())
           
        if power_pres==1:#if single power law function present, save parameter options from the gui for that function
           save_pars['B_sing']=None if init_B_sing_entry.get()=='None' else float(init_B_sing_entry.get())
           save_pars['A_sing']=None if init_A_sing_entry.get()=='None' else float(init_A_sing_entry.get())
           

           
        if kappa_pres==1:#if kappa function present, save parameter options from the gui for that function

           save_pars['A_k']=None if init_A_k_entry.get()=='None' else float(init_A_k_entry.get())
           
           save_pars['T_k']=None if init_T_k_entry.get()=='None' else float(init_T_k_entry.get())
           
           save_pars['m_i']=None if init_m_i_entry.get()=='None' else float(init_m_i_entry.get())
           
           save_pars['n_i']=None if init_n_i_entry.get()=='None' else float(init_n_i_entry.get())

           save_pars['kappa']=None if init_kappa_entry.get()=='None' else float(init_kappa_entry.get())
           
           
        if bpl_and_therm_pres==1:
            save_pars['T_c']=None if init_T_c_entry.get()=='None' else float(init_T_c_entry.get())
            
            save_pars['amp_c']=None if init_amp_c_entry.get()=='None' else float(init_amp_c_entry.get())
            save_pars['alpha_c']=None if init_alpha_c_entry.get()=='None' else float(init_alpha_c_entry.get())
            
            save_pars['x1_c']=None if init_x1_c_entry.get()=='None' else float(init_x1_c_entry.get())
            save_pars['x0_c']=None if init_x0_c_entry.get()=='None' else float(init_x0_c_entry.get())
            save_pars['B2_c']=None if init_B2_c_entry.get()=='None' else float(init_B2_c_entry.get())
            
            save_pars['B_c']=None if init_B_c_entry.get()=='None' else float(init_B_c_entry.get())
            
        if double_therm_func_pres==1:#if double thermal function present, save parameter options from the gui for that function
            
            save_pars['T_d_1']=None if init_T_d_1_entry.get()=='None' else float(init_T_d_1_entry.get())
            save_pars['amp_d_1']=None if init_amp_d_1_entry.get()=='None' else float(init_amp_d_1_entry.get())
            save_pars['alpha_d_1']=None if init_alpha_d_1_entry.get()=='None' else float(init_alpha_d_1_entry.get())
            save_pars['T_d_2']=None if init_T_d_2_entry.get()=='None' else float(init_T_d_2_entry.get())
            
            save_pars['amp_d_2']=None if init_amp_d_2_entry.get()=='None' else float(init_amp_d_2_entry.get())
            save_pars['alpha_d_2']=None if init_alpha_d_2_entry.get()=='None' else float(init_alpha_d_2_entry.get())
        
        
        if tpl_pres==1:#if tpl function present, save parameter options from the gui for that function
            save_pars['x1_tpl']=None if init_x1_tpl_entry.get()=='None' else float(init_x1_tpl_entry.get())
            save_pars['x2_tpl']=None if init_x2_tpl_entry.get()=='None' else float(init_x2_tpl_entry.get())
            save_pars['A2_tpl']=None if init_A2_tpl_entry.get()=='None' else float(init_A2_tpl_entry.get())
            save_pars['B2_tpl']=None if init_B2_tpl_entry.get()=='None' else float(init_B2_tpl_entry.get())
            save_pars['B_tpl']=None if init_B_tpl_entry.get()=='None' else float(init_B_tpl_entry.get())
            save_pars['A_tpl']=None if init_A_tpl_entry.get()=='None' else float(init_A_tpl_entry.get())
            save_pars['A3_tpl']=None if init_A3_tpl_entry.get()=='None' else float(init_tpl_A3_tpl_entry.get())            
            save_pars['B3_tpl']=None if init_B3_tpl_entry.get()=='None' else float(init_B3_tpl_entry.get())
            
        if qpl_pres==1:#if qpl function present, save parameter options from the gui for that function
            save_pars['x1_qpl']=None if init_x1_qpl_entry.get()=='None' else float(init_x1_qpl_entry.get())
            save_pars['x2_qpl']=None if init_x2_qpl_entry.get()=='None' else float(init_x2_qpl_entry.get())
            save_pars['x3_qpl']=None if init_x3_qpl_entry.get()=='None' else float(init_x3_qpl_entry.get())
            save_pars['A2_qpl']=None if init_A2_qpl_entry.get()=='None' else float(init_A2_qpl_entry.get())
            save_pars['B2_qpl']=None if init_B2_qpl_entry.get()=='None' else float(init_B2_qpl_entry.get())
            save_pars['B_qpl']=None if init_B_qpl_entry.get()=='None' else float(init_B_qpl_entry.get())
            save_pars['A_qpl']=None if init_A_qpl_entry.get()=='None' else float(init_A_qpl_entry.get())
            save_pars['A3_qpl']=None if init_A3_qpl_entry.get()=='None' else float(init_A3_qpl_entry.get())
            save_pars['B3_qpl']=None if init_B3_qpl_entry.get()=='None' else float(init_B3_qpl_entry.get())
            save_pars['A4_qpl']=None if init_A4_qpl_entry.get()=='None' else float(init_A4_qpl_entry.get())
            save_pars['B4_qpl']=None if init_B4_qpl_entry.get()=='None' else float(init_B4_qpl_entry.get())
            
        if quint_pl_pres==1:#if 5pl function present, save parameter options from the gui for that function
            save_pars['x1_5pl']=None if init_x1_5pl_entry.get()=='None' else float(init_x1_5pl_entry.get())
            save_pars['x2_5pl']=None if init_x2_5pl_entry.get()=='None' else float(init_x2_5pl_entry.get())
            save_pars['x3_5pl']=None if init_x3_5pl_entry.get()=='None' else float(init_x3_5pl_entry.get())            
            save_pars['x4_5pl']=None if init_x4_5pl_entry.get()=='None' else float(init_x4_5pl_entry.get())
            save_pars['A2_5pl']=None if init_A2_5pl_entry.get()=='None' else float(init_A2_5pl_entry.get())
            save_pars['B2_5pl']=None if init_B2_5pl_entry.get()=='None' else float(init_B2_5pl_entry.get())
            save_pars['B_5pl']=None if init_B_5pl_entry.get()=='None' else float(init_B_5pl_entry.get())
            save_pars['A_5pl']=None if init_A_5pl_entry.get()=='None' else float(init_A_5pl_entry.get())
            save_pars['A3_5pl']=None if init_A3_5pl_entry.get()=='None' else float(init_A3_5pl_entry.get())
            save_pars['B3_5pl']=None if init_B3_5pl_entry.get()=='None' else float(init_B3_5pl_entry.get())
            save_pars['A4_5pl']=None if init_A4_5pl_entry.get()=='None' else float(init_A4_5pl_entry.get())
            save_pars['B4_5pl']=None if init_B4_5pl_entry.get()=='None' else float(init_B4_5pl_entry.get())            
            save_pars['A5_5pl']=None if init_A5_5pl_entry.get()=='None' else float(init_A5_5pl_entry.get())
            save_pars['B5']=None if init_B5_entry.get()=='None' else float(init_B5_entry.get())
            
    except ValueError as e:
           tk.messagebox.showerror("Invalid Input","Inputs should be floating point intergers")
           print(e)
           
           
    for i in range(len(save_pars)):    #for each parameter
        par_name=np.array(list(save_pars.keys()))[i]
        par_value=str(save_pars[par_name])
        
        content.append(f'{par_name}:{par_value}\n')#add parameter name and value to the content, with newline character at the end
        
    content.append(f'redchi:{redchi}\n')
    content.append(f'bic:{bic}\n')
    content=np.array(content)#make content into array
    file_obj.writelines(content)#write content to file
    
    file_obj.close()#close file
    
    
    
    
    
    
    
def param_load(date,inst,spec_type): #function to load data from the file
    file_obj=tk.filedialog.askopenfile()
    content=[x for x in file_obj]#reads file content into a list line by line
       
    file_obj.close()#close file
    
    parvals=dict()#set up dictionary for loaded parameters
    global header#define header for all functions
    header= content[0]#header is first line of file
    for i in content[1:]:#for remaining lines, get parameter name and value and save to pre-defined dict
        parvals[i.split(':')[0]]=float(i.split(':')[1][:-1])
        
    return header, parvals #output header and parameter dict

#%%window fn



def build_fit_window(x_data, y_data, uncert, date, inst, spec_type):
    window_buttons = tk.Toplevel()#define window. everything between here and "mainloop" makes up this window". MUST only have one tk.Tk(), all esle must be .toplevel else crashes
    window_buttons.minsize(500, 600)
    window_buttons.title("Inspex fitting GUI")

    
    container = tk.Frame(window_buttons)
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    frame_params = tk.Frame(canvas)

    frame_params.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=frame_params, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    frame_headings=tk.Frame(master=frame_params)
    tk.Label(frame_headings, text=" ", font=("Arial", 10))\
        .grid(row=0, column=0, padx=25)
    tk.Label(frame_headings, text="Init", font=("Arial", 10))\
        .grid(row=0, column=1, padx=25)
    tk.Label(frame_headings, text="Min", font=("Arial", 10))\
        .grid(row=0, column=2,  padx=25)
    tk.Label(frame_headings, text="Max", font=("Arial", 10))\
        .grid(row=0, column=3,  padx=15)
    tk.Label(frame_headings, text=" ", font=("Arial", 10))\
        .grid(row=0, column=4, padx=15)
    frame_headings.grid(row=1, column=0, sticky="ew")

        
    def add_param_row(frame, row, label_text, init_val, min_val, max_val, var_state, callback, name_prefix):
        globals()[f"init_{name_prefix}_entry"] = tk.Entry(frame, width=10)
        globals()[f"init_{name_prefix}_entry"].insert(0, str(init_val))
        globals()[f"init_{name_prefix}_entry"].grid(row=row, column=1, padx=5)

        globals()[f"minval_{name_prefix}_entry"] = tk.Entry(frame, width=10)
        globals()[f"minval_{name_prefix}_entry"].insert(0, str(min_val))
        globals()[f"minval_{name_prefix}_entry"].grid(row=row, column=2, padx=5)

        globals()[f"maxval_{name_prefix}_entry"] = tk.Entry(frame, width=10)
        globals()[f"maxval_{name_prefix}_entry"].insert(0, str(max_val))
        globals()[f"maxval_{name_prefix}_entry"].grid(row=row, column=3, padx=5)

        
        globals()[f"btn_vary_{name_prefix}"] = tk.Checkbutton(frame, text=f"Vary {label_text}", command=callback)
        globals()[f"btn_vary_{name_prefix}"].grid(row=row, column=4, padx=5)
        
        if var_state:globals()[f"btn_vary_{name_prefix}"].select() 
        else: globals()[f"btn_vary_{name_prefix}"].deselect()

        tk.Label(frame, text=label_text).grid(row=row, column=0, padx=5, pady=5, sticky="w")



    
    #these functions add the test function components when the user selects/loads them
    
    def add_therm():#add the thermal component to the fitted function
        global therm_func_pres #use global value of thermal function's presence
        if therm_func_pres ==0:#if thermal function not already there
            
            global init #define global initial values for the 3 params of the thermal function
            init['amp']=1e9
            init['T']=12e6
            init['alpha']=1
            
            global vary#define globally whether to initially vary for the 3 params of the thermal function
            vary['amp']=True
            vary['T']=True
            vary['alpha']=False
            
            global minval#define global initial minimum values for the 3 params of the thermal function
            minval['amp']=0
            minval['T']=0
            minval['alpha']=0
            
            global maxval#define global initial maximum values for the 3 params of the thermal function
            maxval['amp']=None
            maxval['T']=1e8
            maxval['alpha']=5    
            
            
            #defining the part of the GUI window that contains the options for the thermal curve
            global frame_therm 
            frame_therm=tk.Frame(master=frame_params)
            tk.Label(frame_therm, text="Thermal Curve", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_amp(): vary['amp'] = not vary['amp']
            def toggle_T(): vary['T'] = not vary['T']
            def toggle_alpha(): vary['alpha'] = not vary['alpha']

            add_param_row(frame_therm, 1, "amp", init['amp'], minval['amp'], maxval['amp'], vary['amp'], toggle_amp, "amp")
            add_param_row(frame_therm, 2, "T", init['T'], minval['T'], maxval['T'], vary['T'], toggle_T, "T")
            add_param_row(frame_therm, 3, "alpha", init['alpha'], minval['alpha'], maxval['alpha'], vary['alpha'], toggle_alpha, "alpha")

            def hndl_remove_therm_btn():
                global therm_func_pres
                frame_therm.grid_forget()
                therm_func_pres = 0

            tk.Button(frame_therm, text='Remove thermal component', command=hndl_remove_therm_btn)\
                .grid(row=4, column=0, columnspan=5, pady=10)

            frame_therm.grid(row=2, column=0, sticky="ew")
            for i in range(6):
                frame_therm.grid_columnconfigure(i, weight=1)
            therm_func_pres = 1 #set the thermal function as present
            
            
            
    
    def add_bpl():#function to add the the broken power law
        global bpl_pres
        if  bpl_pres ==0:
    
            global init#define global initial values for the params of the function

            init['x1']=40
            init['A']=1e5
            init['B']=-1
            init['A2']=1e5
            init['B2']=-2    

    
            global vary#define global if vary values for the params of the function

            vary['x1']=True
            vary['A']=True
            vary['B']=True
            vary['A2']=True
            vary['B2']=True            

            
            global maxval##define global maximum values for the params of the function

            maxval['x1']=50
            maxval['A']=None
            maxval['B']=0
            maxval['A2']=None
            maxval['B2']=0            

            
            global minval##define global minimum values for the params of the function

            minval['x1']=15
            minval['A']=0
            minval['B']=-10
            minval['A2']=0
            minval['B2']=-10            

            global frame_bpl#defining gui section to handle bpl param options
            frame_bpl=tk.Frame(master=frame_params)
            
            tk.Label(frame_bpl, text="Broken Power Law", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_A(): vary['A'] = not vary['A']
            def toggle_A2(): vary['A2'] = not vary['A2']
            def toggle_x1(): vary['x1'] = not vary['x1']
            def toggle_B(): vary['B'] = not vary['B']
            def toggle_B2(): vary['B2'] = not vary['B2']

            add_param_row(frame_bpl, 1, "x1", init['x1'], minval['x1'], maxval['x1'], vary['x1'], toggle_x1, "x1")
            add_param_row(frame_bpl, 2, "A", init['A'], minval['A'], maxval['A'], vary['A'], toggle_A, "A")           
            add_param_row(frame_bpl, 3, "B", init['B'], minval['B'], maxval['B'], vary['B'], toggle_B, "B")
            add_param_row(frame_bpl, 4, "A2", init['A2'], minval['A2'], maxval['A2'], vary['A2'], toggle_A2, "A2")
            add_param_row(frame_bpl, 5, "B2", init['B2'], minval['B2'], maxval['B2'], vary['B2'], toggle_B2, "B2")

            def hndl_remove_bpl_btn():
                global bpl_pres
                frame_bpl.grid_forget()
                bpl_pres = 0

            tk.Button(frame_bpl, text='Remove BPL component', command=hndl_remove_bpl_btn)\
                .grid(row=6, column=0, columnspan=5, pady=10)

            frame_bpl.grid(row=3, column=0, sticky="ew")
            for i in range(6):
                frame_bpl.grid_columnconfigure(i, weight=1)
            bpl_pres = 1
    
    
    
    def add_gauss():#function to add gausian function to gui/test function
        global gauss_pres
        if gauss_pres ==0:
            
            global init#define global initial values for the params of the function
            init['gauss_amp']=1e9
            init['gauss_centre']=0
            init['sigma']=1
            
            global vary#define global if vary values for the params of the function
            vary['gauss_amp']=True
            vary['gauss_centre']=True
            vary['sigma']=True
            
            global minval#define global minimum values for the params of the function
            minval['gauss_amp']=0
            minval['gauss_centre']=None
            minval['sigma']=0
            
            global maxval#define global maximum values for the params of the function
            maxval['gauss_amp']=None
            maxval['gauss_centre']=None
            maxval['sigma']=None   
            
            global frame_gauss#defining gui section to handle gaussian param options
            frame_gauss=tk.Frame(master=frame_params)
            tk.Label(frame_gauss, text="Gaussian", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_gauss_amp(): vary['gauss_amp'] = not vary['gauss_amp']
            def toggle_gauss_centre(): vary['gauss_centre'] = not vary['gauss_centre']
            def toggle_sigma(): vary['sigma'] = not vary['sigma']

            add_param_row(frame_gauss, 1, "gauss_amp", init['gauss_amp'], minval['gauss_amp'], maxval['gauss_amp'], vary['gauss_amp'], toggle_gauss_amp, "gauss_amp")
            add_param_row(frame_gauss, 2, "gauss_centre", init['gauss_centre'], minval['gauss_centre'], maxval['gauss_centre'], vary['gauss_centre'], toggle_gauss_centre, "gauss_centre")
            add_param_row(frame_gauss, 3, "sigma", init['sigma'], minval['sigma'], maxval['sigma'], vary['sigma'], toggle_sigma, "sigma")

            def hndl_remove_gauss_btn():
                global gauss_pres
                frame_gauss.grid_forget()
                gauss_pres = 0

            tk.Button(frame_gauss, text='Remove Gaussian component', command=hndl_remove_gauss_btn)\
                .grid(row=4, column=0, columnspan=5, pady=10)

            frame_gauss.grid(row=4, column=0, sticky="ew")
            for i in range(4):
                frame_gauss.grid_columnconfigure(i, weight=1)
            gauss_pres = 1
    
    def add_power():#function to add power law to gui/test function
        global power_pres#defining gui section to handle power law param options
        if power_pres ==0:
            global init#define global initial values for the params of the function
            init['A_sing']=1e9
            init['B_sing']=-1
            
            global vary#define global if vary values for the params of the function
            vary['A_sing']=True
            vary['B_sing']=True
            
            global minval##define global minimum values for the params of the function
            minval['A_sing']=0
            minval['B_sing']=None
            
            global maxval##define global maximum values for the params of the function
            maxval['A_sing']=None
            maxval['B_sing']=0
            
            global frame_power
            frame_power=tk.Frame(master=frame_params)
            tk.Label(frame_power, text="Power Law", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_A_sing(): vary['A_sing'] = not vary['A_sing']
            def toggle_B_sing(): vary['B_sing'] = not vary['B_sing']

            add_param_row(frame_power, 1, "A_sing", init['A_sing'], minval['A_sing'], maxval['A_sing'], vary['A_sing'], toggle_A_sing, "A_sing")
            add_param_row(frame_power, 2, "B_sing", init['B_sing'], minval['B_sing'], maxval['B_sing'], vary['B_sing'], toggle_B_sing, "B_sing")

            def hndl_remove_power_btn():
                global power_pres
                frame_power.grid_forget()
                power_pres = 0

            tk.Button(frame_power, text='Remove Power Law component', command=hndl_remove_power_btn)\
                .grid(row=3, column=0, columnspan=5, pady=10)

            frame_power.grid(row=5, column=0, sticky="ew")
            for i in range(3):
                frame_power.grid_columnconfigure(i, weight=1)
            power_pres = 1
            
    def add_kappa():#function to add kappa law to gui/test function
        
        global kappa_pres#defining gui section to handle kappa law param options
        if kappa_pres ==0:
            
            global init#define global initial values for the params of the function
            
            init['A_k']=10**-22
            init['T_k']=300000000.0
            init['m_i']=9.11*1e-31
            init['n_i']=1e15
            init['kappa']=50
            
            global vary#define global if vary values for the params of the function
            
            vary['A_k']=True
            vary['T_k']=True
            vary['m_i']=False
            vary['n_i']=True
            vary['kappa']=True
            
            global minval##define global minimum values for the params of the function
            
            minval['A_k']=0
            minval['T_k']=1e6
            minval['m_i']=0
            minval['n_i']=None
            minval['kappa']=(3/2)+0.0001#must be greater than 3/2
            
            global maxval##define global maximum values for the params of the function
            
            maxval['A_k']=1
            maxval['T_k']=None
            maxval['m_i']=None
            maxval['n_i']=None
            maxval['kappa']=1000
            
            global frame_kappa
            frame_kappa=tk.Frame(master=frame_params)
            tk.Label(frame_kappa, text="Kappa Function", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_A_k(): vary['A_k'] = not vary['A_k']
            def toggle_T_k(): vary['T_k'] = not vary['T_k']
            def toggle_m_i(): vary['m_i'] = not vary['m_i']
            def toggle_n_i(): vary['n_i'] = not vary['n_i']
            def toggle_kappa(): vary['kappa'] = not vary['kappa']

            add_param_row(frame_kappa, 1, "A_k", init['A_k'], minval['A_k'], maxval['A_k'], vary['A_k'], toggle_A_k, "A_k")
            add_param_row(frame_kappa, 2, "T_k", init['T_k'], minval['T_k'], maxval['T_k'], vary['T_k'], toggle_T_k, "T_k")
            add_param_row(frame_kappa, 3, "m_i", init['m_i'], minval['m_i'], maxval['m_i'], vary['m_i'], toggle_m_i, "m_i")
            add_param_row(frame_kappa, 4, "n_i", init['n_i'], minval['n_i'], maxval['n_i'], vary['n_i'], toggle_n_i, "n_i")
            add_param_row(frame_kappa, 5, "kappa", init['kappa'], minval['kappa'], maxval['kappa'], vary['kappa'], toggle_kappa, "kappa")

            def hndl_remove_kappa_btn():
                global kappa_pres
                frame_kappa.grid_forget()
                kappa_pres = 0

            tk.Button(frame_kappa, text='Remove Kappa component', command=hndl_remove_kappa_btn)\
                .grid(row=6, column=0, columnspan=5, pady=10)

            frame_kappa.grid(row=6, column=0, sticky="ew")
            for i in range(6):
                frame_kappa.grid_columnconfigure(i, weight=1)
            kappa_pres = 1
            
        
    def add_bpl_and_therm():
        global bpl_and_therm_pres
        if bpl_and_therm_pres==0:
            global init #define global initial values for the 3 params of the thermal function
            init['amp_c']=1e9
            init['T_c']=12e6
            init['alpha_c']=1
            init['x0_c']=20
            init['x1_c']=50
            init['B_c']=-1 
            init['B2_c']=-2    
            
            global vary#define globally whether to initially vary for the 3 params of the thermal function
            vary['amp_c']=True
            vary['T_c']=True
            vary['alpha_c']=False
            vary['x0_c']=True
            vary['x1_c']=True
            vary['B_c']=True
            vary['B2_c']=True         
            
            global minval#define global initial minimum values for the 3 params of the thermal function
            minval['amp_c']=0
            minval['T_c']=0
            minval['alpha_c']=0
            minval['x0_c']=13
            minval['x1_c']=40
            minval['B_c']=-10
            minval['B2_c']=-10
         
            
            global maxval#define global initial maximum values for the 3 params of the thermal function
            maxval['amp_c']=None
            maxval['T_c']=1e8
            maxval['alpha_c']=5    
            maxval['x0_c']=25
            maxval['x1_c']=55
            maxval['B_c']=-0.1
            maxval['B2_c']=-0.1   

            
            global frame_bpl_and_therm#defining gui section to handle bpl param options
            frame_bpl_and_therm=tk.Frame(master=frame_params)
            
            tk.Label(frame_bpl_and_therm, text="BPL + Thermal", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_amp_c(): vary['amp_c'] = not vary['amp_c']
            def toggle_T_c(): vary['T_c'] = not vary['T_c']
            def toggle_alpha_c(): vary['alpha_c'] = not vary['alpha_c']
            def toggle_x1_c(): vary['x1_c'] = not vary['x1_c']
            def toggle_x0_c(): vary['x0_c'] = not vary['x0_c']
            def toggle_B_c(): vary['B_c'] = not vary['B_c']
            def toggle_B2_c(): vary['B2_c'] = not vary['B2_c']

            add_param_row(frame_bpl_and_therm, 1, "amp_c", init['amp_c'], minval['amp_c'], maxval['amp_c'], vary['amp_c'], toggle_amp_c, "amp_c")
            add_param_row(frame_bpl_and_therm, 2, "T_c", init['T_c'], minval['T_c'], maxval['T_c'], vary['T_c'], toggle_T_c, "T_c")
            add_param_row(frame_bpl_and_therm, 3, "alpha_c", init['alpha_c'], minval['alpha_c'], maxval['alpha_c'], vary['alpha_c'], toggle_alpha_c, "alpha_c")
            add_param_row(frame_bpl_and_therm, 4, "x0_c", init['x0_c'], minval['x0_c'], maxval['x0_c'], vary['x0_c'], toggle_x0_c, "x0_c")
            add_param_row(frame_bpl_and_therm, 5, "x1_c", init['x1_c'], minval['x1_c'], maxval['x1_c'], vary['x1_c'], toggle_x1_c, "x1_c")
            add_param_row(frame_bpl_and_therm, 6, "B_c", init['B_c'], minval['B_c'], maxval['B_c'], vary['B_c'], toggle_B_c, "B_c")
            add_param_row(frame_bpl_and_therm, 7, "B2_c", init['B2_c'], minval['B2_c'], maxval['B2_c'], vary['B2_c'], toggle_B2_c, "B2_c")

            def hndl_remove_bt_btn():
                global bpl_and_therm_pres
                frame_bpl_and_therm.grid_forget()
                bpl_and_therm_pres = 0

            tk.Button(frame_bpl_and_therm, text='Remove BPL + Thermal component', command=hndl_remove_bt_btn)\
                .grid(row=8, column=0, columnspan=5, pady=10)

            frame_bpl_and_therm.grid(row=7, column=0, sticky="ew")
            for i in range(8):
                frame_bpl_and_therm.grid_columnconfigure(i, weight=1)
            bpl_and_therm_pres = 1
    
    def add_double_therm():#add the thermal component to the fitted function
        
        global double_therm_func_pres #use global value of thermal function's presence
        if double_therm_func_pres ==0:#if thermal function not already there
            
            global init #define global initial values for the 3 params of the thermal function
            init['amp_d_1']=1e10
            init['T_d_1']=3e6
            init['alpha_d_1']=1
            init['amp_d_2']=1e8
            init['T_d_2']=16e6
            init['alpha_d_2']=1
            
            global vary#define globally whether to initially vary for the 3 params of the thermal function
            vary['amp_d_1']=True
            vary['T_d_1']=True
            vary['alpha_d_1']=False
            vary['amp_d_2']=True
            vary['T_d_2']=True
            vary['alpha_d_2']=False
            
            global minval#define global initial minimum values for the 3 params of the thermal function
            minval['amp_d_1']=0
            minval['T_d_1']=0
            minval['alpha_d_1']=0
            minval['amp_d_2']=0
            minval['T_d_2']=0
            minval['alpha_d_2']=0
            
            
            global maxval#define global initial maximum values for the 3 params of the thermal function
            maxval['amp_d_1']=None
            maxval['T_d_1']=1e8
            maxval['alpha_d_1']=5    
            maxval['amp_d_2']=None
            maxval['T_d_2']=1e8
            maxval['alpha_d_2']=5   
            
            
            #defining the part of the GUI window that contains the options for the thermal curve
            global frame_double_therm 
            frame_double_therm=tk.Frame(master=frame_params)
            tk.Label(frame_double_therm, text="Double Thermal", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_amp_d_1(): vary['amp_d_1'] = not vary['amp_d_1']
            def toggle_T_d_1(): vary['T_d_1'] = not vary['T_d_1']
            def toggle_alpha_d_1(): vary['alpha_d_1'] = not vary['alpha_d_1']
            def toggle_amp_d_2(): vary['amp_d_2'] = not vary['amp_d_2']
            def toggle_T_d_2(): vary['T_d_2'] = not vary['T_d_2']
            def toggle_alpha_d_2(): vary['alpha_d_2'] = not vary['alpha_d_2']

            add_param_row(frame_double_therm, 1, "amp_d_1", init['amp_d_1'], minval['amp_d_1'], maxval['amp_d_1'], vary['amp_d_1'], toggle_amp_d_1, "amp_d_1")
            add_param_row(frame_double_therm, 2, "T_d_1", init['T_d_1'], minval['T_d_1'], maxval['T_d_1'], vary['T_d_1'], toggle_T_d_1, "T_d_1")
            add_param_row(frame_double_therm, 3, "alpha_d_1", init['alpha_d_1'], minval['alpha_d_1'], maxval['alpha_d_1'], vary['alpha_d_1'], toggle_alpha_d_1, "alpha_d_1")
            add_param_row(frame_double_therm, 4, "amp_d_2", init['amp_d_2'], minval['amp_d_2'], maxval['amp_d_2'], vary['amp_d_2'], toggle_amp_d_2, "amp_d_2")
            add_param_row(frame_double_therm, 5, "T_d_2", init['T_d_2'], minval['T_d_2'], maxval['T_d_2'], vary['T_d_2'], toggle_T_d_2, "T_d_2")
            add_param_row(frame_double_therm, 6, "alpha_d_2", init['alpha_d_2'], minval['alpha_d_2'], maxval['alpha_d_2'], vary['alpha_d_2'], toggle_alpha_d_2, "alpha_d_2")

            def hndl_remove_double_therm_btn():
                global double_therm_pres
                frame_double_therm.grid_forget()
                double_therm_pres = 0

            tk.Button(frame_double_therm, text='Remove Double Thermal component', command=hndl_remove_double_therm_btn)\
                .grid(row=7, column=0, columnspan=5, pady=10)

            frame_double_therm.grid(row=8, column=0, sticky="ew")
            for i in range(7):
                frame_double_therm.grid_columnconfigure(i, weight=1)
            double_therm_pres = 1 #set the thermal function as present
    
    
    
    def add_tpl():#function to add the triple power law
        global tpl_pres
        if  tpl_pres ==0:
    
            global init#define global initial values for the params of the function
            
            init['x1_tpl']=11
            init['x2_tpl']=40
            init['A_tpl']=1e5
            init['B_tpl']=-2
            init['A2_tpl']=1e5
            init['B2_tpl']=-1    
            init['A3_tpl']=1e5
            init['B3_tpl']=-2   

    
            global vary#define global if vary values for the params of the function

            vary['x1_tpl']=True
            vary['x2_tpl']=True
            vary['A_tpl']=True
            vary['B_tpl']=True
            vary['A2_tpl']=True
            vary['B2_tpl']=True            
            vary['A3_tpl']=True
            vary['B3_tpl']=True            
            
            global maxval##define global maximum values for the params of the function

            maxval['x1_tpl']=50
            maxval['x2_tpl']=50
            maxval['A_tpl']=None
            maxval['B_tpl']=0
            maxval['A2_tpl']=None
            maxval['B2_tpl']=0            
            maxval['A3_tpl']=None
            maxval['B3_tpl']=0           

            
            global minval##define global minimum values for the params of the function

            minval['x1_tpl']=5
            minval['x2_tpl']=15
            minval['A_tpl']=0
            minval['B_tpl']=-10
            minval['A2_tpl']=0
            minval['B2_tpl']=-10            
            minval['A3_tpl']=0
            minval['B3_tpl']=-10            

            global frame_tpl#defining gui section to handle tpl param options
            frame_tpl=tk.Frame(master=frame_params)
            tk.Label(frame_tpl, text="Triple Power Law", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))
                

            def toggle_A_tpl(): vary['A_tpl'] = not vary['A_tpl']
            def toggle_B_tpl(): vary['B_tpl'] = not vary['B_tpl']
            
            def toggle_x1_tpl(): vary['x1_tpl'] = not vary['x1_tpl']
            def toggle_A2_tpl(): vary['A2_tpl'] = not vary['A2_tpl']
            def toggle_B2_tpl(): vary['B2_tpl'] = not vary['B2_tpl']
            
            def toggle_x2_tpl(): vary['x2_tpl'] = not vary['x2_tpl']
            def toggle_A3_tpl(): vary['A3_tpl'] = not vary['A3_tpl']
            def toggle_B3_tpl(): vary['B3_tpl'] = not vary['B3_tpl']


            add_param_row(frame_tpl, 1, "A_tpl", init['A_tpl'], minval['A_tpl'], maxval['A_tpl'], vary['A_tpl'], toggle_A_tpl, "A_tpl")
            add_param_row(frame_tpl, 2, "B_tpl", init['B_tpl'], minval['B_tpl'], maxval['B_tpl'], vary['B_tpl'], toggle_B_tpl, "B_tpl")
            add_param_row(frame_tpl, 3, "x1_tpl", init['x1_tpl'], minval['x1_tpl'], maxval['x1_tpl'], vary['x1_tpl'], toggle_x1_tpl, "x1_tpl")
            add_param_row(frame_tpl, 4, "A2_tpl", init['A2_tpl'], minval['A2_tpl'], maxval['A2_tpl'], vary['A2_tpl'], toggle_A2_tpl, "A2_tpl")
            add_param_row(frame_tpl, 5, "B2_tpl", init['B2_tpl'], minval['B2_tpl'], maxval['B2_tpl'], vary['B2_tpl'], toggle_B2_tpl, "B2_tpl")
            add_param_row(frame_tpl, 6, "x2_tpl", init['x2_tpl'], minval['x2_tpl'], maxval['x2_tpl'], vary['x2_tpl'], toggle_x2_tpl, "x2_tpl")
            add_param_row(frame_tpl, 7, "A3_tpl", init['A3_tpl'], minval['A3_tpl'], maxval['A3_tpl'], vary['A3_tpl'], toggle_A3_tpl, "A3_tpl")
            add_param_row(frame_tpl, 8, "B3_tpl", init['B3_tpl'], minval['B3_tpl'], maxval['B3_tpl'], vary['B3_tpl'], toggle_B3_tpl, "B3_tpl")
      
            

            def hndl_remove_tpl_btn():
                global tpl_pres
                frame_tpl.grid_forget()
                tpl_pres = 0

            tk.Button(frame_tpl, text='Remove TPL component', command=hndl_remove_tpl_btn)\
                .grid(row=9, column=0, columnspan=5, pady=10)

            frame_tpl.grid(row=9, column=0, sticky="ew")
            for i in range(9):
                frame_tpl.grid_columnconfigure(i, weight=1)
            tpl_pres = 1
    
    
    def add_qpl():#function to add the quad power law
        global qpl_pres
        if  qpl_pres ==0:
    
            global init#define global initial values for the params of the function
            
            init['x1_qpl']=5
            init['x2_qpl']=11
            init['x3_qpl']=40
            init['A_qpl']=1e8
            init['B_qpl']=-1
            init['A2_qpl']=1e9
            init['B2_qpl']=-2    
            init['A3_qpl']=1.5e8
            init['B3_qpl']=-1   
            init['A4_qpl']=1e9
            init['B4_qpl']=-2 
    
            global vary#define global if vary values for the params of the function

            vary['x1_qpl']=True
            vary['x2_qpl']=True
            vary['x3_qpl']=True
            vary['A_qpl']=True
            vary['B_qpl']=True
            vary['A2_qpl']=True
            vary['B2_qpl']=True            
            vary['A3_qpl']=True
            vary['B3_qpl']=True
            vary['A4_qpl']=True
            vary['B4_qpl']=True            
            
            global maxval##define global maximum values for the params of the function

            maxval['x1_qpl']=10
            maxval['x2_qpl']=20
            maxval['x3_qpl']=50
            maxval['A_qpl']=None
            maxval['B_qpl']=0
            maxval['A2_qpl']=None
            maxval['B2_qpl']=0            
            maxval['A3_qpl']=None
            maxval['B3_qpl']=0           
            maxval['A4_qpl']=None
            maxval['B4_qpl']=0           
            
            global minval##define global minimum values for the params of the function

            minval['x1_qpl']=0
            minval['x2_qpl']=5
            minval['x3_qpl']=30
            minval['A_qpl']=0
            minval['B_qpl']=-10
            minval['A2_qpl']=0
            minval['B2_qpl']=-10            
            minval['A3_qpl']=0
            minval['B3_qpl']=-10
            minval['A4_qpl']=0
            minval['B4_qpl']=-10            

            global frame_qpl#defining gui section to handle qpl param options
            frame_qpl=tk.Frame(master=frame_params)
            
            tk.Label(frame_qpl, text="Quadruple Power Law", font=("Arial", 12, "bold"))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_A_qpl(): vary['A_qpl'] = not vary['A_qpl']
            def toggle_B_qpl(): vary['B_qpl'] = not vary['B_qpl']
            
            def toggle_x1_qpl(): vary['x1_qpl'] = not vary['x1_qpl']
            def toggle_A2_qpl(): vary['A2_qpl'] = not vary['A2_qpl']
            def toggle_B2_qpl(): vary['B2_qpl'] = not vary['B2_qpl']
            
            def toggle_x2_qpl(): vary['x2_qpl'] = not vary['x2_qpl']
            def toggle_A3_qpl(): vary['A3_qpl'] = not vary['A3_qpl']
            def toggle_B3_qpl(): vary['B3_qpl'] = not vary['B3_qpl']
            
            def toggle_x3_qpl(): vary['x3_qpl'] = not vary['x3_qpl']
            def toggle_A4_qpl(): vary['A4_qpl'] = not vary['A4_qpl']
            def toggle_B4_qpl(): vary['B4_qpl'] = not vary['B4_qpl']
            
            
            

            


            add_param_row(frame_qpl, 1, "A_qpl", init['A_qpl'], minval['A_qpl'], maxval['A_qpl'], vary['A_qpl'], toggle_A_qpl, "A_qpl")
            add_param_row(frame_qpl, 2, "B_qpl", init['B_qpl'], minval['B_qpl'], maxval['B_qpl'], vary['B_qpl'], toggle_B_qpl, "B_qpl")
            
            add_param_row(frame_qpl, 3, "x1_qpl", init['x1_qpl'], minval['x1_qpl'], maxval['x1_qpl'], vary['x1_qpl'], toggle_x1_qpl, "x1_qpl")
            add_param_row(frame_qpl, 4, "A2_qpl", init['A2_qpl'], minval['A2_qpl'], maxval['A2_qpl'], vary['A2_qpl'], toggle_A2_qpl, "A2_qpl")
            add_param_row(frame_qpl, 5, "B2_qpl", init['B2_qpl'], minval['B2_qpl'], maxval['B2_qpl'], vary['B2_qpl'], toggle_B2_qpl, "B2_qpl")
            
            add_param_row(frame_qpl, 6, "x2_qpl", init['x2_qpl'], minval['x2_qpl'], maxval['x2_qpl'], vary['x2_qpl'], toggle_x2_qpl, "x2_qpl")
            add_param_row(frame_qpl, 7, "A3_qpl", init['A3_qpl'], minval['A3_qpl'], maxval['A3_qpl'], vary['A3_qpl'], toggle_A3_qpl, "A3_qpl")
            add_param_row(frame_qpl, 8, "B3_qpl", init['B3_qpl'], minval['B3_qpl'], maxval['B3_qpl'], vary['B3_qpl'], toggle_B3_qpl, "B3_qpl")
            
            add_param_row(frame_qpl, 9, "x3_qpl", init['x3_qpl'], minval['x3_qpl'], maxval['x3_qpl'], vary['x3_qpl'], toggle_x3_qpl, "x3_qpl")
            add_param_row(frame_qpl, 10, "A4_qpl", init['A4_qpl'], minval['A4_qpl'], maxval['A4_qpl'], vary['A4_qpl'], toggle_A4_qpl, "A4_qpl")
            add_param_row(frame_qpl, 11, "B4_qpl", init['B4_qpl'], minval['B4_qpl'], maxval['B4_qpl'], vary['B4_qpl'], toggle_B4_qpl, "B4_qpl")
            
         


            def hndl_remove_qpl_btn():
                global qpl_pres
                frame_qpl.grid_forget()
                qpl_pres = 0

            tk.Button(frame_qpl, text='Remove QPL component', command=hndl_remove_qpl_btn)\
                .grid(row=12, column=0, columnspan=5, pady=10)

            frame_qpl.grid(row=10, column=0, sticky="ew")
            for i in range(12):
                frame_qpl.grid_columnconfigure(i, weight=1)
            qpl_pres = 1
    
    def add_quint_pl():#function to add the quad power law
        global quint_pl_pres
        if  quint_pl_pres ==0:
    
            global init#define global initial values for the params of the function
            
            init['x1_5pl']=2
            init['x2_5pl']=5
            init['x3_5pl']=11
            init['x4_5pl']=40
            init['A_5pl']=1e8
            init['B_5pl']=-1
            init['A2_5pl']=1e9
            init['B2_5pl']=-2    
            init['A3_5pl']=1.5e8
            init['B3_5pl']=-1   
            init['A4_5pl']=1e9
            init['B4_5pl']=-2 
            init['A5_5pl']=1e9
            init['B5_5pl']=-2 
    
            global vary#define global if vary values for the params of the function

            vary['x1_5pl']=True
            vary['x2_5pl']=True
            vary['x3_5pl']=True
            vary['x4_5pl']=True
            vary['A_5pl']=True
            vary['B_5pl']=True
            vary['A2_5pl']=True
            vary['B2_5pl']=True            
            vary['A3_5pl']=True
            vary['B3_5pl']=True
            vary['A4_5pl']=True
            vary['B4_5pl']=True
            vary['A5_5pl']=True
            vary['B5_5pl']=True            
            
            global maxval##define global maximum values for the params of the function

            maxval['x1_5pl']=10
            maxval['x2_5pl']=20
            maxval['x3_5pl']=50
            maxval['x4_5pl']=50
            maxval['A_5pl']=None
            maxval['B_5pl']=0
            maxval['A2_5pl']=None
            maxval['B2_5pl']=0            
            maxval['A3_5pl']=None
            maxval['B3_5pl']=0           
            maxval['A4_5pl']=None
            maxval['B4_5pl']=0     
            maxval['A5_5pl']=None
            maxval['B5_5pl']=0 
            
            global minval##define global minimum values for the params of the function

            minval['x1_5pl']=0
            minval['x2_5pl']=5
            minval['x3_5pl']=10
            minval['x4_5pl']=30
            minval['A_5pl']=0
            minval['B_5pl']=-10
            minval['A2_5pl']=0
            minval['B2_5pl']=-10            
            minval['A3_5pl']=0
            minval['B3_5pl']=-10
            minval['A4_5pl']=0
            minval['B4_5pl']=-10
            minval['A5_5pl']=0
            minval['B5_5pl']=-10            

            global frame_quint_pl#defining gui section to handle quint_pl param options
            frame_quint_pl=tk.Frame(master=frame_params)
            
            tk.Label(frame_quint_pl, text="Quintuple Power Law", font=("Arial", 12, "bold"))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle(var):
                vary[var] = not vary[var]

            row_counter = 1
            for name in ["A_5pl", "B_5pl","x1_5pl","A2_5pl", "B2_5pl","x2_5pl", "A3_5pl","B3_5pl","x3_5pl","A4_5pl", "B4_5pl", "x4_5pl","A5_5pl", "B5_5pl"]:
                add_param_row(
                    frame_quint_pl,
                    row_counter,
                    name,
                    init[name],
                    minval[name],
                    maxval[name],
                    vary[name],
                    lambda n=name: toggle(n),
                    name
                )
                row_counter += 1

            def hndl_remove_quint_btn():
                global quint_pl_pres
                frame_quint_pl.grid_forget()
                quint_pl_pres = 0

            tk.Button(frame_quint_pl, text='Remove Quintuple Power Law component', command=hndl_remove_quint_btn)\
                .grid(row=row_counter, column=0, columnspan=5, pady=10)

            frame_quint_pl.grid(row=11, column=0, sticky="ew")
            for i in range(row_counter):
                frame_quint_pl.grid_columnconfigure(i, weight=1)
            quint_pl_pres = 1
    
    

    
    
    #dropdown menu to add function components
    frame_fitopts=tk.Frame(master=window_buttons)
    # Options for the dropdown menu
    OPTIONS = [
        "thermal",
        "broken power law",
        "power law",
        "Gaussian",
        "kappa function",
        "broken power law + thermal",
        "double thermal",
        "triple power law",
        "quadruple power law",
        "quintuple power law"
    ]
    
    # Variable to hold the selected option
    variable_o = tk.StringVar()
    variable_o.set(OPTIONS[5])  # Default value
    
    # Create the dropdown menu
    fit_opts = tk.OptionMenu(frame_fitopts, variable_o, *OPTIONS)
    fit_opts.pack()
    
    # Function to handle the button click
    def fit_opts_select():
        selected_func = variable_o.get()
        
        if selected_func == 'thermal':
            add_therm()
        elif selected_func == 'broken power law':
            add_bpl()
        elif selected_func == 'Gaussian':
            add_gauss()
        elif selected_func == 'power law':
            add_power()
        elif selected_func == 'kappa function':
            add_kappa()
        elif selected_func == 'broken power law + thermal':
            add_bpl_and_therm()
        elif selected_func == 'double thermal':
            add_double_therm()
        elif selected_func == 'triple power law':
            add_tpl()
        elif selected_func == "quadruple power law":
            add_qpl()
        elif selected_func == "quintuple power law":
            add_quint_pl()
 
    

    button = tk.Button(master=frame_fitopts, text="ADD COMPONENT", command=fit_opts_select)#button to add selected function
    button.pack()

    frame_fitopts.pack(side="top")   
         
   

    
    frame_fitlims=tk.Frame(master=window_buttons)#part of gui to handle user definition of energt range to fit over
    label_fitlims=tk.Label(master=frame_fitlims, text='Range to fit')
    label_fitlims.pack(side=tk.LEFT)
    label_fitmin=tk.Label(master=frame_fitlims, text='     Min:')
    label_fitmin.pack(side=tk.LEFT)
    fitmin_entry = tk.Entry(master=frame_fitlims,fg="black", bg="white", width=10)
    fitmin_entry.pack(side=tk.LEFT)
    fitmin_entry.insert(0,str(min(x_data)))    
    
    label_fitmax=tk.Label(master=frame_fitlims, text='     Max:')
    label_fitmax.pack(side=tk.LEFT)
    fitmax_entry = tk.Entry(master=frame_fitlims,fg="black", bg="white", width=10)
    fitmax_entry.pack(side=tk.LEFT)
    fitmax_entry.insert(0,str(max(x_data)))
    frame_fitlims.pack(side="top")
    
    def validate_minmaxval(min_val,max_val): #max must be float greater than minval or none
        if (type(min_val)==float and type(max_val)==float and max_val>min_val) or (min_val==None and type(max_val)==float) or (type(min_val)==float and max_val==None) or (min_val==None and max_val==None):
            return True
        return False

    def validate_init(init_val,min_val,max_val):#must be float between max and min val
        if type(init)==float and init_val>min_val and  max_val>init_val:
            return True
        return False
    def validate_lims(min_val,max_val):
        if (type(min_val)==float and type(max_val)==float and max_val>min_val):
            return True
        return False
        
    def fit_btn_hndl():#function to handle button to perform fit
        global init
        global vary
        global minval
        global maxval
        global fit_window
        global preview_window
        global resid_window
        if fit_window is not None:
            #close any open figues
            fit_window.destroy()
            fit_window=None
            
        if preview_window is not None:
            #close any open figues
            preview_window.destroy()
            preview_window=None
            
        if resid_window is not None:
            #close any open figues
            resid_window.destroy()
            resid_window=None
        
        global header
        header=f"bpl_pres={bpl_pres}; therm_func_pres={therm_func_pres}; gauss_pres={gauss_pres}; power_pres={power_pres}; kappa_pres={kappa_pres}; bpl_and_therm_pres={bpl_and_therm_pres}; double_therm_func_pres={double_therm_func_pres}; tpl_pres={tpl_pres}; qpl_pres={qpl_pres}; quint_pl_pres={quint_pl_pres};"#defines header according to what functions are currently present in the gui
        try:#try excpet statement is to validate inputs as integers
            if therm_func_pres==1:#if thermal function present, save parameter options from the gui for that function
                
                global frame_therm
                
                init['T']=None if init_T_entry.get()=='None' else float(init_T_entry.get())
                minval['T']=None if minval_T_entry.get()=='None' else float(minval_T_entry.get())
                maxval['T']=None if maxval_T_entry.get()=='None' else float(maxval_T_entry.get())
                
                init['amp']=None if init_amp_entry.get()=='None' else float(init_amp_entry.get())
                minval['amp']=None if minval_amp_entry.get()=='None' else float(minval_amp_entry.get())
                maxval['amp']=None if maxval_amp_entry.get()=='None' else float(maxval_amp_entry.get())
                
                init['alpha']=None if init_alpha_entry.get()=='None' else float(init_alpha_entry.get())
                minval['alpha']=None if minval_alpha_entry.get()=='None' else float(minval_alpha_entry.get())
                maxval['alpha']=None if maxval_alpha_entry.get()=='None' else float(maxval_alpha_entry.get())
                
            if bpl_pres==1:#if bpl function present, save parameter options from the gui for that function
                
                
                
                init['x1']=None if init_x1_entry.get()=='None' else float(init_x1_entry.get())
                minval['x1']=None if minval_x1_entry.get()=='None' else float(minval_x1_entry.get())
                maxval['x1']=None if maxval_x1_entry.get()=='None' else float(maxval_x1_entry.get())
                
                
                init['A2']=None if init_A2_entry.get()=='None' else float(init_A2_entry.get())
                minval['A2']=None if minval_A2_entry.get()=='None' else float(minval_A2_entry.get())
                maxval['A2']=None if maxval_A2_entry.get()=='None' else float(maxval_A2_entry.get())
                
                
                init['B2']=None if init_B2_entry.get()=='None' else float(init_B2_entry.get())
                minval['B2']=None if minval_B2_entry.get()=='None' else float(minval_B2_entry.get())
                maxval['B2']=None if maxval_B2_entry.get()=='None' else float(maxval_B2_entry.get())
                
                
                init['B']=None if init_B_entry.get()=='None' else float(init_B_entry.get())
                minval['B']=None if minval_B_entry.get()=='None' else float(minval_B_entry.get())
                maxval['B']=None if maxval_B_entry.get()=='None' else float( maxval_B_entry.get())
    
    
                init['A']=None if init_A_entry.get()=='None' else float(init_A_entry.get())
                minval['A']=None if minval_A_entry.get()=='None' else float(minval_A_entry.get())
                maxval['A']=None if maxval_A_entry.get()=='None' else float(maxval_A_entry.get())
                
            if gauss_pres==1:#if gaussian function present, save parameter options from the gui for that function
                init['gauss_centre']=None if init_gauss_centre_entry.get()=='None' else float(init_gauss_centre_entry.get())
                minval['gauss_centre']=None if minval_gauss_centre_entry.get()=='None' else float(minval_gauss_centre_entry.get())
                maxval['gauss_centre']=None if maxval_gauss_centre_entry.get()=='None' else float(maxval_gauss_centre_entry.get())
                
                
                init['gauss_amp']=None if init_gauss_amp_entry.get()=='None' else float(init_gauss_amp_entry.get())
                minval['gauss_amp']=None if minval_gauss_amp_entry.get()=='None' else float(minval_gauss_amp_entry.get())
                maxval['gauss_amp']=None if maxval_gauss_amp_entry.get()=='None' else float( maxval_gauss_amp_entry.get())
    
    
                init['sigma']=None if init_sigma_entry.get()=='None' else float(init_sigma_entry.get())
                minval['sigma']=None if minval_sigma_entry.get()=='None' else float(minval_sigma_entry.get())
                maxval['sigma']=None if maxval_sigma_entry.get()=='None' else float(maxval_sigma_entry.get())
               
               
               
               
               
            if power_pres==1:#if single power law function present, save parameter options from the gui for that function
               init['B_sing']=None if init_B_sing_entry.get()=='None' else float(init_B_sing_entry.get())
               minval['B_sing']=None if minval_B_sing_entry.get()=='None' else float(minval_B_sing_entry.get())
               maxval['B_sing']=None if maxval_B_sing_entry.get()=='None' else float( maxval_B_sing_entry.get())
    
    
               init['A_sing']=None if init_A_sing_entry.get()=='None' else float(init_A_sing_entry.get())
               minval['A_sing']=None if minval_A_sing_entry.get()=='None' else float(minval_A_sing_entry.get())
               maxval['A_sing']=None if maxval_A_sing_entry.get()=='None' else float(maxval_A_sing_entry.get())
               

               
            if kappa_pres==1:#if kappa function present, save parameter options from the gui for that function
   
               init['A_k']=None if init_A_k_entry.get()=='None' else float(init_A_k_entry.get())
               minval['A_k']=None if minval_A_k_entry.get()=='None' else float(minval_A_k_entry.get())
               maxval['A_k']=None if maxval_A_k_entry.get()=='None' else float(maxval_A_k_entry.get())
               
               init['T_k']=None if init_T_k_entry.get()=='None' else float(init_T_k_entry.get())
               minval['T_k']=None if minval_T_k_entry.get()=='None' else float(minval_T_k_entry.get())
               maxval['T_k']=None if maxval_T_k_entry.get()=='None' else float( maxval_T_k_entry.get())
               
               init['m_i']=None if init_m_i_entry.get()=='None' else float(init_m_i_entry.get())
               minval['m_i']=None if minval_m_i_entry.get()=='None' else float(minval_m_i_entry.get())
               maxval['m_i']=None if maxval_m_i_entry.get()=='None' else float( maxval_m_i_entry.get())
               
               init['n_i']=None if init_n_i_entry.get()=='None' else float(init_n_i_entry.get())
               minval['n_i']=None if minval_n_i_entry.get()=='None' else float(minval_n_i_entry.get())
               maxval['n_i']=None if maxval_n_i_entry.get()=='None' else float( maxval_n_i_entry.get())    
 
               init['kappa']=None if init_kappa_entry.get()=='None' else float(init_kappa_entry.get())
               minval['kappa']=None if minval_kappa_entry.get()=='None' else float(minval_kappa_entry.get())
               maxval['kappa']=None if maxval_kappa_entry.get()=='None' else float( maxval_kappa_entry.get())
               
               
            if bpl_and_therm_pres==1:
                init['T_c']=None if init_T_c_entry.get()=='None' else float(init_T_c_entry.get())
                minval['T_c']=None if minval_T_c_entry.get()=='None' else float(minval_T_c_entry.get())
                maxval['T_c']=None if maxval_T_c_entry.get()=='None' else float(maxval_T_c_entry.get())
                
                init['amp_c']=None if init_amp_c_entry.get()=='None' else float(init_amp_c_entry.get())
                minval['amp_c']=None if minval_amp_c_entry.get()=='None' else float(minval_amp_c_entry.get())
                maxval['amp_c']=None if maxval_amp_c_entry.get()=='None' else float(maxval_amp_c_entry.get())
                
                init['alpha_c']=None if init_alpha_c_entry.get()=='None' else float(init_alpha_c_entry.get())
                minval['alpha_c']=None if minval_alpha_c_entry.get()=='None' else float(minval_alpha_c_entry.get())
                maxval['alpha_c']=None if maxval_alpha_c_entry.get()=='None' else float(maxval_alpha_c_entry.get())
                
                init['x1_c']=None if init_x1_c_entry.get()=='None' else float(init_x1_c_entry.get())
                minval['x1_c']=None if minval_x1_c_entry.get()=='None' else float(minval_x1_c_entry.get())
                maxval['x1_c']=None if maxval_x1_c_entry.get()=='None' else float(maxval_x1_c_entry.get())
                
                init['x0_c']=None if init_x0_c_entry.get()=='None' else float(init_x0_c_entry.get())
                minval['x0_c']=None if minval_x0_c_entry.get()=='None' else float(minval_x0_c_entry.get())
                maxval['x0_c']=None if maxval_x0_c_entry.get()=='None' else float(maxval_x0_c_entry.get())
                
                init['B2_c']=None if init_B2_c_entry.get()=='None' else float(init_B2_c_entry.get())
                minval['B2_c']=None if minval_B2_c_entry.get()=='None' else float(minval_B2_c_entry.get())
                maxval['B2_c']=None if maxval_B2_c_entry.get()=='None' else float(maxval_B2_c_entry.get())
                
                
                init['B_c']=None if init_B_c_entry.get()=='None' else float(init_B_c_entry.get())
                minval['B_c']=None if minval_B_c_entry.get()=='None' else float(minval_B_c_entry.get())
                maxval['B_c']=None if maxval_B_c_entry.get()=='None' else float( maxval_B_c_entry.get())
                
            if double_therm_func_pres==1:#if double thermal function present, save parameter options from the gui for that function
                
                
                
                init['T_d_1']=None if init_T_d_1_entry.get()=='None' else float(init_T_d_1_entry.get())
                minval['T_d_1']=None if minval_T_d_1_entry.get()=='None' else float(minval_T_d_1_entry.get())
                maxval['T_d_1']=None if maxval_T_d_1_entry.get()=='None' else float(maxval_T_d_1_entry.get())
                
                init['amp_d_1']=None if init_amp_d_1_entry.get()=='None' else float(init_amp_d_1_entry.get())
                minval['amp_d_1']=None if minval_amp_d_1_entry.get()=='None' else float(minval_amp_d_1_entry.get())
                maxval['amp_d_1']=None if maxval_amp_d_1_entry.get()=='None' else float(maxval_amp_d_1_entry.get())
                
                init['alpha_d_1']=None if init_alpha_d_1_entry.get()=='None' else float(init_alpha_d_1_entry.get())
                minval['alpha_d_1']=None if minval_alpha_d_1_entry.get()=='None' else float(minval_alpha_d_1_entry.get())
                maxval['alpha_d_1']=None if maxval_alpha_d_1_entry.get()=='None' else float(maxval_alpha_d_1_entry.get())

                init['T_d_2']=None if init_T_d_2_entry.get()=='None' else float(init_T_d_2_entry.get())
                minval['T_d_2']=None if minval_T_d_2_entry.get()=='None' else float(minval_T_d_2_entry.get())
                maxval['T_d_2']=None if maxval_T_d_2_entry.get()=='None' else float(maxval_T_d_2_entry.get())
                
                init['amp_d_2']=None if init_amp_d_2_entry.get()=='None' else float(init_amp_d_2_entry.get())
                minval['amp_d_2']=None if minval_amp_d_2_entry.get()=='None' else float(minval_amp_d_2_entry.get())
                maxval['amp_d_2']=None if maxval_amp_d_2_entry.get()=='None' else float(maxval_amp_d_2_entry.get())
                
                init['alpha_d_2']=None if init_alpha_d_2_entry.get()=='None' else float(init_alpha_d_2_entry.get())
                minval['alpha_d_2']=None if minval_alpha_d_2_entry.get()=='None' else float(minval_alpha_d_2_entry.get())
                maxval['alpha_d_2']=None if maxval_alpha_d_2_entry.get()=='None' else float(maxval_alpha_d_2_entry.get())
            
            
            if tpl_pres==1:#if tpl function present, save parameter options from the gui for that function
                init['x1_tpl']=None if init_x1_tpl_entry.get()=='None' else float(init_x1_tpl_entry.get())
                minval['x1_tpl']=None if minval_x1_tpl_entry.get()=='None' else float(minval_x1_tpl_entry.get())
                maxval['x1_tpl']=None if maxval_x1_tpl_entry.get()=='None' else float(maxval_x1_tpl_entry.get())
                
                                
                init['x2_tpl']=None if init_x2_tpl_entry.get()=='None' else float(init_x2_tpl_entry.get())
                minval['x2_tpl']=None if minval_x2_tpl_entry.get()=='None' else float(minval_x2_tpl_entry.get())
                maxval['x2_tpl']=None if maxval_x2_tpl_entry.get()=='None' else float(maxval_x2_tpl_entry.get())
                
                
                init['A2_tpl']=None if init_A2_tpl_entry.get()=='None' else float(init_A2_tpl_entry.get())
                minval['A2_tpl']=None if minval_A2_tpl_entry.get()=='None' else float(minval_A2_tpl_entry.get())
                maxval['A2_tpl']=None if maxval_A2_tpl_entry.get()=='None' else float(maxval_A2_tpl_entry.get())
                
                
                init['B2_tpl']=None if init_B2_tpl_entry.get()=='None' else float(init_B2_tpl_entry.get())
                minval['B2_tpl']=None if minval_B2_tpl_entry.get()=='None' else float(minval_B2_tpl_entry.get())
                maxval['B2_tpl']=None if maxval_B2_tpl_entry.get()=='None' else float(maxval_B2_tpl_entry.get())
                
                
                init['B_tpl']=None if init_B_tpl_entry.get()=='None' else float(init_B_tpl_entry.get())
                minval['B_tpl']=None if minval_B_tpl_entry.get()=='None' else float(minval_B_tpl_entry.get())
                maxval['B_tpl']=None if maxval_B_tpl_entry.get()=='None' else float( maxval_B_tpl_entry.get())
            
            
                init['A_tpl']=None if init_A_tpl_entry.get()=='None' else float(init_A_tpl_entry.get())
                minval['A_tpl']=None if minval_A_tpl_entry.get()=='None' else float(minval_A_tpl_entry.get())
                maxval['A_tpl']=None if maxval_A_tpl_entry.get()=='None' else float(maxval_A_tpl_entry.get())
                
                init['A3_tpl']=None if init_A3_tpl_entry.get()=='None' else float(init_A3_tpl_entry.get())
                minval['A3_tpl']=None if minval_A3_tpl_entry.get()=='None' else float(minval_A3_tpl_entry.get())
                maxval['A3_tpl']=None if maxval_A3_tpl_entry.get()=='None' else float(maxval_A3_tpl_entry.get())
                
                
                init['B3_tpl']=None if init_B3_tpl_entry.get()=='None' else float(init_B3_tpl_entry.get())
                minval['B3_tpl']=None if minval_B3_tpl_entry.get()=='None' else float(minval_B3_tpl_entry.get())
                maxval['B3_tpl']=None if maxval_B3_tpl_entry.get()=='None' else float(maxval_B3_tpl_entry.get())
            
            if qpl_pres==1:#if qpl function present, save parameter options from the gui for that function
                init['x1_qpl']=None if init_x1_qpl_entry.get()=='None' else float(init_x1_qpl_entry.get())
                minval['x1_qpl']=None if minval_x1_qpl_entry.get()=='None' else float(minval_x1_qpl_entry.get())
                maxval['x1_qpl']=None if maxval_x1_qpl_entry.get()=='None' else float(maxval_x1_qpl_entry.get())
                
                                
                init['x2_qpl']=None if init_x2_qpl_entry.get()=='None' else float(init_x2_qpl_entry.get())
                minval['x2_qpl']=None if minval_x2_qpl_entry.get()=='None' else float(minval_x2_qpl_entry.get())
                maxval['x2_qpl']=None if maxval_x2_qpl_entry.get()=='None' else float(maxval_x2_qpl_entry.get())
                
                init['x3_qpl']=None if init_x3_qpl_entry.get()=='None' else float(init_x3_qpl_entry.get())
                minval['x3_qpl']=None if minval_x3_qpl_entry.get()=='None' else float(minval_x3_qpl_entry.get())
                maxval['x3_qpl']=None if maxval_x3_qpl_entry.get()=='None' else float(maxval_x3_qpl_entry.get())
                
                init['A2_qpl']=None if init_A2_qpl_entry.get()=='None' else float(init_A2_qpl_entry.get())
                minval['A2_qpl']=None if minval_A2_qpl_entry.get()=='None' else float(minval_A2_qpl_entry.get())
                maxval['A2_qpl']=None if maxval_A2_qpl_entry.get()=='None' else float(maxval_A2_qpl_entry.get())
                
                
                init['B2_qpl']=None if init_B2_qpl_entry.get()=='None' else float(init_B2_qpl_entry.get())
                minval['B2_qpl']=None if minval_B2_qpl_entry.get()=='None' else float(minval_B2_qpl_entry.get())
                maxval['B2_qpl']=None if maxval_B2_qpl_entry.get()=='None' else float(maxval_B2_qpl_entry.get())
                
                
                init['B_qpl']=None if init_B_qpl_entry.get()=='None' else float(init_B_qpl_entry.get())
                minval['B_qpl']=None if minval_B_qpl_entry.get()=='None' else float(minval_B_qpl_entry.get())
                maxval['B_qpl']=None if maxval_B_qpl_entry.get()=='None' else float( maxval_B_qpl_entry.get())
            
            
                init['A_qpl']=None if init_A_qpl_entry.get()=='None' else float(init_A_qpl_entry.get())
                minval['A_qpl']=None if minval_A_qpl_entry.get()=='None' else float(minval_A_qpl_entry.get())
                maxval['A_qpl']=None if maxval_A_qpl_entry.get()=='None' else float(maxval_A_qpl_entry.get())
                
                init['A3_qpl']=None if init_A3_qpl_entry.get()=='None' else float(init_A3_qpl_entry.get())
                minval['A3_qpl']=None if minval_A3_qpl_entry.get()=='None' else float(minval_A3_qpl_entry.get())
                maxval['A3_qpl']=None if maxval_A3_qpl_entry.get()=='None' else float(maxval_A3_qpl_entry.get())
                
                
                init['B3_qpl']=None if init_B3_qpl_entry.get()=='None' else float(init_B3_qpl_entry.get())
                minval['B3_qpl']=None if minval_B3_qpl_entry.get()=='None' else float(minval_B3_qpl_entry.get())
                maxval['B3_qpl']=None if maxval_B3_qpl_entry.get()=='None' else float(maxval_B3_qpl_entry.get())
                
                init['A4_qpl']=None if init_A4_qpl_entry.get()=='None' else float(init_A4_qpl_entry.get())
                minval['A4_qpl']=None if minval_A4_qpl_entry.get()=='None' else float(minval_A4_qpl_entry.get())
                maxval['A4_qpl']=None if maxval_A4_qpl_entry.get()=='None' else float(maxval_A4_qpl_entry.get())
                
                init['B4_qpl']=None if init_B4_qpl_entry.get()=='None' else float(init_B4_qpl_entry.get())
                minval['B4_qpl']=None if minval_B4_qpl_entry.get()=='None' else float(minval_B4_qpl_entry.get())
                maxval['B4_qpl']=None if maxval_B4_qpl_entry.get()=='None' else float(maxval_B4_qpl_entry.get())
                
            if quint_pl_pres==1:#if 5pl function present, save parameter options from the gui for that function
                init['x1_5pl']=None if init_x1_5pl_entry.get()=='None' else float(init_x1_5pl_entry.get())
                minval['x1_5pl']=None if minval_x1_5pl_entry.get()=='None' else float(minval_x1_5pl_entry.get())
                maxval['x1_5pl']=None if maxval_x1_5pl_entry.get()=='None' else float(maxval_x1_5pl_entry.get())
                
                                
                init['x2_5pl']=None if init_x2_5pl_entry.get()=='None' else float(init_x2_5pl_entry.get())
                minval['x2_5pl']=None if minval_x2_5pl_entry.get()=='None' else float(minval_x2_5pl_entry.get())
                maxval['x2_5pl']=None if maxval_x2_5pl_entry.get()=='None' else float(maxval_x2_5pl_entry.get())
                
                init['x3_5pl']=None if init_x3_5pl_entry.get()=='None' else float(init_x3_5pl_entry.get())
                minval['x3_5pl']=None if minval_x3_5pl_entry.get()=='None' else float(minval_x3_5pl_entry.get())
                maxval['x3_5pl']=None if maxval_x3_5pl_entry.get()=='None' else float(maxval_x3_5pl_entry.get())
                
                init['x4_5pl']=None if init_x4_entry.get()=='None' else float(init_x4_entry.get())
                minval['x4_5pl']=None if minval_x4_entry.get()=='None' else float(minval_x4_entry.get())
                maxval['x4_5pl']=None if maxval_x4_entry.get()=='None' else float(maxval_x4_entry.get())
                
                init['A2_5pl']=None if init_A2_5pl_entry.get()=='None' else float(init_A2_5pl_entry.get())
                minval['A2_5pl']=None if minval_A2_5pl_entry.get()=='None' else float(minval_A2_5pl_entry.get())
                maxval['A2_5pl']=None if maxval_A2_5pl_entry.get()=='None' else float(maxval_A2_5pl_entry.get())
                
                
                init['B2_5pl']=None if init_B2_5pl_entry.get()=='None' else float(init_B2_5pl_entry.get())
                minval['B2_5pl']=None if minval_B2_5pl_entry.get()=='None' else float(minval_B2_5pl_entry.get())
                maxval['B2_5pl']=None if maxval_B2_5pl_entry.get()=='None' else float(maxval_B2_5pl_entry.get())
                
                
                init['B_5pl']=None if init_B_5pl_entry.get()=='None' else float(init_B_5pl_entry.get())
                minval['B_5pl']=None if minval_B_5pl_entry.get()=='None' else float(minval_B_5pl_entry.get())
                maxval['B_5pl']=None if maxval_B_5pl_entry.get()=='None' else float( maxval_B_5pl_entry.get())
            
            
                init['A_5pl']=None if init_A_5pl_entry.get()=='None' else float(init_A_5pl_entry.get())
                minval['A_5pl']=None if minval_A_5pl_entry.get()=='None' else float(minval_A_5pl_entry.get())
                maxval['A_5pl']=None if maxval_A_5pl_entry.get()=='None' else float(maxval_A_5pl_entry.get())
                
                init['A3_5pl']=None if init_A3_5pl_entry.get()=='None' else float(init_A3_5pl_entry.get())
                minval['A3_5pl']=None if minval_A3_5pl_entry.get()=='None' else float(minval_A3_5pl_entry.get())
                maxval['A3_5pl']=None if maxval_A3_5pl_entry.get()=='None' else float(maxval_A3_5pl_entry.get())
                
                
                init['B3_5pl']=None if init_B3_5pl_entry.get()=='None' else float(init_B3_5pl_entry.get())
                minval['B3_5pl']=None if minval_B3_5pl_entry.get()=='None' else float(minval_B3_5pl_entry.get())
                maxval['B3_5pl']=None if maxval_B3_5pl_entry.get()=='None' else float(maxval_B3_5pl_entry.get())
                
                init['A4_5pl']=None if init_A4_5pl_entry.get()=='None' else float(init_A4_5pl_entry.get())
                minval['A4_5pl']=None if minval_A4_5pl_entry.get()=='None' else float(minval_A4_5pl_entry.get())
                maxval['A4_5pl']=None if maxval_A4_5pl_entry.get()=='None' else float(maxval_A4_5pl_entry.get())
                
                init['B4_5pl']=None if init_B4_5pl_entry.get()=='None' else float(init_B4_5pl_entry.get())
                minval['B4_5pl']=None if minval_B4_5pl_entry.get()=='None' else float(minval_B4_5pl_entry.get())
                maxval['B4_5pl']=None if maxval_B4_5pl_entry.get()=='None' else float(maxval_B4_5pl_entry.get()) 
                
                init['A5_5pl']=None if init_A5_5pl_entry.get()=='None' else float(init_A5_5pl_entry.get())
                minval['A5_5pl']=None if minval_A5_5pl_entry.get()=='None' else float(minval_A5_5pl_entry.get())
                maxval['A5_5pl']=None if maxval_A5_5pl_entry.get()=='None' else float(maxval_A5_5pl_entry.get())
                
                init['B5_5pl']=None if init_B5_5pl_entry.get()=='None' else float(init_B5_5pl_entry.get())
                minval['B5_5pl']=None if minval_B5_5pl_entry.get()=='None' else float(minval_B5_5pl_entry.get())
                maxval['B5_5pl']=None if maxval_B5_5pl_entry.get()=='None' else float(maxval_B5_5pl_entry.get()) 
            
                
            #pull the min/max energy (x) values to fit to
            global fitmin
            global fitmax
            fitmin=float(fitmin_entry.get())
            fitmax=float(fitmax_entry.get())
            
            #validate limits
            if not validate_lims(fitmin,fitmax):
                tk.messagebox.showerror("Invalid Input","Fit limits should be floats with max greater than min")
            else:
                
                #validate entries
                validity=dict()
                for ind in minval.keys():
                    min_val=minval[ind]
                    max_val=maxval[ind]
                    valid = validate_minmaxval(min_val,max_val)
                    validity[ind]=(valid)
                if False in validity:#show where error is !!!!!
                    false_keys=list()            
                    for key, value in validity.items():
                        if value is False:
                            false_keys.append(key)
                    
                    tk.messagebox.showerror("Invalid Input",f"Parameter limits should be floats with max greater than min for parameter(s) {false_keys}")
                else:
                    validity=dict()
                    for ind in init.keys():
                        min_val=minval[ind]
                        max_val=maxval[ind]
                        init_val=init[ind]
                        validity[ind]=validate_init(init_val,min_val,max_val)
                    if False in validity:#show where error is !!!!!
                        false_keys=list()            
                        for key, value in validity.items():
                            if value is False:
                                false_keys.append(key)
                        
                        tk.messagebox.showerror("Invalid Input",f"Parameter initial values should be floats between their max and min values for parameter(s) {false_keys}")
                    else:
#%%fit outputs                        #perform the fitting function defined above to obtain the minimised parameters
                        


                        #conduct fitting process
                        
                        parvals,param_uncert_calced=fitting(header,init,vary,minval,maxval,x_data,y_data,uncert,fitmin,fitmax,spec_type)
                        

                        #add the results into the entry boxes
                        if bpl_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui

                            init_x1_entry.delete(0, tk.END)
                            init_x1_entry.insert(0,parvals["x1"])
                            init_A_entry.delete(0, tk.END)
                            init_A_entry.insert(0,parvals["A"])
                            init_B_entry.delete(0, tk.END)
                            init_B_entry.insert(0,parvals["B"])
                            init_A2_entry.delete(0, tk.END)
                            init_A2_entry.insert(0,parvals["A2"])
                            init_B2_entry.delete(0, tk.END)
                            init_B2_entry.insert(0,parvals["B2"])

                            
                        if therm_func_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui
                            init_amp_entry.delete(0, tk.END)
                            init_amp_entry.insert(0,parvals["amp"])
                            init_T_entry.delete(0, tk.END)
                            init_T_entry.insert(0,parvals["T"])
                            init_alpha_entry.delete(0, tk.END)
                            init_alpha_entry.insert(0,parvals["alpha"])
                
                        if gauss_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui
                            init_gauss_amp_entry.delete(0, tk.END)
                            init_gauss_amp_entry.insert(0,parvals["gauss_amp"])
                            init_gauss_centre_entry.delete(0, tk.END)
                            init_gauss_centre_entry.insert(0,parvals["gauss_centre"])
                            init_sigma_entry.delete(0, tk.END)
                            init_sigma_entry.insert(0,parvals["sigma"]) 
                
                
                        if power_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui
                            init_A_sing_entry.delete(0, tk.END)
                            init_A_sing_entry.insert(0,parvals["A_sing"])
                            init_B_sing_entry.delete(0, tk.END)
                            init_B_sing_entry.insert(0,parvals["B_sing"])
                            
                            
                        if kappa_pres==1:
                            
                            init_A_k_entry.delete(0, tk.END)
                            init_A_k_entry.insert(0,parvals["A_k"])
                            init_T_k_entry.delete(0, tk.END)
                            init_T_k_entry.insert(0,parvals["T_k"])
                            init_m_i_entry.delete(0, tk.END)
                            init_m_i_entry.insert(0,parvals["m_i"])
                            init_n_i_entry.delete(0, tk.END)
                            init_n_i_entry.insert(0,parvals["n_i"])
                            init_kappa_entry.delete(0, tk.END)                            
                            init_kappa_entry.insert(0,parvals["kappa"])
                        
                        
                        if bpl_and_therm_pres==1:
                            init_amp_c_entry.delete(0, tk.END)
                            init_amp_c_entry.insert(0,parvals["amp_c"])
                            init_T_c_entry.delete(0, tk.END)
                            init_T_c_entry.insert(0,parvals["T_c"])
                            init_alpha_c_entry.delete(0, tk.END)
                            init_alpha_c_entry.insert(0,parvals["alpha_c"])
                            init_x0_c_entry.delete(0, tk.END)
                            init_x0_c_entry.insert(0,parvals["x0_c"])
                            init_x1_c_entry.delete(0, tk.END)
                            init_x1_c_entry.insert(0,parvals["x1_c"])
                            init_B_c_entry.delete(0, tk.END)
                            init_B_c_entry.insert(0,parvals["B_c"])
                            init_B2_c_entry.delete(0, tk.END)
                            init_B2_c_entry.insert(0,parvals["B2_c"])
                            
                            
                            
                            
                        if double_therm_func_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui
                            init_amp_d_1_entry.delete(0, tk.END)
                            init_amp_d_1_entry.insert(0,parvals["amp_d_1"])
                            init_T_d_1_entry.delete(0, tk.END)
                            init_T_d_1_entry.insert(0,parvals["T_d_1"])
                            init_alpha_d_1_entry.delete(0, tk.END)
                            init_alpha_d_1_entry.insert(0,parvals["alpha_d_1"])
                            init_amp_d_2_entry.delete(0, tk.END)
                            init_amp_d_2_entry.insert(0,parvals["amp_d_2"])
                            init_T_d_2_entry.delete(0, tk.END)
                            init_T_d_2_entry.insert(0,parvals["T_d_2"])
                            init_alpha_d_2_entry.delete(0, tk.END)
                            init_alpha_d_2_entry.insert(0,parvals["alpha_d_2"])
                
                        if tpl_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui

                            init_x1_tpl_entry.delete(0, tk.END)
                            init_x1_tpl_entry.insert(0,parvals["x1_tpl"])
                            init_x2_tpl_entry.delete(0, tk.END)
                            init_x2_tpl_entry.insert(0,parvals["x2_tpl"])
                            init_A_tpl_entry.delete(0, tk.END)
                            init_A_tpl_entry.insert(0,parvals["A_tpl"])
                            init_B_tpl_entry.delete(0, tk.END)
                            init_B_tpl_entry.insert(0,parvals["B_tpl"])
                            init_A2_tpl_entry.delete(0, tk.END)
                            init_A2_tpl_entry.insert(0,parvals["A2_tpl"])
                            init_B2_tpl_entry.delete(0, tk.END)
                            init_B2_tpl_entry.insert(0,parvals["B2_tpl"])
                            init_A3_tpl_entry.delete(0, tk.END)
                            init_A3_tpl_entry.insert(0,parvals["A3_tpl"])
                            init_B3_tpl_entry.delete(0, tk.END)
                            init_B3_tpl_entry.insert(0,parvals["B3_tpl"])


                        if qpl_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui

                            init_x1_qpl_entry.delete(0, tk.END)
                            init_x1_qpl_entry.insert(0,parvals["x1_qpl"])
                            init_x2_qpl_entry.delete(0, tk.END)
                            init_x2_qpl_entry.insert(0,parvals["x2_qpl"])
                            init_x3_qpl_entry.delete(0, tk.END)
                            init_x3_qpl_entry.insert(0,parvals["x3_qpl"])
                            init_A_qpl_entry.delete(0, tk.END)
                            init_A_qpl_entry.insert(0,parvals["A_qpl"])
                            init_B_qpl_entry.delete(0, tk.END)
                            init_B_qpl_entry.insert(0,parvals["B_qpl"])
                            init_A2_qpl_entry.delete(0, tk.END)
                            init_A2_qpl_entry.insert(0,parvals["A2_qpl"])
                            init_B2_qpl_entry.delete(0, tk.END)
                            init_B2_qpl_entry.insert(0,parvals["B2_qpl"])
                            init_A3_qpl_entry.delete(0, tk.END)
                            init_A3_qpl_entry.insert(0,parvals["A3_qpl"])
                            init_B3_qpl_entry.delete(0, tk.END)
                            init_B3_qpl_entry.insert(0,parvals["B3_qpl"])
                            init_A4_qpl_entry.delete(0, tk.END)
                            init_A4_qpl_entry.insert(0,parvals["A4_qpl"])
                            init_B4_qpl_entry.delete(0, tk.END)
                            init_B4_qpl_entry.insert(0,parvals["B4_qpl"])
                            
                        if quint_pl_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui

                            init_x1_5pl_entry.delete(0, tk.END)
                            init_x1_5pl_entry.insert(0,parvals["x1_5pl"])
                            init_x2_5pl_entry.delete(0, tk.END)
                            init_x2_5pl_entry.insert(0,parvals["x2_5pl"])
                            init_x3_5pl_entry.delete(0, tk.END)
                            init_x3_5pl_entry.insert(0,parvals["x3_5pl"])
                            init_x4_5pl_entry.delete(0, tk.END)
                            init_x4_5pl_entry.insert(0,parvals["x4_5pl"])
                            init_A_5pl_entry.delete(0, tk.END)
                            init_A_5pl_entry.insert(0,parvals["A_5pl"])
                            init_B_5pl_entry.delete(0, tk.END)
                            init_B_5pl_entry.insert(0,parvals["B_5pl"])
                            init_A2_5pl_entry.delete(0, tk.END)
                            init_A2_5pl_entry.insert(0,parvals["A2_5pl"])
                            init_B2_5pl_entry.delete(0, tk.END)
                            init_B2_5pl_entry.insert(0,parvals["B2_5pl"])
                            init_A3_5pl_entry.delete(0, tk.END)
                            init_A3_5pl_entry.insert(0,parvals["A3_5pl"])
                            init_B3_5pl_entry.delete(0, tk.END)
                            init_B3_5pl_entry.insert(0,parvals["B3_5pl"])
                            init_A4_5pl_entry.delete(0, tk.END)
                            init_A4_5pl_entry.insert(0,parvals["A4_5pl"])
                            init_B4_5pl_entry.delete(0, tk.END)
                            init_B4_5pl_entry.insert(0,parvals["B4_5pl"])
                            init_A5_5pl_entry.delete(0, tk.END)
                            init_A5_5pl_entry.insert(0,parvals["A5_5pl"])
                            init_B5_5pl_entry.delete(0, tk.END)
                            init_B5_5pl_entry.insert(0,parvals["B5_5pl"])
            
                        global parvals_new
                        parvals_new=parvals
               
                        print("uncerts")
                        print(param_uncert_calced)
                        
                        
                        #the percentage uncerts 
                        print('percent uncerts')
                        for key in list(param_uncert_calced.keys()):
                           frac=param_uncert_calced[key]/parvals_new[key]
                           print(str(key)+":"+str(frac*100))               
        
        
        
        
            
        except ValueError as e:
               tk.messagebox.showerror("Invalid Input","Inputs should be floating point intergers")
               print(e)
       

#%%preview buttns handling   

    def preview_btn_hndl():#function to handle preview button
        global init
        global vary
        global minval
        global maxval
        global fit_window
        global preview_window
        global resid_window
        if preview_window is not None:# and preview_window.winfo_exists():
            #close any open figues
            preview_window.destroy()
            preview_window=None
            
        if fit_window is not None:# and fit_window.winfo_exists():
            #close any open figues
            fit_window.destroy()
            fit_window=None
        if resid_window is not None:# and resid_window.winfo_exists():
            #close any open figues
            resid_window.destroy()
            resid_window=None
        
        
        global header
        header=f"bpl_pres={bpl_pres}; therm_func_pres={therm_func_pres}; gauss_pres={gauss_pres}; power_pres={power_pres}; kappa_pres={kappa_pres}; bpl_and_therm_pres={bpl_and_therm_pres}; double_therm_func_pres={double_therm_func_pres}; tpl_pres={tpl_pres}; qpl_pres={qpl_pres}; quint_pl_pres={quint_pl_pres};"#defines header according to what functions are currently present in the gui
        try:#try excpet statement is to validate inputs as integers
            if therm_func_pres==1:#if thermal function present, save parameter options from the gui for that function
                
                global frame_therm
                
                init['T']=None if init_T_entry.get()=='None' else float(init_T_entry.get())
                minval['T']=None if minval_T_entry.get()=='None' else float(minval_T_entry.get())
                maxval['T']=None if maxval_T_entry.get()=='None' else float(maxval_T_entry.get())
                
                init['amp']=None if init_amp_entry.get()=='None' else float(init_amp_entry.get())
                minval['amp']=None if minval_amp_entry.get()=='None' else float(minval_amp_entry.get())
                maxval['amp']=None if maxval_amp_entry.get()=='None' else float(maxval_amp_entry.get())
                
                init['alpha']=None if init_alpha_entry.get()=='None' else float(init_alpha_entry.get())
                minval['alpha']=None if minval_alpha_entry.get()=='None' else float(minval_alpha_entry.get())
                maxval['alpha']=None if maxval_alpha_entry.get()=='None' else float(maxval_alpha_entry.get())
                
            if bpl_pres==1:#if bpl function present, save parameter options from the gui for that function
                
                
                
                init['x1']=None if init_x1_entry.get()=='None' else float(init_x1_entry.get())
                minval['x1']=None if minval_x1_entry.get()=='None' else float(minval_x1_entry.get())
                maxval['x1']=None if maxval_x1_entry.get()=='None' else float(maxval_x1_entry.get())
                
                
                
                
                init['A2']=None if init_A2_entry.get()=='None' else float(init_A2_entry.get())
                minval['A2']=None if minval_A2_entry.get()=='None' else float(minval_A2_entry.get())
                maxval['A2']=None if maxval_A2_entry.get()=='None' else float(maxval_A2_entry.get())
                
                
                init['B2']=None if init_B2_entry.get()=='None' else float(init_B2_entry.get())
                minval['B2']=None if minval_B2_entry.get()=='None' else float(minval_B2_entry.get())
                maxval['B2']=None if maxval_B2_entry.get()=='None' else float(maxval_B2_entry.get())
                
                
                init['B']=None if init_B_entry.get()=='None' else float(init_B_entry.get())
                minval['B']=None if minval_B_entry.get()=='None' else float(minval_B_entry.get())
                maxval['B']=None if maxval_B_entry.get()=='None' else float( maxval_B_entry.get())
    
    
                init['A']=None if init_A_entry.get()=='None' else float(init_A_entry.get())
                minval['A']=None if minval_A_entry.get()=='None' else float(minval_A_entry.get())
                maxval['A']=None if maxval_A_entry.get()=='None' else float(maxval_A_entry.get())
                
            if gauss_pres==1:#if gaussian function present, save parameter options from the gui for that function
                init['gauss_centre']=None if init_gauss_centre_entry.get()=='None' else float(init_gauss_centre_entry.get())
                minval['gauss_centre']=None if minval_gauss_centre_entry.get()=='None' else float(minval_gauss_centre_entry.get())
                maxval['gauss_centre']=None if maxval_gauss_centre_entry.get()=='None' else float(maxval_gauss_centre_entry.get())
                
                
                init['gauss_amp']=None if init_gauss_amp_entry.get()=='None' else float(init_gauss_amp_entry.get())
                minval['gauss_amp']=None if minval_gauss_amp_entry.get()=='None' else float(minval_gauss_amp_entry.get())
                maxval['gauss_amp']=None if maxval_gauss_amp_entry.get()=='None' else float( maxval_gauss_amp_entry.get())
    
    
                init['sigma']=None if init_sigma_entry.get()=='None' else float(init_sigma_entry.get())
                minval['sigma']=None if minval_sigma_entry.get()=='None' else float(minval_sigma_entry.get())
                maxval['sigma']=None if maxval_sigma_entry.get()=='None' else float(maxval_sigma_entry.get())
               
               
               
               
               
            if power_pres==1:#if single power law function present, save parameter options from the gui for that function
               init['B_sing']=None if init_B_sing_entry.get()=='None' else float(init_B_sing_entry.get())
               minval['B_sing']=None if minval_B_sing_entry.get()=='None' else float(minval_B_sing_entry.get())
               maxval['B_sing']=None if maxval_B_sing_entry.get()=='None' else float( maxval_B_sing_entry.get())
    
    
               init['A_sing']=None if init_A_sing_entry.get()=='None' else float(init_A_sing_entry.get())
               minval['A_sing']=None if minval_A_sing_entry.get()=='None' else float(minval_A_sing_entry.get())
               maxval['A_sing']=None if maxval_A_sing_entry.get()=='None' else float(maxval_A_sing_entry.get())


            if kappa_pres==1:#if kappa function present, save parameter options from the gui for that function

    
               init['A_k']=None if init_A_k_entry.get()=='None' else float(init_A_k_entry.get())
               minval['A_k']=None if minval_A_k_entry.get()=='None' else float(minval_A_k_entry.get())
               maxval['A_k']=None if maxval_A_k_entry.get()=='None' else float(maxval_A_k_entry.get())
               
               init['T_k']=None if init_T_k_entry.get()=='None' else float(init_T_k_entry.get())
               minval['T_k']=None if minval_T_k_entry.get()=='None' else float(minval_T_k_entry.get())
               maxval['T_k']=None if maxval_T_k_entry.get()=='None' else float( maxval_T_k_entry.get())
               
               init['m_i']=None if init_m_i_entry.get()=='None' else float(init_m_i_entry.get())
               minval['m_i']=None if minval_m_i_entry.get()=='None' else float(minval_m_i_entry.get())
               maxval['m_i']=None if maxval_m_i_entry.get()=='None' else float( maxval_m_i_entry.get())
               
               init['n_i']=None if init_n_i_entry.get()=='None' else float(init_n_i_entry.get())
               minval['n_i']=None if minval_n_i_entry.get()=='None' else float(minval_n_i_entry.get())
               maxval['n_i']=None if maxval_n_i_entry.get()=='None' else float( maxval_n_i_entry.get())    
               
               init['kappa']=None if init_kappa_entry.get()=='None' else float(init_kappa_entry.get())
               minval['kappa']=None if minval_kappa_entry.get()=='None' else float(minval_kappa_entry.get())
               maxval['kappa']=None if maxval_kappa_entry.get()=='None' else float( maxval_kappa_entry.get())
               
            if bpl_and_therm_pres==1:
                init['T_c']=None if init_T_c_entry.get()=='None' else float(init_T_c_entry.get())
                minval['T_c']=None if minval_T_c_entry.get()=='None' else float(minval_T_c_entry.get())
                maxval['T_c']=None if maxval_T_c_entry.get()=='None' else float(maxval_T_c_entry.get())
                
                init['amp_c']=None if init_amp_c_entry.get()=='None' else float(init_amp_c_entry.get())
                minval['amp_c']=None if minval_amp_c_entry.get()=='None' else float(minval_amp_c_entry.get())
                maxval['amp_c']=None if maxval_amp_c_entry.get()=='None' else float(maxval_amp_c_entry.get())
                
                init['alpha_c']=None if init_alpha_c_entry.get()=='None' else float(init_alpha_c_entry.get())
                minval['alpha_c']=None if minval_alpha_c_entry.get()=='None' else float(minval_alpha_c_entry.get())
                maxval['alpha_c']=None if maxval_alpha_c_entry.get()=='None' else float(maxval_alpha_c_entry.get())
                
                init['x1_c']=None if init_x1_c_entry.get()=='None' else float(init_x1_c_entry.get())
                minval['x1_c']=None if minval_x1_c_entry.get()=='None' else float(minval_x1_c_entry.get())
                maxval['x1_c']=None if maxval_x1_c_entry.get()=='None' else float(maxval_x1_c_entry.get())
                
                init['x0_c']=None if init_x0_c_entry.get()=='None' else float(init_x0_c_entry.get())
                minval['x0_c']=None if minval_x0_c_entry.get()=='None' else float(minval_x0_c_entry.get())
                maxval['x0_c']=None if maxval_x0_c_entry.get()=='None' else float(maxval_x0_c_entry.get())
                
                init['B2_c']=None if init_B2_c_entry.get()=='None' else float(init_B2_c_entry.get())
                minval['B2_c']=None if minval_B2_c_entry.get()=='None' else float(minval_B2_c_entry.get())
                maxval['B2_c']=None if maxval_B2_c_entry.get()=='None' else float(maxval_B2_c_entry.get())
                
                
                init['B_c']=None if init_B_c_entry.get()=='None' else float(init_B_c_entry.get())
                minval['B_c']=None if minval_B_c_entry.get()=='None' else float(minval_B_c_entry.get())
                maxval['B_c']=None if maxval_B_c_entry.get()=='None' else float( maxval_B_c_entry.get())
                
                
                
            if double_therm_func_pres==1:#if double thermal function present, save parameter options from the gui for that function
                
                global frame_double_therm
                
                init['T_d_1']=None if init_T_d_1_entry.get()=='None' else float(init_T_d_1_entry.get())
                minval['T_d_1']=None if minval_T_d_1_entry.get()=='None' else float(minval_T_d_1_entry.get())
                maxval['T_d_1']=None if maxval_T_d_1_entry.get()=='None' else float(maxval_T_d_1_entry.get())
                
                init['amp_d_1']=None if init_amp_d_1_entry.get()=='None' else float(init_amp_d_1_entry.get())
                minval['amp_d_1']=None if minval_amp_d_1_entry.get()=='None' else float(minval_amp_d_1_entry.get())
                maxval['amp_d_1']=None if maxval_amp_d_1_entry.get()=='None' else float(maxval_amp_d_1_entry.get())
                
                init['alpha_d_1']=None if init_alpha_d_1_entry.get()=='None' else float(init_alpha_d_1_entry.get())
                minval['alpha_d_1']=None if minval_alpha_d_1_entry.get()=='None' else float(minval_alpha_d_1_entry.get())
                maxval['alpha_d_1']=None if maxval_alpha_d_1_entry.get()=='None' else float(maxval_alpha_d_1_entry.get())

                init['T_d_2']=None if init_T_d_2_entry.get()=='None' else float(init_T_d_2_entry.get())
                minval['T_d_2']=None if minval_T_d_2_entry.get()=='None' else float(minval_T_d_2_entry.get())
                maxval['T_d_2']=None if maxval_T_d_2_entry.get()=='None' else float(maxval_T_d_2_entry.get())
                
                init['amp_d_2']=None if init_amp_d_2_entry.get()=='None' else float(init_amp_d_2_entry.get())
                minval['amp_d_2']=None if minval_amp_d_2_entry.get()=='None' else float(minval_amp_d_2_entry.get())
                maxval['amp_d_2']=None if maxval_amp_d_2_entry.get()=='None' else float(maxval_amp_d_2_entry.get())
                
                init['alpha_d_2']=None if init_alpha_d_2_entry.get()=='None' else float(init_alpha_d_2_entry.get())
                minval['alpha_d_2']=None if minval_alpha_d_2_entry.get()=='None' else float(minval_alpha_d_2_entry.get())
                maxval['alpha_d_2']=None if maxval_alpha_d_2_entry.get()=='None' else float(maxval_alpha_d_2_entry.get())
            
            
            if tpl_pres==1:#if tpl function present, save parameter options from the gui for that function
                
                
                
                init['x1_tpl']=None if init_x1_tpl_entry.get()=='None' else float(init_x1_tpl_entry.get())
                minval['x1_tpl']=None if minval_x1_tpl_entry.get()=='None' else float(minval_x1_tpl_entry.get())
                maxval['x1_tpl']=None if maxval_x1_tpl_entry.get()=='None' else float(maxval_x1_tpl_entry.get())
                
                                
                init['x2_tpl']=None if init_x2_tpl_entry.get()=='None' else float(init_x2_tpl_entry.get())
                minval['x2_tpl']=None if minval_x2_tpl_entry.get()=='None' else float(minval_x2_tpl_entry.get())
                maxval['x2_tpl']=None if maxval_x2_tpl_entry.get()=='None' else float(maxval_x2_tpl_entry.get())
                
                
                init['A2_tpl']=None if init_A2_tpl_entry.get()=='None' else float(init_A2_tpl_entry.get())
                minval['A2_tpl']=None if minval_A2_tpl_entry.get()=='None' else float(minval_A2_tpl_entry.get())
                maxval['A2_tpl']=None if maxval_A2_tpl_entry.get()=='None' else float(maxval_A2_tpl_entry.get())
                
                
                init['B2_tpl']=None if init_B2_tpl_entry.get()=='None' else float(init_B2_tpl_entry.get())
                minval['B2_tpl']=None if minval_B2_tpl_entry.get()=='None' else float(minval_B2_tpl_entry.get())
                maxval['B2_tpl']=None if maxval_B2_tpl_entry.get()=='None' else float(maxval_B2_tpl_entry.get())
                
                
                init['B_tpl']=None if init_B_tpl_entry.get()=='None' else float(init_B_tpl_entry.get())
                minval['B_tpl']=None if minval_B_tpl_entry.get()=='None' else float(minval_B_tpl_entry.get())
                maxval['B_tpl']=None if maxval_B_tpl_entry.get()=='None' else float( maxval_B_tpl_entry.get())
    
    
                init['A_tpl']=None if init_A_tpl_entry.get()=='None' else float(init_A_tpl_entry.get())
                minval['A_tpl']=None if minval_A_tpl_entry.get()=='None' else float(minval_A_tpl_entry.get())
                maxval['A_tpl']=None if maxval_A_tpl_entry.get()=='None' else float(maxval_A_tpl_entry.get())
                
                init['A3_tpl']=None if init_A3_tpl_entry.get()=='None' else float(init_A3_tpl_entry.get())
                minval['A3_tpl']=None if minval_A3_tpl_entry.get()=='None' else float(minval_A3_tpl_entry.get())
                maxval['A3_tpl']=None if maxval_A3_tpl_entry.get()=='None' else float(maxval_A3_tpl_entry.get())
                
                
                init['B3_tpl']=None if init_B3_tpl_entry.get()=='None' else float(init_B3_tpl_entry.get())
                minval['B3_tpl']=None if minval_B3_tpl_entry.get()=='None' else float(minval_B3_tpl_entry.get())
                maxval['B3_tpl']=None if maxval_B3_tpl_entry.get()=='None' else float(maxval_B3_tpl_entry.get())
                
            if qpl_pres==1:#if qpl function present, save parameter options from the gui for that function
                init['x1_qpl']=None if init_x1_qpl_entry.get()=='None' else float(init_x1_qpl_entry.get())
                minval['x1_qpl']=None if minval_x1_qpl_entry.get()=='None' else float(minval_x1_qpl_entry.get())
                maxval['x1_qpl']=None if maxval_x1_qpl_entry.get()=='None' else float(maxval_x1_qpl_entry.get())
                
                                
                init['x2_qpl']=None if init_x2_qpl_entry.get()=='None' else float(init_x2_qpl_entry.get())
                minval['x2_qpl']=None if minval_x2_qpl_entry.get()=='None' else float(minval_x2_qpl_entry.get())
                maxval['x2_qpl']=None if maxval_x2_qpl_entry.get()=='None' else float(maxval_x2_qpl_entry.get())
                
                init['x3_qpl']=None if init_x3_qpl_entry.get()=='None' else float(init_x3_qpl_entry.get())
                minval['x3_qpl']=None if minval_x3_qpl_entry.get()=='None' else float(minval_x3_qpl_entry.get())
                maxval['x3_qpl']=None if maxval_x3_qpl_entry.get()=='None' else float(maxval_x3_qpl_entry.get())
                
                init['A2_qpl']=None if init_A2_qpl_entry.get()=='None' else float(init_A2_qpl_entry.get())
                minval['A2_qpl']=None if minval_A2_qpl_entry.get()=='None' else float(minval_A2_qpl_entry.get())
                maxval['A2_qpl']=None if maxval_A2_qpl_entry.get()=='None' else float(maxval_A2_qpl_entry.get())
                
                
                init['B2_qpl']=None if init_B2_qpl_entry.get()=='None' else float(init_B2_qpl_entry.get())
                minval['B2_qpl']=None if minval_B2_qpl_entry.get()=='None' else float(minval_B2_qpl_entry.get())
                maxval['B2_qpl']=None if maxval_B2_qpl_entry.get()=='None' else float(maxval_B2_qpl_entry.get())
                
                
                init['B_qpl']=None if init_B_qpl_entry.get()=='None' else float(init_B_qpl_entry.get())
                minval['B_qpl']=None if minval_B_qpl_entry.get()=='None' else float(minval_B_qpl_entry.get())
                maxval['B_qpl']=None if maxval_B_qpl_entry.get()=='None' else float( maxval_B_qpl_entry.get())
            
            
                init['A_qpl']=None if init_A_qpl_entry.get()=='None' else float(init_A_qpl_entry.get())
                minval['A_qpl']=None if minval_A_qpl_entry.get()=='None' else float(minval_A_qpl_entry.get())
                maxval['A_qpl']=None if maxval_A_qpl_entry.get()=='None' else float(maxval_A_qpl_entry.get())
                
                init['A3_qpl']=None if init_A3_qpl_entry.get()=='None' else float(init_A3_qpl_entry.get())
                minval['A3_qpl']=None if minval_A3_qpl_entry.get()=='None' else float(minval_A3_qpl_entry.get())
                maxval['A3_qpl']=None if maxval_A3_qpl_entry.get()=='None' else float(maxval_A3_qpl_entry.get())
                
                
                init['B3_qpl']=None if init_B3_qpl_entry.get()=='None' else float(init_B3_qpl_entry.get())
                minval['B3_qpl']=None if minval_B3_qpl_entry.get()=='None' else float(minval_B3_qpl_entry.get())
                maxval['B3_qpl']=None if maxval_B3_qpl_entry.get()=='None' else float(maxval_B3_qpl_entry.get())
                
                init['A4_qpl']=None if init_A4_qpl_entry.get()=='None' else float(init_A4_qpl_entry.get())
                minval['A4_qpl']=None if minval_A4_qpl_entry.get()=='None' else float(minval_A4_qpl_entry.get())
                maxval['A4_qpl']=None if maxval_A4_qpl_entry.get()=='None' else float(maxval_A4_qpl_entry.get())
                
                init['B4_qpl']=None if init_B4_qpl_entry.get()=='None' else float(init_B4_qpl_entry.get())
                minval['B4_qpl']=None if minval_B4_qpl_entry.get()=='None' else float(minval_B4_qpl_entry.get())
                maxval['B4_qpl']=None if maxval_B4_qpl_entry.get()=='None' else float(maxval_B4_qpl_entry.get())
                
            if quint_pl_pres==1:#if 5pl function present, save parameter options from the gui for that function
                init['x1_5pl']=None if init_x1_5pl_entry.get()=='None' else float(init_x1_5pl_entry.get())
                minval['x1_5pl']=None if minval_x1_5pl_entry.get()=='None' else float(minval_x1_5pl_entry.get())
                maxval['x1_5pl']=None if maxval_x1_5pl_entry.get()=='None' else float(maxval_x1_5pl_entry.get())
                
                                
                init['x2_5pl']=None if init_x2_5pl_entry.get()=='None' else float(init_x2_5pl_entry.get())
                minval['x2_5pl']=None if minval_x2_5pl_entry.get()=='None' else float(minval_x2_5pl_entry.get())
                maxval['x2_5pl']=None if maxval_x2_5pl_entry.get()=='None' else float(maxval_x2_5pl_entry.get())
                
                init['x3_5pl']=None if init_x3_5pl_entry.get()=='None' else float(init_x3_5pl_entry.get())
                minval['x3_5pl']=None if minval_x3_5pl_entry.get()=='None' else float(minval_x3_5pl_entry.get())
                maxval['x3_5pl']=None if maxval_x3_5pl_entry.get()=='None' else float(maxval_x3_5pl_entry.get())
                
                init['x4_5pl']=None if init_x4_5pl_entry.get()=='None' else float(init_x4_5pl_entry.get())
                minval['x4_5pl']=None if minval_x4_5pl_entry.get()=='None' else float(minval_x4_5pl_entry.get())
                maxval['x4_5pl']=None if maxval_x4_5pl_entry.get()=='None' else float(maxval_x4_5pl_entry.get())
                
                init['A2_5pl']=None if init_A2_5pl_entry.get()=='None' else float(init_A2_5pl_entry.get())
                minval['A2_5pl']=None if minval_A2_5pl_entry.get()=='None' else float(minval_A2_5pl_entry.get())
                maxval['A2_5pl']=None if maxval_A2_5pl_entry.get()=='None' else float(maxval_A2_5pl_entry.get())
                
                
                init['B2_5pl']=None if init_B2_5pl_entry.get()=='None' else float(init_B2_5pl_entry.get())
                minval['B2_5pl']=None if minval_B2_5pl_entry.get()=='None' else float(minval_B2_5pl_entry.get())
                maxval['B2_5pl']=None if maxval_B2_5pl_entry.get()=='None' else float(maxval_B2_5pl_entry.get())
                
                
                init['B_5pl']=None if init_B_5pl_entry.get()=='None' else float(init_B_5pl_entry.get())
                minval['B_5pl']=None if minval_B_5pl_entry.get()=='None' else float(minval_B_5pl_entry.get())
                maxval['B_5pl']=None if maxval_B_5pl_entry.get()=='None' else float( maxval_B_5pl_entry.get())
            
            
                init['A_5pl']=None if init_A_5pl_entry.get()=='None' else float(init_A_5pl_entry.get())
                minval['A_5pl']=None if minval_A_5pl_entry.get()=='None' else float(minval_A_5pl_entry.get())
                maxval['A_5pl']=None if maxval_A_5pl_entry.get()=='None' else float(maxval_A_5pl_entry.get())
                
                init['A3_5pl']=None if init_A3_5pl_entry.get()=='None' else float(init_A3_5pl_entry.get())
                minval['A3_5pl']=None if minval_A3_5pl_entry.get()=='None' else float(minval_A3_5pl_entry.get())
                maxval['A3_5pl']=None if maxval_A3_5pl_entry.get()=='None' else float(maxval_A3_5pl_entry.get())
                
                
                init['B3_5pl']=None if init_B3_5pl_entry.get()=='None' else float(init_B3_5pl_entry.get())
                minval['B3_5pl']=None if minval_B3_5pl_entry.get()=='None' else float(minval_B3_5pl_entry.get())
                maxval['B3_5pl']=None if maxval_B3_5pl_entry.get()=='None' else float(maxval_B3_5pl_entry.get())
                
                init['A4_5pl']=None if init_A4_5pl_entry.get()=='None' else float(init_A4_5pl_entry.get())
                minval['A4_5pl']=None if minval_A4_5pl_entry.get()=='None' else float(minval_A4_5pl_entry.get())
                maxval['A4_5pl']=None if maxval_A4_5pl_entry.get()=='None' else float(maxval_A4_5pl_entry.get())
                
                init['B4_5pl']=None if init_B4_5pl_entry.get()=='None' else float(init_B4_5pl_entry.get())
                minval['B4_5pl']=None if minval_B4_5pl_entry.get()=='None' else float(minval_B4_5pl_entry.get())
                maxval['B4_5pl']=None if maxval_B4_5pl_entry.get()=='None' else float(maxval_B4_5pl_entry.get())               
                
                init['A5_5pl']=None if init_A5_5pl_entry.get()=='None' else float(init_A5_5pl_entry.get())
                minval['A5_5pl']=None if minval_A5_5pl_entry.get()=='None' else float(minval_A5_5pl_entry.get())
                maxval['A5_5pl']=None if maxval_A5_5pl_entry.get()=='None' else float(maxval_A5_5pl_entry.get())
                
                init['B5_5pl']=None if init_B5_5pl_entry.get()=='None' else float(init_B5_5pl_entry.get())
                minval['B5_5pl']=None if minval_B5_5pl_entry.get()=='None' else float(minval_B5_5pl_entry.get())
                maxval['B5_5pl']=None if maxval_B5_5pl_entry.get()=='None' else float(maxval_B5_5pl_entry.get())  
                
                
            #pull the min/max energy (x) values to fit to
            fitmin=float(fitmin_entry.get())
            fitmax=float(fitmax_entry.get())
            #validate limits
            if not validate_lims(fitmin,fitmax):
                tk.messagebox.showerror("Invalid Input","Fit limits should be floats with max greater than min")
            else:
                
                #validate entries
                validity=dict()
                for ind in minval.keys():
                    min_val=minval[ind]
                    max_val=maxval[ind]
                    valid = validate_minmaxval(min_val,max_val)
                    validity[ind]=(valid)
                if False in validity:#show where error is !!!!!
                    false_keys=list()            
                    for key, value in validity.items():
                        if value is False:
                            false_keys.append(key)
                    
                    tk.messagebox.showerror("Invalid Input",f"Parameter limits should be floats with max greater than min for parameter(s) {false_keys}")
                else:
                    validity=dict()
                    for ind in init.keys():
                        min_val=minval[ind]
                        max_val=maxval[ind]
                        init_val=init[ind]
                        validity[ind]=validate_init(init_val,min_val,max_val)
                    if False in validity:#show where error is !!!!!
                        false_keys=list()            
                        for key, value in validity.items():
                            if value is False:
                                false_keys.append(key)
                        
                        tk.messagebox.showerror("Invalid Input",f"Parameter initial values should be floats between their max and min values for parameter(s) {false_keys}")                
        
        
        
        
        
        
        except ValueError:
               tk.messagebox.showerror("Invalid Input","inputs should be floating point intergers preview")
        param_preview(x_data,y_data,init,header)#calls previously defined save function
    

    
    
    #option to save the spectrum
    def spec_save_hndl():
        #organise into dataframe
        spec_dict={'energies':list(x_data) ,'fluxes': list(y_data),'errors':list(uncert),'date':[str(date) for i in list(x_data)],'inst':[inst for i in list(x_data)],'spec_type':[spec_type for i in list(x_data)]}
    
        spec_frame=pd.DataFrame(spec_dict)
        files = [('Text Document','*.txt')]
        file_obj=tk.filedialog.asksaveasfile(filetypes = files, defaultextension=".txt")
        spec_frame.to_csv(file_obj)

    
    
    def close_btn_hndl():
        global fit_window
        window_buttons.destroy()
        if fit_window != None:
            fit_window.destroy()
            fit_window=None
    



#%%load savehandling
    def save_btn_hndl():#function to handle save button
        param_save(date,inst,spec_type, bpl_pres, therm_func_pres, gauss_pres, power_pres, kappa_pres,bpl_and_therm_pres, double_therm_func_pres, tpl_pres, qpl_pres,quint_pl_pres,redchi)#calls previously defined save function
    

    def load_btn_hndl():#function to handle load button
        global header
        header,parvals_ld=param_load(date,inst,spec_type)
        
        global init
        global vary
        global minval
        global maxval
        
        if header[9]=='1':# ie if the bpl is present in the save, add the function with the saved param values
            
            add_bpl()
    
            init_x1_entry.delete(0, tk.END)
            init_x1_entry.insert(0,parvals_ld["x1"])
            init_A_entry.delete(0, tk.END)
            init_A_entry.insert(0,parvals_ld["A"])
            init_B_entry.delete(0, tk.END)
            init_B_entry.insert(0,parvals_ld["B"])
            init_A2_entry.delete(0, tk.END)
            init_A2_entry.insert(0,parvals_ld["A2"])
            init_B2_entry.delete(0, tk.END)
            init_B2_entry.insert(0,parvals_ld["B2"])


            
        if header[28]=='1':#ie if the therm func is present in the save, add the function with the saved param values
           
            add_therm()
            
            init_amp_entry.delete(0, tk.END)
            init_amp_entry.insert(0,parvals_ld["amp"])
            init_T_entry.delete(0, tk.END)
            init_T_entry.insert(0,parvals_ld["T"])
            init_alpha_entry.delete(0, tk.END)
            init_alpha_entry.insert(0,parvals_ld["alpha"])
        
        if header[42]=='1': #ie if gaussian is present in the save, add the function with the saved param values
            add_gauss()
            
            init_gauss_amp_entry.delete(0, tk.END)
            init_gauss_amp_entry.insert(0,parvals_ld["gauss_amp"])
            init_gauss_centre_entry.delete(0, tk.END)
            init_gauss_centre_entry.insert(0,parvals_ld["gauss_centre"])
            init_sigma_entry.delete(0, tk.END)
            init_sigma_entry.insert(0,parvals_ld["sigma"]) 
        
        if header[56]=='1': #ie if the single power law is present in the save, add the function with the saved param values
            add_power()
            init_A_sing_entry.delete(0, tk.END)
            init_A_sing_entry.insert(0,parvals_ld["A_sing"])
            init_B_sing_entry.delete(0, tk.END)
            init_B_sing_entry.insert(0,parvals_ld["B_sing"])
            
        if header[70]=='1':#ie if the kappa function is present
            add_kappa()

            init_A_k_entry.delete(0, tk.END)
            init_A_k_entry.insert(0,parvals_ld["A_k"])
            init_T_k_entry.delete(0, tk.END)
            init_T_k_entry.insert(0,parvals_ld["T_k"])
            init_m_i_entry.delete(0, tk.END)
            init_m_i_entry.insert(0,parvals_ld["m_i"])
            init_n_i_entry.delete(0, tk.END)
            init_n_i_entry.insert(0,parvals_ld["n_i"])
            init_kappa_entry.delete(0, tk.END)
            init_kappa_entry.insert(0,parvals_ld["kappa"])
                                  
        if header[92]=='1':
            add_bpl_and_therm()
            init_amp_c_entry.delete(0, tk.END)
            init_amp_c_entry.insert(0,parvals_ld["amp_c"])
            init_T_c_entry.delete(0, tk.END)
            init_T_c_entry.insert(0,parvals_ld["T_c"])
            init_alpha_c_entry.delete(0, tk.END)
            init_alpha_c_entry.insert(0,parvals_ld["alpha_c"])
            init_x0_c_entry.delete(0, tk.END)
            init_x0_c_entry.insert(0,parvals_ld["x0_c"])
            init_x1_c_entry.delete(0, tk.END)
            init_x1_c_entry.insert(0,parvals_ld["x1_c"])
            init_B_c_entry.delete(0, tk.END)
            init_B_c_entry.insert(0,parvals_ld["B_c"])
            init_B2_c_entry.delete(0, tk.END)
            init_B2_c_entry.insert(0,parvals_ld["B2_c"])
            
        if header[118]=='1':
            add_double_therm()
            
            init_amp_d_1_entry.delete(0, tk.END)
            init_amp_d_1_entry.insert(0,parvals_ld["amp_d_1"])
            init_T_d_1_entry.delete(0, tk.END)
            init_T_d_1_entry.insert(0,parvals_ld["T_d_1"])
            init_alpha_d_1_entry.delete(0, tk.END)
            init_alpha_d_1_entry.insert(0,parvals_ld["alpha_d_1"])
            init_amp_d_2_entry.delete(0, tk.END)
            init_amp_d_2_entry.insert(0,parvals_ld["amp_d_2"])
            init_T_d_2_entry.delete(0, tk.END)
            init_T_d_2_entry.insert(0,parvals_ld["T_d_2"])
            init_alpha_d_2_entry.delete(0, tk.END)
            init_alpha_d_2_entry.insert(0,parvals_ld["alpha_d_2"])
                                  
            
            
            
        if header[130]=='1':# ie if the tpl is present in the save, add the function with the saved param values
            
            add_tpl()
    
            init_x1_tpl_entry.delete(0, tk.END)
            init_x1_tpl_entry.insert(0,parvals_ld["x1_tpl"])
            init_x2_tpl_entry.delete(0, tk.END)
            init_x2_tpl_entry.insert(0,parvals_ld["x2_tpl"])
            init_A_tpl_entry.delete(0, tk.END)
            init_A_tpl_entry.insert(0,parvals_ld["A_tpl"])
            init_B_tpl_entry.delete(0, tk.END)
            init_B_tpl_entry.insert(0,parvals_ld["B_tpl"])
            init_A2_tpl_entry.delete(0, tk.END)
            init_A2_tpl_entry.insert(0,parvals_ld["A2_tpl"])
            init_B2_tpl_entry.delete(0, tk.END)
            init_B2_tpl_entry.insert(0,parvals_ld["B2_tpl"])
            init_A3_tpl_entry.delete(0, tk.END)
            init_A3_tpl_entry.insert(0,parvals_ld["A3_tpl"])
            init_B3_tpl_entry.delete(0, tk.END)
            init_B3_tpl_entry.insert(0,parvals_ld["B3_tpl"])
            
        if header[142]=='1':# ie if the qpl is present in the save, add the function with the saved param values
            
            add_qpl()
    
            init_x1_qpl_entry.delete(0, tk.END)
            init_x1_qpl_entry.insert(0,parvals_ld["x1_qpl"])
            init_x2_qpl_entry.delete(0, tk.END)
            init_x2_qpl_entry.insert(0,parvals_ld["x2_qpl"])
            init_x3_qpl_entry.delete(0, tk.END)
            init_x3_qpl_entry.insert(0,parvals_ld["x3_qpl"])
            init_A_qpl_entry.delete(0, tk.END)
            init_A_qpl_entry.insert(0,parvals_ld["A_qpl"])
            init_B_qpl_entry.delete(0, tk.END)
            init_B_qpl_entry.insert(0,parvals_ld["B_qpl"])
            init_A2_qpl_entry.delete(0, tk.END)
            init_A2_qpl_entry.insert(0,parvals_ld["A2_qpl"])
            init_B2_qpl_entry.delete(0, tk.END)
            init_B2_qpl_entry.insert(0,parvals_ld["B2_qpl"])
            init_A3_qpl_entry.delete(0, tk.END)
            init_A3_qpl_entry.insert(0,parvals_ld["A3_qpl"])
            init_B3_qpl_entry.delete(0, tk.END)
            init_B3_qpl_entry.insert(0,parvals_ld["B3_qpl"])
            init_A4_qpl_entry.delete(0, tk.END)
            init_A4_qpl_entry.insert(0,parvals_ld["A4_qpl"])
            init_B4_qpl_entry.delete(0, tk.END)
            init_B4_qpl_entry.insert(0,parvals_ld["B4_qpl"])
        
            
        if header[159]=='1':# ie if the quint pl is present in the save, add the function with the saved param values
            
            add_quint_pl()
    
            init_x1_5pl_entry.delete(0, tk.END)
            init_x1_5pl_entry.insert(0,parvals_ld["x1_5pl"])
            init_x2_5pl_entry.delete(0, tk.END)
            init_x2_5pl_entry.insert(0,parvals_ld["x2_5pl"])
            init_x3_5pl_entry.delete(0, tk.END)
            init_x3_5pl_entry.insert(0,parvals_ld["x3_5pl"])
            init_x4_5pl_entry.delete(0, tk.END)
            init_x4_5pl_entry.insert(0,parvals_ld["x4_5pl"])
            init_A_5pl_entry.delete(0, tk.END)
            init_A_5pl_entry.insert(0,parvals_ld["A_5pl"])
            init_B_5pl_entry.delete(0, tk.END)
            init_B_5pl_entry.insert(0,parvals_ld["B_5pl"])
            init_A2_5pl_entry.delete(0, tk.END)
            init_A2_5pl_entry.insert(0,parvals_ld["A2_5pl"])
            init_B2_5pl_entry.delete(0, tk.END)
            init_B2_5pl_entry.insert(0,parvals_ld["B2_5pl"])
            init_A3_5pl_entry.delete(0, tk.END)
            init_A3_5pl_entry.insert(0,parvals_ld["A3_5pl"])
            init_B3_5pl_entry.delete(0, tk.END)
            init_B3_5pl_entry.insert(0,parvals_ld["B3_5pl"])
            init_A4_5pl_entry.delete(0, tk.END)
            init_A4_5pl_entry.insert(0,parvals_ld["A4_5pl"])
            init_B4_5pl_entry.delete(0, tk.END)
            init_B4_5pl_entry.insert(0,parvals_ld["B4_5pl"])
            init_A5_5pl_entry.delete(0, tk.END)
            init_A5_5pl_entry.insert(0,parvals_ld["A5_5pl"])
            init_B5_5pl_entry.delete(0, tk.END)
            init_B5_5pl_entry.insert(0,parvals_ld["B5_5pl"])  
                   
    def fit_sum_hndl():
        try:
            fit_summary
            
        except NameError:tk.messagebox.showwarning("No Results", "No uncertainties and statistics yet, run a fit first")

        
        else:
            summ_window=tk.Toplevel()
            summ_window.title("Parameter Uncertainty Summary")
            summary_text=tk.Message(summ_window, text=fit_summary)
            summary_text.pack(padx=10, pady=10)
            
            tk.Label(summ_window, text="Absolute Uncertainties").pack()
            uncerts_text=tk.Message(summ_window, text=param_uncert_calced)
            uncerts_text.pack(padx=10, pady=10)
            
            

            
            summ_window.mainloop()
    
    def on_selection(event):
        selection = combo.get()
        option_handlers={"Load Parameters":load_btn_hndl,
                        "Save Parameters":save_btn_hndl,
                        "Close (and proceed to next interval if set)":close_btn_hndl,
                        "Save Spectrum":spec_save_hndl,
                        "Preview Parameters":preview_btn_hndl,
                        "Perform Fit":fit_btn_hndl,
                        "Summary of fit statistics and uncerainties":fit_sum_hndl}
        option_handlers[selection]()
        
    fit_window_options=["Load Parameters","Save Parameters","Close (and proceed to next interval if set)","Save Spectrum","Preview Parameters","Perform Fit","Summary of fit statistics and uncerainties"]
    combo=ttk.Combobox(window_buttons, values=fit_window_options)
    combo.bind('<<ComboboxSelected>>', on_selection)
    combo.pack()
    
    window_buttons.mainloop()#this creates the gui window as defined above



#%%main program handling
        
        
def inspex(x_data,y_data,uncert,date,inst,spec_type):# mainloop function for the curve fitting window
    
    #%initialise values for the fit params: initial values, vary, max value and min value
    global init
    init=dict()   
    global vary
    vary=dict()  
    global maxval
    maxval=dict()    
    global minval
    minval=dict()
    
    
    
    #initially, no funcions are present. set this for all functions
    global therm_func_pres
    therm_func_pres=0
    
    global bpl_pres
    bpl_pres=0
    
    global gauss_pres
    gauss_pres=0
    
    global power_pres
    power_pres=0
    
    global kappa_pres
    kappa_pres=0
    
    global bpl_and_therm_pres
    bpl_and_therm_pres=0

    global double_therm_func_pres
    double_therm_func_pres=0

    global tpl_pres
    tpl_pres=0
    
    global qpl_pres
    qpl_pres=0
    
    global quint_pl_pres
    quint_pl_pres=0

    #show the spectrum
    
    plot_wind_size=(4,3)#define the window size for the plots

    fit_window=tk.Toplevel()
    fit_window.title('Initial fit preview')
    fig_fit =plt.Figure(figsize=plot_wind_size, dpi=300)
    ax_fit= fig_fit.add_subplot(1, 1, 1)


    #plot data
    ax_fit.scatter(list(x_data),list(y_data))
    ax_fit.set_xlabel("Energy (keV)")
    ax_fit.set_ylabel("Electron flux\n"+r"(cm$^2$ sr s keV)$^{-1}$")
    ax_fit.set_yscale("log")
    ax_fit.set_xscale("log")
    #set plot limits so that it is focussed on the data, to avoid scaling issues from fitted curve
    ax_fit.set_ylim(min(y_data)/2,max(y_data)*2) 
    ax_fit.set_xlim(min(x_data),max(x_data))
    
    #add legend to plot
    ax_fit.set_title(f"Spectrum to fit {date}")

    #add error bars
    for count,i in enumerate(list(x_data)):
        this_y=list(y_data)[count]
        this_err=list(uncert)[count]
        ax_fit.plot([i,i],[this_y-this_err,this_y+this_err],color='k', linestyle='-', linewidth=2)

    ax_fit.grid()
    global canvas_fit
    canvas_fit = FigureCanvasTkAgg(fig_fit, master=fit_window) 
    canvas_fit.draw()  
    canvas_fit.get_tk_widget().pack()
    
    
    
    build_fit_window(x_data,y_data,uncert,date,inst,spec_type)
    fit_window.mainloop()
    


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
    
    
#%%spectrum generation for a single time
def interval_spec_gen(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime,intervals):
    bg_spec, bg_spec_uncert=avg_bg_calc(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime)#background generating function
    #print(time_series_time)
    spectra=list()
    for interval in intervals:
        
        #find time in data closest to selected time
        closest=min(time_series_time, key=lambda x: abs(x - interval))
        
        pos=list(time_series_time).index(closest)
        this_spec_raw=time_series_data[pos,:]
        this_spec_uncert_raw=time_series_uncert[pos,:]

        this_spec=this_spec_raw-list(bg_spec.values())
        this_spec_uncert=np.sqrt(np.array(this_spec_uncert_raw)**2+np.array(list(bg_spec_uncert.values()))**2)
        spectra.append((this_spec,this_spec_uncert))
    
    
    return spectra
#%%peak/fluence time range and background selection

def time_rng_select(inst, start_time, end_time,spec_type_sel):#function for the time range window
    
    

#    window_inst.destroy()#closes instrument window window

    #load in data for selected probe. list of times, list of energies in keV, array of data in (times by energies), array of uncerts in (times by energies)
    if inst=="SolO-STEP":
        if dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S")<dt.datetime.strptime('2021/10/22',"%Y/%m/%d"):#time before recalibration:
            time_series_time,time_series_energies,time_series_data,time_series_uncert=load_early_step_data(start_time, end_time)
        else:#porst-recalibration
            time_series_time,time_series_energies,time_series_data,time_series_uncert=load_late_step_data(start_time, end_time)
            
            #later data recalibrated and changed-must have different routines to interpret
    
    if inst=="STEREO STE":
        time_series_time,time_series_energies,time_series_data,time_series_uncert=stereo_data_load(start_time, end_time)
    
    
    
    #slice loaded data to range selected by user, as generally loads in full days
    
    
    #set range to user defined fitting limits
    x_data_sliced=list()
    y_data_sliced=list()
    uncert_sliced=list()
    for pos,time in enumerate(time_series_time):
      if time>=dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S")  and time<=dt.datetime.strptime(end_time,"%Y/%m/%d %H:%M:%S"):
          x_data_sliced.append(time)
          y_data_sliced.append(time_series_data[pos][:])
          uncert_sliced.append(time_series_uncert[pos][:])
    
    time_series_time_raw=time_series_time
    time_series_data_raw=time_series_data
    time_series_uncert_raw= time_series_uncert
    
    
    
    
    time_series_time=np.array(x_data_sliced)
    time_series_data=np.array(y_data_sliced)
    time_series_uncert=np.array(uncert_sliced)     

    
    
    
    
    
    #display loaded data
    TS_window=tk.Tk()#create window for the range selection
    TS_window.title("Select background and spectrum ranges")
    fig_TS =plt.Figure(figsize=(4,3), dpi=300)    
    ax_TS= fig_TS.add_subplot(1, 1, 1)
    

    
    #set global variables
    global tot
    global sliders
    global ys
    global set_funcs
    #define some lists to put sliders and the set functions into 

    sliders=[]#where the sliders are stored 
    ys=[]#currently not settled fits
    set_funcs=[]#for fits that have been set 
    #print('loaded pack')
    

    global slider_num#global variable
    global low_x
    global upper_x
    slider_num=4 #number of sliders

    tsres="15min"

    def update(idx): #an update function that gets called everytime the sliders get sild around 

        low_x=ax_TS.get_xlim()[0]
        upper_x=ax_TS.get_xlim()[1]#get the x axis limits 
        ax_TS.cla() #clears the plot but leaves the window open 
        
        for channel in [0, 4, 8, 12, 16, 20, 24, 28, 30]:
            label=f'{round(time_series_energies[channel],2)} keV'
            
            pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=label,linewidth=0.75)#
        
        ax_TS.set_xlim(min(time_series_time),max(time_series_time))
        #ax_TS.set_ylim(bottom=time_series_data_raw.min())
        time_range_s=max(time_series_time)-min(time_series_time)        
        
        bg_mintime=min(time_series_time)+(sliders[0].get()*time_range_s)
        bg_maxtime=min(time_series_time)+(sliders[1].get()*time_range_s)
        spec_mintime=min(time_series_time)+(sliders[2].get()*time_range_s)
        spec_maxtime=min(time_series_time)+(sliders[3].get()*time_range_s)
        
        line_top=np.nanmax(time_series_data_raw)
        width=0.5
        ax_TS.vlines(bg_mintime, 0, line_top, colors='k',linestyles=[(0,(9,3,4,4))],linewidth=width)
        ax_TS.vlines(bg_maxtime, 0, line_top, colors='k',linestyles=[(0,(9,3,4,4))],linewidth=width)
        ax_TS.vlines(spec_mintime, 0, line_top, colors='k',linestyles=[(0,(9,3,4,4))],linewidth=width)
        ax_TS.vlines(spec_maxtime, 0, line_top, colors='k',linestyles=[(0,(9,3,4,4))],linewidth=width)

        
        fig_TS.canvas.draw_idle()#this draws it on the canvas and end of update 
        

    low_x=ax_TS.get_xlim()[0]
    upper_x=ax_TS.get_xlim()[1]#sets the max and min for the location slider
    slider_res=0.01
    slider_frame=tk.Frame(master=TS_window)
    
    time_range_s=max(time_series_time)-min(time_series_time)
    half_time=min(time_series_time)+(0.5*time_range_s)
    
    #bg min slider
    sliders.append(tk.Scale(master=slider_frame,from_=0, to=1,resolution=slider_res,command=update,orient=tk.HORIZONTAL,label='BG min'))
    sliders[0].set((upper_x+low_x)/2)#this sets the initial value 
    sliders[0].pack(side=tk.TOP)   #defines where it is in the window 

    bg_min_ent=tk.Entry(master=slider_frame,fg="black", bg="white", width=10)
    bg_min_ent.pack(side=tk.BOTTOM)
    bg_min_ent.delete(0, tk.END)
    bg_min_ent.insert(0,half_time)
    
    
    
    #bgmax slider
    sliders.append(tk.Scale(master=slider_frame,from_=0, to=1,resolution=slider_res,command=update,orient=tk.HORIZONTAL,label='BG max'))
    sliders[1].set((upper_x+low_x)/2)#this sets the initial value 
    sliders[1].pack(side=tk.TOP)   #defines where it is in the window 
    
    bg_max_ent=tk.Entry(master=slider_frame,fg="black", bg="white", width=10)
    bg_max_ent.pack(side=tk.BOTTOM)
    bg_max_ent.delete(0, tk.END)
    bg_max_ent.insert(0,half_time)
    
    #specmin slider
    sliders.append(tk.Scale(master=slider_frame,from_=0, to=1,resolution=slider_res,command=update,orient=tk.HORIZONTAL,label='Spec min'))
    sliders[2].set((upper_x+low_x)/2)#this sets the initial value 
    sliders[2].pack(side=tk.TOP)   #defines where it is in the window 
    
    specmin_ent=tk.Entry(master=slider_frame,fg="black", bg="white", width=10)
    specmin_ent.pack(side=tk.BOTTOM)
    specmin_ent.delete(0, tk.END)
    specmin_ent.insert(0,half_time)
    
    #specmax slider
    sliders.append(tk.Scale(master=slider_frame,from_=0, to=1,resolution=slider_res,command=update,orient=tk.HORIZONTAL,label='Spec max'))
    sliders[3].set((upper_x+low_x)/2)#this sets the initial value 
    sliders[3].pack(side=tk.TOP)   #defines where it is in the window 

    specmax_ent=tk.Entry(master=slider_frame,fg="black", bg="white", width=10)
    specmax_ent.pack(side=tk.BOTTOM)
    specmax_ent.delete(0, tk.END)
    specmax_ent.insert(0,half_time)

    slider_frame.pack(side=tk.LEFT)
    

    
    
    def TS_Select_btn_hndl():
        
        time_range_s=max(time_series_time)-min(time_series_time)
        bg_mintime=min(time_series_time)+(sliders[0].get()*time_range_s)
        bg_maxtime=min(time_series_time)+(sliders[1].get()*time_range_s)
        spec_mintime=min(time_series_time)+(sliders[2].get()*time_range_s)
        spec_maxtime=min(time_series_time)+(sliders[3].get()*time_range_s)
        #print(np.shape(time_series_data))
        selected_func=spec_type_sel
        if selected_func=='fluence':
            spec,spec_uncert=fluence_spec_gen(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime,spec_mintime,spec_maxtime)
            spec_type='fluence'
            
            
        if selected_func=='peak flux':
            spec,spec_uncert=peak_flux_spec_gen(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime,spec_mintime,spec_maxtime)
            spec_type='peak_flux'
        
        
        
        date= spec_mintime.strftime("%d-%m-%Y")        
        TS_window.destroy()
        plt.close(fig_TS)
        inspex(time_series_energies, spec, spec_uncert, date, inst, spec_type)
    

    
    button = tk.Button(master=TS_window, text="Select this background and spectrum range", command=TS_Select_btn_hndl)
    button.pack(side=tk.TOP)
    
    
    #for a selection of energy channels, convert from array to pd.series, resample for clarity then plot with appropriate label
    for channel in [0, 4, 8, 12, 16, 20, 24, 28, 30]:
        pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=f'{round(time_series_energies[channel],2)} keV')#
    ax_TS.set_ylim(bottom=1)
    
    
    canvas_TS = FigureCanvasTkAgg(fig_TS, master=TS_window) 
    canvas_TS.draw()  
    canvas_TS.get_tk_widget().pack()
    
    
    
    TS_window.mainloop()
    
    
    


    







#%% instrument selection window


def instrument_choice():#function for the instrument choice window
    global window_inst
    window_inst = tk.Tk()#define window
    window_inst.title('Select instrument and time range for spectral calculations')
    window_inst.minsize(400, 400)
    
    frame_instopts=tk.Frame(master=window_inst)
    
    
    greeting = tk.Label(master=frame_instopts,text="Inspex fitting GUI")#window name. MUST only have one tk.Tk(), all else must be .toplevel else crashes. This is .TK as is first to open 
    greeting.pack()
    OPTIONS = [
        "SolO STEP",
        "STEREO STE"
        ]     
    variable = tk.StringVar()
    variable.set(OPTIONS[0]) # default value
    
    inst_opts = tk.OptionMenu(frame_instopts, variable, *OPTIONS)
    inst_opts.pack()


    label_explain=tk.Label(master=frame_instopts, text='Enter times in format: YYYY/mm/dd HH:MM:SS')
    label_explain.pack(side=tk.LEFT)
        
    
    
    label_fitlims=tk.Label(master=frame_instopts, text='Start time')
    label_fitlims.pack(side=tk.LEFT)
    
    start_entry = tk.Entry(master=frame_instopts,fg="black", bg="white", width=10)
    start_entry.pack(side=tk.LEFT)
    
    label_fitlims=tk.Label(master=frame_instopts, text='End time')
    label_fitlims.pack(side=tk.LEFT)
    
    end_entry = tk.Entry(master=frame_instopts,fg="black", bg="white", width=10)
    end_entry.pack(side=tk.LEFT)
    
    
    frame_specopts=tk.Frame(master=window_inst)
    OPTIONS = [
        "fluence","peak flux","flux at set time(s)"
        
        ]     
    var_spec_type = tk.StringVar()
    var_spec_type.set(OPTIONS[1]) # default value    
    spec_opts = tk.OptionMenu(frame_specopts, var_spec_type, *OPTIONS)
    spec_opts.pack()
    frame_specopts.pack(side=tk.LEFT)
    
    
    #define spectrum loading
    def load_spec_hndl():
        file_obj=tk.filedialog.askopenfile()
        spec_df=pd.read_csv(file_obj)
        file_obj.close()
        load_energies=spec_df['energies'].values
        load_fluxes=spec_df['fluxes'].values
        load_uncerts=spec_df['errors'].values
        date=spec_df['date'].values[0]
        inst=spec_df['inst'].values[0]
        spec_type=spec_df['spec_type'].values[0]
        window_inst.destroy()#closes current window
    
        inspex(load_energies, load_fluxes, load_uncerts, date, inst, spec_type)

    

        
    #create spectrum load button
    load_spec_button=tk.Button(
    master=window_inst,
    text="Load Previously created spectrum",
    width=25,
    height=5,
    bg="white",
    fg="black",
    command=load_spec_hndl
    )
    load_spec_button.pack(side=tk.BOTTOM)
    
    
    
    
    
    def validate_date(date):
        pattern = r'^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}$'
        if len(date) == 19 and re.match(pattern, date):
            return True
        return False
    
    
    def inst_opts_select():
        
        start_time=start_entry.get()#gets the entered value
        end_time=end_entry.get()#gets the entered value
        if not (validate_date(start_time) and validate_date(end_time)):
            tk.messagebox.showerror("Invalid Input",'Dates must have format: YYYY/mm/dd HH:MM:SS')
        else:
            selected_func=variable.get()
            if selected_func=='SolO STEP':
                inst="SolO-STEP"
            if selected_func=='STEREO STE':
                inst="STEREO STE"
                    
                    
                    
            window_inst.destroy()#closes current window
            window_inst.update()
        
            spec_type_sel=var_spec_type.get()
            
            if spec_type_sel=="fluence" or spec_type_sel=="peak flux":
                time_rng_select(inst,start_time,end_time,spec_type_sel)#runs time range selection function
            
            elif spec_type_sel=="flux at set time(s)":
                intervals_select(inst,start_time,end_time,spec_type_sel)
        
        



    button = tk.Button(master=frame_instopts, text="Choose Instrument, data dates, and spectrum type", command=inst_opts_select)
    button.pack(side=tk.BOTTOM)

    frame_instopts.pack()   


    window_inst.mainloop()

#%%window to select time intervals

def intervals_select(inst,start_time,end_time,spec_type_sel):
    
#    window_inst.destroy()#closes instrument window window

    #load in data for selected probe. list of times, list of energies in keV, array of data in (times by energies), array of uncerts in (times by energies)
    if inst=="SolO-STEP":
        if dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S")<dt.datetime.strptime('2021/10/22',"%Y/%m/%d"):#time before recalibration:
            time_series_time,time_series_energies,time_series_data,time_series_uncert=load_early_step_data(start_time, end_time)
        else:#porst-recalibration
            time_series_time,time_series_energies,time_series_data,time_series_uncert=load_late_step_data(start_time, end_time)
            
            #later data recalibrated and changed-must have different routines to interpret
    
    if inst=="STEREO STE":
        time_series_time,time_series_energies,time_series_data,time_series_uncert=stereo_data_load(start_time, end_time)
    
    
    spec_type="intervals" #the type of the spectra this generates
    
    #slice loaded data to range selected by user, as generally loads in full days    
    #set range to user defined fitting limits
    x_data_sliced=list()
    y_data_sliced=list()
    uncert_sliced=list()
    for pos,time in enumerate(time_series_time):
      if time>=dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S")  and time<=dt.datetime.strptime(end_time,"%Y/%m/%d %H:%M:%S"):
          x_data_sliced.append(time)
          y_data_sliced.append(time_series_data[pos][:])
          uncert_sliced.append(time_series_uncert[pos][:])
    
    time_series_time_raw=time_series_time
    time_series_data_raw=time_series_data
    time_series_uncert_raw= time_series_uncert
    
    
    
    
    time_series_time=np.array(x_data_sliced)
    time_series_data=np.array(y_data_sliced)
    time_series_uncert=np.array(uncert_sliced)
    
    #display loaded data
    inters_window=tk.Tk()#create window for the range selection
    inters_window.title("Select background and spectrum intervals")
    fig_TS =plt.Figure(figsize=(4,3), dpi=300)    
    ax_TS= fig_TS.add_subplot(1, 1, 1)
    
    tsres="15min"

    #for a selection of energy channels, convert from array to pd.series, resample for clarity then plot with appropriate label
    for channel in [0, 4, 8, 12, 16, 20, 24, 28, 30]:
        pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=f'{round(time_series_energies[channel],2)} keV')#
    #ax_TS.set_ylim(bottom=time_series_data_raw.min())
    

    canvas_TS = FigureCanvasTkAgg(fig_TS, master=inters_window) 
    canvas_TS.draw()  
    canvas_TS.get_tk_widget().pack(side=tk.RIGHT)
    
    #user selects number of intervals they want, and whether they want evenly spaced or individually set
    frame_inter_opts=tk.Frame(master=inters_window)
    
    
    label_inter_opts=tk.Label(master=frame_inter_opts, text='Select number of intervals and method of generation')
    label_inter_opts.pack(side=tk.TOP)
    
    OPTIONS = [
        "Generate intervals",
        "Select intervals"
        ]     
    method_variable = tk.StringVar()
    method_variable.set(OPTIONS[0]) # default value
    
    inst_opts = tk.OptionMenu(frame_inter_opts, method_variable, *OPTIONS)
    inst_opts.pack()
    
    no_inter_ent= tk.Entry(master=frame_inter_opts,fg="black", bg="white", width=10)
    no_inter_ent.pack()
    
    inter_method=tk.StringVar()
    no_inter=tk.IntVar()
    global frame_inter_gen
    frame_inter_gen=None
    inter_method=tk.StringVar()
    interval_length=tk.IntVar()

        
        
    #this handles the button for method of generation, including how the intervals are generated
    def inter_gen_hndl():
        global frame_inter_gen
        inter_method.set(method_variable.get())
        no_inter.set(no_inter_ent.get())
        global slider_num
        slider_num=no_inter.get()

        
        if frame_inter_gen==None:#if fram doesn't exist yet, generate. if exists, destroy and re-create
            frame_inter_gen=tk.Frame(master=inters_window)
            
#        else:
 #           frame_inter_gen.pack_forget()
  #          frame_inter_gen=tk.Frame(master=inters_window)
        
        
        if inter_method.get()=="Select intervals":
            add_sliders(no_inter.get())
            
        if inter_method.get()=="Generate intervals":
            global interval_length_ent
            label_inter_len=tk.Label(master=frame_inter_gen, text='Select duration of interval in seconds')
            label_inter_len.pack()
            
            interval_length_ent=tk.Entry(master=frame_inter_gen,fg="black", bg="white", width=10)
            interval_length_ent.pack()
            
            label_inter_start=tk.Label(master=frame_inter_gen, text='Select first interval')
            label_inter_start.pack()
            
            low_x=ax_TS.get_xlim()[0]
            upper_x=ax_TS.get_xlim()[1]#sets the max and min for the location slider
            slider_res=0.01
            sliders_ints.append(tk.Scale(master=frame_inter_gen,from_=0, to=1,resolution=slider_res,command=interval_generate,orient=tk.HORIZONTAL,label='Interval 0'))
            sliders_ints[0].set(1/2)#this sets the initial value 
            sliders_ints[0].pack()   #defines where it is in the window 
            frame_inter_gen.pack(side=tk.LEFT)
            
        
        
        
    def update_interval(idx):
        low_x=ax_TS.get_xlim()[0]
        upper_x=ax_TS.get_xlim()[1]#get the x axis limits 
        ax_TS.cla() #clears the plot but leaves the window open 
        
        for channel in [0, 4, 8, 12, 16, 20, 24, 28, 30]:
            label=f'{round(time_series_energies[channel],2)} keV'
            
            pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=label,linewidth=0.75)#
        
        ax_TS.set_xlim(min(time_series_time),max(time_series_time))
        #ax_TS.set_ylim(bottom=time_series_data_raw.min())
        time_range_s=max(time_series_time)-min(time_series_time)        
        
        bg_mintime=min(time_series_time)+(sliders[0].get()*time_range_s)
        bg_maxtime=min(time_series_time)+(sliders[1].get()*time_range_s)

        
        line_top=np.nanmax(time_series_data_raw)
        width=0.5
        ax_TS.vlines(bg_mintime, 0, line_top, colors='k',linestyles=[(0,(9,3,4,4))],linewidth=width)
        ax_TS.vlines(bg_maxtime, 0, line_top, colors='k',linestyles=[(0,(9,3,4,4))],linewidth=width)
        
        
        for i in range(slider_num):
            int_time=min(time_series_time)+(sliders_ints[i].get()*time_range_s)
            if int_time<=max(time_series_time):
                ax_TS.vlines(int_time, 0, line_top, colors='r',linestyles=[(0,(9,3,4,4))],linewidth=width)
            
        
        fig_TS.canvas.draw_idle()#this draws it on the canvas and end of update 
    
    def add_sliders(slider_num):#function that runs for manual interval select
        #add number of intervals if this is what user selects        
        for i in range(slider_num):
            #interval slider
            sliders_ints.append(tk.Scale(master=frame_inter_gen,from_=0, to=1,resolution=slider_res,command=update_interval,orient=tk.HORIZONTAL,label=f'Interval {i}'))
            sliders_ints[i].set((upper_x+low_x)/2)#this sets the initial value 
            sliders_ints[i].pack(side=tk.BOTTOM)   #defines where it is in the window 
        frame_inter_gen.pack(side=tk.LEFT)
    
    
    def interval_generate(idx):#an update function that gets called everytime the slider initial slider get sild around for interval generation
        interval_length.set(interval_length_ent.get()) 
        low_x=ax_TS.get_xlim()[0]
        upper_x=ax_TS.get_xlim()[1]#get the x axis limits 
        ax_TS.cla() #clears the plot but leaves the window open 
        
        for channel in [0, 4, 8, 12, 16, 20, 24, 28, 30]:
            label=f'{round(time_series_energies[channel],2)} keV'
            
            pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=label,linewidth=0.75)#
        
        ax_TS.set_xlim(min(time_series_time),max(time_series_time))
        #ax_TS.set_ylim(bottom=time_series_data_raw.min())
        time_range_s=max(time_series_time)-min(time_series_time)        
        
        bg_mintime=min(time_series_time)+(sliders[0].get()*time_range_s)
        bg_maxtime=min(time_series_time)+(sliders[1].get()*time_range_s)
        
        inter_0=min(time_series_time)+(sliders_ints[0].get()*time_range_s)
        
        line_top=np.nanmax(time_series_data_raw)
        width=0.5
        ax_TS.vlines(bg_mintime, 0, line_top, colors='k',linestyles=[(0,(9,3,4,4))],linewidth=width)
        ax_TS.vlines(bg_maxtime, 0, line_top, colors='k',linestyles=[(0,(9,3,4,4))],linewidth=width)
        ax_TS.vlines(inter_0, 0, line_top, colors='r',linestyles=[(0,(9,3,4,4))],linewidth=width)        
        
        inters=[inter_0+(dt.timedelta(seconds=interval_length.get())*i) for i in np.arange(1,slider_num)]

        for i in inters:
            if i<=max(time_series_time):
                ax_TS.vlines(i, 0, line_top, colors='r',linestyles=[(0,(9,3,4,4))],linewidth=width)

        
        fig_TS.canvas.draw_idle()#this draws it on the canvas and end of update 
    
    
    
        
        
        
        
    #select these options and generate the frame that allows the user to use their selected interval generation method
    inter_sel_btn=tk.Button(master=frame_inter_opts, text="Select these interval options", command=inter_gen_hndl,width=25, height=2, bg="white", fg="black") 
    inter_sel_btn.pack(side=tk.BOTTOM)
    
    frame_inter_opts.pack(side=tk.LEFT)

    
    #set global variables
    global tot
    global sliders
    global sliders_ints
    global ys
    global set_funcs
    #define some lists to put sliders and the set functions into 

    sliders=[]#where the sliders are stored 
    sliders_ints=[]#to contain the sliders for interval selection
    ys=[]#currently not settled fits
    set_funcs=[]#for fits that have been set 
    #print('loaded pack')
    


    global low_x
    global upper_x
    



    def update_bg(idx): #an update function that gets called everytime the sliders get sild around 

        low_x=ax_TS.get_xlim()[0]
        upper_x=ax_TS.get_xlim()[1]#get the x axis limits 
        ax_TS.cla() #clears the plot but leaves the window open 
        
        for channel in [0, 4, 8, 12, 16, 20, 24, 28, 30]:
            label=f'{round(time_series_energies[channel],2)} keV'
            
            pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=label,linewidth=0.75)#
        
        ax_TS.set_xlim(min(time_series_time),max(time_series_time))
        #ax_TS.set_ylim(bottom=time_series_data_raw.min())
        time_range_s=max(time_series_time)-min(time_series_time)        
        
        bg_mintime=min(time_series_time)+(sliders[0].get()*time_range_s)
        bg_maxtime=min(time_series_time)+(sliders[1].get()*time_range_s)
        
        
        line_top=np.nanmax(time_series_data_raw)
        width=0.5
        ax_TS.vlines(bg_mintime, 0, line_top, colors='k',linestyles=[(0,(9,3,4,4))],linewidth=width)
        ax_TS.vlines(bg_maxtime, 0, line_top, colors='k',linestyles=[(0,(9,3,4,4))],linewidth=width)
        
        
        #must update the intervals so they re-appear after bg slider adjustment
        #for generated
        if inter_method.get()=="Generate intervals":
            inter_0=min(time_series_time)+(sliders_ints[0].get()*time_range_s)
            inters=[inter_0+(dt.timedelta(seconds=interval_length.get())*i) for i in np.arange(1,no_inter.get())]
            ax_TS.vlines(inter_0, 0, line_top, colors='r',linestyles=[(0,(9,3,4,4))],linewidth=width)
            for i in inters:
                if i<=max(time_series_time):
                    ax_TS.vlines(i, 0, line_top, colors='r',linestyles=[(0,(9,3,4,4))],linewidth=width)
         
                    
        #for selected
        if inter_method.get()=="Select intervals":
            for i in range(slider_num):
                int_time=min(time_series_time)+(sliders_ints[i].get()*time_range_s)
                if int_time<=max(time_series_time):
                    ax_TS.vlines(int_time, 0, line_top, colors='r',linestyles=[(0,(9,3,4,4))],linewidth=width)
         
        
        fig_TS.canvas.draw_idle()#this draws it on the canvas and end of update 


    low_x=ax_TS.get_xlim()[0]
    upper_x=ax_TS.get_xlim()[1]#sets the max and min for the location slider
    slider_res=0.01
    slider_frame=tk.Frame(master=inters_window)
    
    #bg min slider
    sliders.append(tk.Scale(master=slider_frame,from_=0, to=1,resolution=slider_res,command=update_bg,orient=tk.HORIZONTAL,label='BG min'))
    sliders[0].set(1/2)#this sets the initial value 
    sliders[0].pack(side=tk.BOTTOM)   #defines where it is in the window 

    #bgmax slider
    sliders.append(tk.Scale(master=slider_frame,from_=0, to=1,resolution=slider_res,command=update_bg,orient=tk.HORIZONTAL,label='BG max'))
    sliders[1].set(1/2)#this sets the initial value 
    sliders[1].pack(side=tk.BOTTOM)   #defines where it is in the window  

        


    slider_frame.pack(side=tk.LEFT)
    
    #option to do all fits mannually or loop using previous fits
    
    frame_loop_opts=tk.Frame(master=inters_window)
    global manl_loop
    manl_loop=False#auto loop by default
    def hndl_btn_manl_loop():#it's a check button, so swaps whether is on or off
        global manl_loop
        manl_loop=not manl_loop         
    btn_manl_loop=tk.Checkbutton(master=frame_loop_opts,text="Manually loop through intervals", height=2, bg="white",fg="black",command=hndl_btn_manl_loop)
    btn_manl_loop.pack(side=tk.BOTTOM)
    if  manl_loop:btn_manl_loop.select()
    if not  manl_loop:btn_manl_loop.deselect()
    
    

    
    frame_loop_opts.pack(side=tk.LEFT)
    def TS_Select_btn_hndl():
        global param_uncert_calced
        global parvals_new
        global parvals
        
        time_range_s=max(time_series_time)-min(time_series_time)
        bg_mintime=min(time_series_time)+(sliders[0].get()*time_range_s)
        bg_maxtime=min(time_series_time)+(sliders[1].get()*time_range_s)

        #print(np.shape(time_series_data))
        
        
        #SPECTRUM GENERATION
        
        intervals=[]
        
        #for generated
        if inter_method.get()=="Generate intervals":
            inter_0=min(time_series_time)+(sliders_ints[0].get()*time_range_s)
            inters=[inter_0+(dt.timedelta(seconds=interval_length.get())*i) for i in np.arange(1,no_inter.get())]
            inters.insert(0,inter_0)#remember to add first interval
            intervals=inters

        #for selected
        if inter_method.get()=="Select intervals":
            for i in range(slider_num):
                int_time=min(time_series_time)+(sliders_ints[i].get()*time_range_s)
                intervals.append(int_time)
                
        spectra=interval_spec_gen(time_series_time, time_series_energies, time_series_data, time_series_uncert, bg_mintime, bg_maxtime, intervals)
        
        date= min(time_series_time)        
        inters_window.destroy()
        plt.close(fig_TS)
        
        global fit_window
        global preview_window
        fitted_params=np.empty([len(spectra),2],dtype=dict)
        if manl_loop: #if user wishes to do all fits mannually
            for count,i in enumerate(spectra):#fit all the spectra
                spec=i[0]
                spec_uncert=i[1]
                inspex(time_series_energies, spec, spec_uncert, intervals[count], inst, spec_type)
                fitted_params[count,0]=parvals_new
                fitted_params[count,1]=param_uncert_calced
                

        else: #auto loop intervals after first one
            inspex(time_series_energies, spectra[0][0], spectra[0][1], intervals[0], inst, spec_type)
            fitted_params[0,0]=parvals
            fitted_params[0,1]=param_uncert_calced
            output=parvals
            for count,i in enumerate(spectra[1:]):#fit all the spectra
                spec=i[0]
                spec_uncert=i[1]
                parvals,param_uncert_calced=fitting(header,output,vary,minval,maxval,time_series_energies,spec,spec_uncert,fitmin,fitmax)
                #fit_window.quit()
                #preview_window.quit()
                fitted_params[count+1,0]=parvals
                fitted_params[count+1,1]=param_uncert_calced
                output=parvals

        if fit_window is not None:
            #close any open figues
            #fit_window.destroy()
            fit_window=None
            
        if preview_window is not None:
            #close any open figues
            #preview_window.destroy()
            preview_window=None
        
        
        #allow user to save the spectra and the fits
        invl_res_display_wind=tk.Tk()
        
        label_disp_opts=tk.Label(master=invl_res_display_wind,text="Select Parameters to generate time series")
        label_disp_opts.pack(side=tk.TOP)
        
        invl_res_display_opts=tk.Frame(master=invl_res_display_wind)
        
        def param_ev_btn_hndl(key):
            #print(fitted_params)
            this_params=[d[key] for d in fitted_params[:,0]]
            this_uncerts=[d[key] for d in fitted_params[:,1]]
            time=intervals
            
            tev_window=tk.Tk()
            tev_window.title('Time Evolution')
            
            fig_tev =plt.Figure(figsize=(4,3.5), dpi=200)
            ax_tev= fig_tev.add_subplot(1, 1, 1)
            #plot data
            ax_tev.scatter(list(time),list(this_params))
            ax_tev.set_xlabel("Time")
            #add error bars
            for count,i in enumerate(list(time)):
                this_y=list(this_params)[count]
                this_err=list(this_uncerts)[count]
                ax_tev.plot([i,i],[this_y-this_err,this_y+this_err],color='k', linestyle='-', linewidth=2)
            ax_tev.grid()           
            
            canvas_tev = FigureCanvasTkAgg(fig_tev, master=tev_window) 
            canvas_tev.draw()  
            canvas_tev.get_tk_widget().pack()
            #add buttton to save figure
            def fig_save_hndl():
                file_obj=tk.filedialog.asksaveasfilename()
                fig_tev.savefig(file_obj,bbox_inches='tight')
            
            #create preview button
            fig_save_button=tk.Button(
            text="Save Plot",  width=25,  height=2,  bg="white",  fg="black",  command=fig_save_hndl,  master=tev_window)
            fig_save_button.pack(side=tk.BOTTOM)
            
        
        #time series generation for each variable depending on the functions fitted
        # Helper function to create a button
        def create_button(text, param, master):
            return tk.Button(text=text, width=25, height=2, bg="white", fg="black", command=lambda: param_ev_btn_hndl(param), master=master).pack(side=tk.BOTTOM)
        
        # Dictionary mapping header index to the required buttons
        button_definitions = {
            28: ["amp", "T", "alpha"],
            9: ["x1", "B", "B2", "A", "A2"],
            42: ["gauss_amp", "gauss_centre", "sigma"],
            56: ["A_sing", "B_sing"],
            70: [ "A_k", "T_k", "m_i","n_i", "kappa"],
            92: ["amp_c", "T_c", "alpha_c", "x0_c", "x1_c", "B_c", "B2_c"],
            118: ["amp_d_1", "T_d_1", "alpha_d_1","amp_d_2", "T_d_2", "alpha_d_2"],
            130:["x1","x2","A","B","A2","B2","A3","B3"],
            142:["x1","x2","x3","A","B","A2","B2","A3","B3","A4","B4"],
            159:["x1","x2","x3","x4","A","B","A2","B2","A3","B3","A4","B4","A5","B5"]
        }
        
        # Iterate over the button definitions
        for index, buttons in button_definitions.items():
            if header[index] == '1':  # Check if the corresponding function is present
                for button_text in buttons:
                    create_button(button_text, button_text, invl_res_display_opts)
        
        invl_res_display_opts.pack()
        invl_res_display_wind.mainloop()
        
        
    
        
    
    button = tk.Button(master=inters_window, text="Select this background and spectrum intervals", command=TS_Select_btn_hndl)
    button.pack(side=tk.BOTTOM)
    
    
    
    
    
    
    inters_window.mainloop()

#%% run the code to test it
#instrument_choice()

    




