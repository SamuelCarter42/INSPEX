#%%Initial set up
import sys#for file path handling
import os#has general functions for file manipulation

from .build_fit_window import build_fit_window

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

import re#for handling regexs to validate inputs
import random as rn#for random number and choice utility, particularly in uncertainty estimation
from tqdm import tqdm #for tracking progress of long iterables

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


#%%main program handling
        
        
def fitting_gui(x_data,y_data,uncert,date,inst,spec_type):# mainloop function for the curve fitting window
    
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
    
    plot_wind_size=(6,4)#define the window size for the plots

    fit_window=tk.Toplevel()
    fit_window.title('Initial fit preview')
    # Make window resizable
    fit_window.rowconfigure(0, weight=1)
    fit_window.columnconfigure(0, weight=1) 
    fig_fit =plt.Figure(figsize=plot_wind_size, dpi=200)
    fig_fit.tight_layout()
    ax_fit= fig_fit.add_subplot(1, 1, 1)


    #plot data
    ax_fit.scatter(list(x_data),list(y_data))
    ax_fit.set_xlabel("Energy (keV)")
    ax_fit.set_ylabel("Electron flux\n"+r"(cm$^2$ sr s keV)$^{-1}$")
    ax_fit.set_yscale("log")
    ax_fit.set_xscale("log")
    #set plot limits so that it is focussed on the data, to avoid scaling issues from fitted curve
    #including a nan filter max/min so that fill values do not break the code
    ax_fit.set_ylim(np.nanmin(y_data)/2,np.nanmax(y_data)*2) 
    ax_fit.set_xlim(np.nanmin(x_data),np.nanmax(x_data))
    
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
    canvas_fit.get_tk_widget().pack(fill="both",expand=True)
    
    
    
    build_fit_window(x_data,y_data,uncert,date,inst,spec_type)
    fit_window.mainloop()