#%%Initial set up
import sys#for file path handling
import os#has general functions for file manipulation

from .build_fit_window import build_fit_window
from . import state  #shared cross-module state

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

#windows initialised in state module

#setting the fit methods
method_1='basinhopping'#'dual_annealing'#
method_2='lbfgsb'



#%%main program handling
        
        
def fitting_gui(x_data,y_data,uncert,date,inst,spec_type):# mainloop function for the curve fitting window
    

    
    
    
    #initially, no funcions are present. set this for all functions
    state.therm_func_pres=0
    
    state.bpl_pres=0
    
    state.gauss_pres=0
    
    state.power_pres=0
    
    state.kappa_pres=0
    
    state.bpl_and_therm_pres=0

    state.double_therm_func_pres=0

    state.tpl_pres=0
    
    state.qpl_pres=0
    
    state.quint_pl_pres=0
    #reset session state from any previous run
    state.fit_window = None
    state.preview_window = None
    state.resid_window = None
    state.entries = {}

    #show the spectrum
    
    plot_wind_size=(6,4)#define the window size for the plots

    state.fit_window=tk.Toplevel()
    state.fit_window.title('Initial fit preview')
    # Make window resizable
    state.fit_window.rowconfigure(0, weight=1)
    state.fit_window.columnconfigure(0, weight=1) 
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
    canvas_fit = FigureCanvasTkAgg(fig_fit, master=state.fit_window) 
    canvas_fit.draw()  
    canvas_fit.get_tk_widget().pack(fill="both",expand=True)
    
    
    
    build_fit_window(x_data,y_data,uncert,date,inst,spec_type)
    state.fit_window.mainloop()
