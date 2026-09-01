#%%Initial set up
import sys#for file path handling
import os#has general functions for file manipulation

from .par_preview import param_preview
from .param_save_load import param_load, param_save
from .fitting_and_resids import fitting
from . import state  #shared cross-module state

import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk

import numpy as np #general mathematical operations
from scipy.special import erf #imports an erf function for use in some of the fitting operations
from matplotlib import pyplot as plt #general plotting operations
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)#allows plotting to a tkinter window
from solo_epd_loader import epd_load#module for loading SolO EPD data
import datetime as dt#handles general datetime operations
import pandas as pd #module for dataframe and time series handling

#%%window fn

plt.rcParams['figure.dpi'] = 200 #set the dpi for the plots. this can be tuned to improve figure quality

bootstrap_n=30000 #this controls the number of iterations used for the bootstrap to get uncerts, need min 30 for central limit theorem

#initialise some windows as nones so they exist

#setting the fit methods
method_1='basinhopping'#'dual_annealing'#
method_2='lbfgsb'


def build_fit_window(x_data, y_data, uncert, date, inst, spec_type):
    
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
        
        
    #%initialise values for the fit params: initial values, vary, max value and min value
    state.init=dict()   
    state.vary=dict()  
    state.maxval=dict()    
    state.minval=dict()
    
    
    
    
    
    window_buttons = tk.Toplevel()#define window. everything between here and "mainloop" makes up this window". MUST only have one tk.Tk(), all esle must be .toplevel else crashes
    window_buttons.minsize(500, 600)
    window_buttons.title("Inspex fitting GUI")
    state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
    
    
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
        state.entries[f"init_{name_prefix}_entry"] = tk.Entry(frame, width=10)
        state.entries[f"init_{name_prefix}_entry"].insert(0, str(init_val))
        state.entries[f"init_{name_prefix}_entry"].grid(row=row, column=1, padx=5)

        state.entries[f"minval_{name_prefix}_entry"] = tk.Entry(frame, width=10)
        state.entries[f"minval_{name_prefix}_entry"].insert(0, str(min_val))
        state.entries[f"minval_{name_prefix}_entry"].grid(row=row, column=2, padx=5)

        state.entries[f"maxval_{name_prefix}_entry"] = tk.Entry(frame, width=10)
        state.entries[f"maxval_{name_prefix}_entry"].insert(0, str(max_val))
        state.entries[f"maxval_{name_prefix}_entry"].grid(row=row, column=3, padx=5)

        
        state.entries[f"btn_vary_{name_prefix}"] = tk.Checkbutton(frame, text=f"Vary {label_text}", command=callback, variable=tk.IntVar())
        state.entries[f"btn_vary_{name_prefix}"].grid(row=row, column=4, padx=5)
        
        if var_state:state.entries[f"btn_vary_{name_prefix}"].select() 
        else: state.entries[f"btn_vary_{name_prefix}"].deselect()

        tk.Label(frame, text=label_text).grid(row=row, column=0, padx=5, pady=5, sticky="w")



    
    #these functions add the test function components when the user selects/loads them
    
    def add_therm():#add the thermal component to the fitted function
        
        if state.therm_func_pres == 0:#if thermal function not already there
            
            state.init['amp']=1e9
            state.init['T']=12e6
            state.init['alpha']=1
            
            state.vary['amp']=True
            state.vary['T']=True
            state.vary['alpha']=False
            
            state.minval['amp']=0
            state.minval['T']=0
            state.minval['alpha']=0
            
            state.maxval['amp']=1e10
            state.maxval['T']=1e8
            state.maxval['alpha']=5    
            
            
            #defining the part of the GUI window that contains the options for the thermal curve
            global frame_therm 
            frame_therm=tk.Frame(master=frame_params)
            tk.Label(frame_therm, text="Thermal Curve", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_amp(): state.vary['amp'] = not state.vary['amp']
            def toggle_T(): state.vary['T'] = not state.vary['T']
            def toggle_alpha(): state.vary['alpha'] = not state.vary['alpha']

            add_param_row(frame_therm, 1, "amp", state.init['amp'], state.minval['amp'], state.maxval['amp'], state.vary['amp'], toggle_amp, "amp")
            add_param_row(frame_therm, 2, "T", state.init['T'], state.minval['T'], state.maxval['T'], state.vary['T'], toggle_T, "T")
            add_param_row(frame_therm, 3, "alpha", state.init['alpha'], state.minval['alpha'], state.maxval['alpha'], state.vary['alpha'], toggle_alpha, "alpha")

            def hndl_remove_therm_btn():
                frame_therm.destroy()
                state.therm_func_pres = 0

            tk.Button(frame_therm, text='Remove thermal component', command=hndl_remove_therm_btn)\
                .grid(row=4, column=0, columnspan=5, pady=10)

            frame_therm.grid(row=2, column=0, sticky="ew")
            for i in range(6):
                frame_therm.grid_columnconfigure(i, weight=1)
            state.therm_func_pres = 1 #set the thermal function as present
            
            
            
    
    def add_bpl():#function to add the the broken power law
        if state.bpl_pres == 0:
    

            state.init['x1']=40
            state.init['A']=1e5
            state.init['B']=-1
            state.init['A2']=1e5
            state.init['B2']=-2    
            state.init['x0_bpl']=1
            state.init['dx_bpl']=0.1    

    

            state.vary['x1']=True
            state.vary['A']=True
            state.vary['B']=True
            state.vary['A2']=True
            state.vary['B2']=True            
            state.vary['x0_bpl']=True
            state.vary['dx_bpl']=True
            

            state.maxval['x1']=50
            state.maxval['A']=1e10
            state.maxval['B']=0
            state.maxval['A2']=1e10
            state.maxval['B2']=0            
            state.maxval['x0_bpl']=10
            state.maxval['dx_bpl']=1


            

            state.minval['x1']=15
            state.minval['A']=0
            state.minval['B']=-10
            state.minval['A2']=0
            state.minval['B2']=-10            
            state.minval['x0_bpl']=-1
            state.minval['dx_bpl']=0.01



            global frame_bpl#defining gui section to handle bpl param options
            frame_bpl=tk.Frame(master=frame_params)
            
            tk.Label(frame_bpl, text="Broken Power Law", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_A(): state.vary['A'] = not state.vary['A']
            def toggle_A2(): state.vary['A2'] = not state.vary['A2']
            def toggle_x1(): state.vary['x1'] = not state.vary['x1']
            def toggle_B(): state.vary['B'] = not state.vary['B']
            def toggle_B2(): state.vary['B2'] = not state.vary['B2']
            def toggle_x0_bpl(): state.vary['x0_bpl'] = not state.vary['x0_bpl']
            def toggle_dx_bpl(): state.vary['dx_bpl'] = not state.vary['dx_bpl']

            add_param_row(frame_bpl, 1, "x1", state.init['x1'], state.minval['x1'], state.maxval['x1'], state.vary['x1'], toggle_x1, "x1")
            add_param_row(frame_bpl, 2, "A", state.init['A'], state.minval['A'], state.maxval['A'], state.vary['A'], toggle_A, "A")           
            add_param_row(frame_bpl, 3, "B", state.init['B'], state.minval['B'], state.maxval['B'], state.vary['B'], toggle_B, "B")
            add_param_row(frame_bpl, 4, "A2", state.init['A2'], state.minval['A2'], state.maxval['A2'], state.vary['A2'], toggle_A2, "A2")
            add_param_row(frame_bpl, 5, "B2", state.init['B2'], state.minval['B2'], state.maxval['B2'], state.vary['B2'], toggle_B2, "B2")
            add_param_row(frame_bpl, 6, "x0_bpl", state.init['x0_bpl'], state.minval['x0_bpl'], state.maxval['x0_bpl'], state.vary['x0_bpl'], toggle_x0_bpl, "x0_bpl")
            add_param_row(frame_bpl, 7, "dx_bpl", state.init['dx_bpl'], state.minval['dx_bpl'], state.maxval['dx_bpl'], state.vary['dx_bpl'], toggle_dx_bpl, "dx_bpl")

            def hndl_remove_bpl_btn():
                frame_bpl.destroy()
                state.bpl_pres = 0

            tk.Button(frame_bpl, text='Remove BPL component', command=hndl_remove_bpl_btn)\
                .grid(row=8, column=0, columnspan=5, pady=10)

            frame_bpl.grid(row=3, column=0, sticky="ew")
            for i in range(8):
                frame_bpl.grid_columnconfigure(i, weight=1)
            state.bpl_pres = 1
    
    
    
    def add_gauss():#function to add gausian function to gui/test function
        if state.gauss_pres == 0:
            
            state.init['gauss_amp']=1e9
            state.init['gauss_centre']=0
            state.init['sigma']=1
            
            state.vary['gauss_amp']=True
            state.vary['gauss_centre']=True
            state.vary['sigma']=True
            
            state.minval['gauss_amp']=0
            state.minval['gauss_centre']=0
            state.minval['sigma']=0
            
            state.maxval['gauss_amp']=1e10
            state.maxval['gauss_centre']=150
            state.maxval['sigma']=3   
            
            global frame_gauss#defining gui section to handle gaussian param options
            frame_gauss=tk.Frame(master=frame_params)
            tk.Label(frame_gauss, text="Gaussian", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_gauss_amp(): state.vary['gauss_amp'] = not state.vary['gauss_amp']
            def toggle_gauss_centre(): state.vary['gauss_centre'] = not state.vary['gauss_centre']
            def toggle_sigma(): state.vary['sigma'] = not state.vary['sigma']

            add_param_row(frame_gauss, 1, "gauss_amp", state.init['gauss_amp'], state.minval['gauss_amp'], state.maxval['gauss_amp'], state.vary['gauss_amp'], toggle_gauss_amp, "gauss_amp")
            add_param_row(frame_gauss, 2, "gauss_centre", state.init['gauss_centre'], state.minval['gauss_centre'], state.maxval['gauss_centre'], state.vary['gauss_centre'], toggle_gauss_centre, "gauss_centre")
            add_param_row(frame_gauss, 3, "sigma", state.init['sigma'], state.minval['sigma'], state.maxval['sigma'], state.vary['sigma'], toggle_sigma, "sigma")

            def hndl_remove_gauss_btn():
                frame_gauss.destroy()
                state.gauss_pres = 0

            tk.Button(frame_gauss, text='Remove Gaussian component', command=hndl_remove_gauss_btn)\
                .grid(row=4, column=0, columnspan=5, pady=10)

            frame_gauss.grid(row=4, column=0, sticky="ew")
            for i in range(4):
                frame_gauss.grid_columnconfigure(i, weight=1)
            state.gauss_pres = 1
    
    def add_power():#function to add power law to gui/test function
        if state.power_pres == 0:
            state.init['A_sing']=1e9
            state.init['B_sing']=-1
            state.init['x0_sing']=1
            state.init['dx_sing']=0.1    

            
            state.vary['A_sing']=True
            state.vary['B_sing']=True
            state.vary['x0_sing']=True
            state.vary['dx_sing']=True
            
            state.minval['A_sing']=0
            state.minval['B_sing']=-10
            state.minval['x0_sing']=-1
            state.minval['dx_sing']=0.01

            
            state.maxval['A_sing']=1e10
            state.maxval['B_sing']=0
            state.maxval['x0_sing']=10
            state.maxval['dx_sing']=1

            
            global frame_power
            frame_power=tk.Frame(master=frame_params)
            tk.Label(frame_power, text="Power Law", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_A_sing(): state.vary['A_sing'] = not state.vary['A_sing']
            def toggle_B_sing(): state.vary['B_sing'] = not state.vary['B_sing']
            def toggle_x0_sing(): state.vary['x0_sing'] = not state.vary['x0_sing']
            def toggle_dx_sing(): state.vary['dx_sing'] = not state.vary['dx_sing']

            add_param_row(frame_power, 1, "A_sing", state.init['A_sing'], state.minval['A_sing'], state.maxval['A_sing'], state.vary['A_sing'], toggle_A_sing, "A_sing")
            add_param_row(frame_power, 2, "B_sing", state.init['B_sing'], state.minval['B_sing'], state.maxval['B_sing'], state.vary['B_sing'], toggle_B_sing, "B_sing")
            add_param_row(frame_power, 3, "x0_sing", state.init['x0_sing'], state.minval['x0_sing'], state.maxval['x0_sing'], state.vary['x0_sing'], toggle_x0_sing, "x0_sing")
            add_param_row(frame_power, 4, "dx_sing", state.init['dx_sing'], state.minval['dx_sing'], state.maxval['dx_sing'], state.vary['dx_sing'], toggle_dx_sing, "dx_sing")

            def hndl_remove_power_btn():
                frame_power.destroy()
                state.power_pres = 0

            tk.Button(frame_power, text='Remove Power Law component', command=hndl_remove_power_btn)\
                .grid(row=5, column=0, columnspan=5, pady=10)

            frame_power.grid(row=5, column=0, sticky="ew")
            for i in range(6):
                frame_power.grid_columnconfigure(i, weight=1)
            state.power_pres = 1
            
    def add_kappa():#function to add kappa law to gui/test function
        
        if state.kappa_pres == 0:
            
            
            state.init['A_k']=10**-20
            state.init['T_k']=300000000.0
            state.init['m_i']=9.11*1e-31
            state.init['n_i']=1e15
            state.init['kappa']=50
            
            
            state.vary['A_k']=True
            state.vary['T_k']=True
            state.vary['m_i']=False
            state.vary['n_i']=True
            state.vary['kappa']=True
            
            
            state.minval['A_k']=1e-22
            state.minval['T_k']=1e6
            state.minval['m_i']=0
            state.minval['n_i']=0
            state.minval['kappa']=(3/2)+0.0001#must be greater than 3/2
            
            
            state.maxval['A_k']=1
            state.maxval['T_k']=1e10
            state.maxval['m_i']=1
            state.maxval['n_i']=1e10
            state.maxval['kappa']=1000
            
            global frame_kappa
            frame_kappa=tk.Frame(master=frame_params)
            tk.Label(frame_kappa, text="Kappa Function", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_A_k(): state.vary['A_k'] = not state.vary['A_k']
            def toggle_T_k(): state.vary['T_k'] = not state.vary['T_k']
            def toggle_m_i(): state.vary['m_i'] = not state.vary['m_i']
            def toggle_n_i(): state.vary['n_i'] = not state.vary['n_i']
            def toggle_kappa(): state.vary['kappa'] = not state.vary['kappa']

            add_param_row(frame_kappa, 1, "A_k", state.init['A_k'], state.minval['A_k'], state.maxval['A_k'], state.vary['A_k'], toggle_A_k, "A_k")
            add_param_row(frame_kappa, 2, "T_k", state.init['T_k'], state.minval['T_k'], state.maxval['T_k'], state.vary['T_k'], toggle_T_k, "T_k")
            add_param_row(frame_kappa, 3, "m_i", state.init['m_i'], state.minval['m_i'], state.maxval['m_i'], state.vary['m_i'], toggle_m_i, "m_i")
            add_param_row(frame_kappa, 4, "n_i", state.init['n_i'], state.minval['n_i'], state.maxval['n_i'], state.vary['n_i'], toggle_n_i, "n_i")
            add_param_row(frame_kappa, 5, "kappa", state.init['kappa'], state.minval['kappa'], state.maxval['kappa'], state.vary['kappa'], toggle_kappa, "kappa")

            def hndl_remove_kappa_btn():
                frame_kappa.destroy()
                state.kappa_pres = 0

            tk.Button(frame_kappa, text='Remove Kappa component', command=hndl_remove_kappa_btn)\
                .grid(row=6, column=0, columnspan=5, pady=10)

            frame_kappa.grid(row=6, column=0, sticky="ew")
            for i in range(7):
                frame_kappa.grid_columnconfigure(i, weight=1)
            state.kappa_pres = 1
            
        
    def add_bpl_and_therm():
        if state.bpl_and_therm_pres == 0:
            state.init['amp_c']=1e9
            state.init['T_c']=12e6
            state.init['alpha_c']=1
            state.init['x0_c']=20
            state.init['x1_c']=50
            state.init['B_c']=-1 
            state.init['B2_c']=-2    
            
            state.vary['amp_c']=True
            state.vary['T_c']=True
            state.vary['alpha_c']=False
            state.vary['x0_c']=True
            state.vary['x1_c']=True
            state.vary['B_c']=True
            state.vary['B2_c']=True         
            
            state.minval['amp_c']=0
            state.minval['T_c']=0
            state.minval['alpha_c']=0
            state.minval['x0_c']=13
            state.minval['x1_c']=40
            state.minval['B_c']=-10
            state.minval['B2_c']=-10
         
            
            state.maxval['amp_c']=1e10
            state.maxval['T_c']=1e8
            state.maxval['alpha_c']=5    
            state.maxval['x0_c']=25
            state.maxval['x1_c']=55
            state.maxval['B_c']=-0.1
            state.maxval['B2_c']=-0.1   

            
            global frame_bpl_and_therm#defining gui section to handle bpl param options
            frame_bpl_and_therm=tk.Frame(master=frame_params)
            
            tk.Label(frame_bpl_and_therm, text="BPL + Thermal", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_amp_c(): state.vary['amp_c'] = not state.vary['amp_c']
            def toggle_T_c(): state.vary['T_c'] = not state.vary['T_c']
            def toggle_alpha_c(): state.vary['alpha_c'] = not state.vary['alpha_c']
            def toggle_x1_c(): state.vary['x1_c'] = not state.vary['x1_c']
            def toggle_x0_c(): state.vary['x0_c'] = not state.vary['x0_c']
            def toggle_B_c(): state.vary['B_c'] = not state.vary['B_c']
            def toggle_B2_c(): state.vary['B2_c'] = not state.vary['B2_c']

            add_param_row(frame_bpl_and_therm, 1, "amp_c", state.init['amp_c'], state.minval['amp_c'], state.maxval['amp_c'], state.vary['amp_c'], toggle_amp_c, "amp_c")
            add_param_row(frame_bpl_and_therm, 2, "T_c", state.init['T_c'], state.minval['T_c'], state.maxval['T_c'], state.vary['T_c'], toggle_T_c, "T_c")
            add_param_row(frame_bpl_and_therm, 3, "alpha_c", state.init['alpha_c'], state.minval['alpha_c'], state.maxval['alpha_c'], state.vary['alpha_c'], toggle_alpha_c, "alpha_c")
            add_param_row(frame_bpl_and_therm, 4, "x0_c", state.init['x0_c'], state.minval['x0_c'], state.maxval['x0_c'], state.vary['x0_c'], toggle_x0_c, "x0_c")
            add_param_row(frame_bpl_and_therm, 5, "x1_c", state.init['x1_c'], state.minval['x1_c'], state.maxval['x1_c'], state.vary['x1_c'], toggle_x1_c, "x1_c")
            add_param_row(frame_bpl_and_therm, 6, "B_c", state.init['B_c'], state.minval['B_c'], state.maxval['B_c'], state.vary['B_c'], toggle_B_c, "B_c")
            add_param_row(frame_bpl_and_therm, 7, "B2_c", state.init['B2_c'], state.minval['B2_c'], state.maxval['B2_c'], state.vary['B2_c'], toggle_B2_c, "B2_c")

            def hndl_remove_bt_btn():
                frame_bpl_and_therm.destroy()
                state.bpl_and_therm_pres = 0

            tk.Button(frame_bpl_and_therm, text='Remove BPL + Thermal component', command=hndl_remove_bt_btn)\
                .grid(row=8, column=0, columnspan=5, pady=10)

            frame_bpl_and_therm.grid(row=7, column=0, sticky="ew")
            for i in range(9):
                frame_bpl_and_therm.grid_columnconfigure(i, weight=1)
            state.bpl_and_therm_pres = 1
    
    def add_double_therm():#add the thermal component to the fitted function
        
        if state.double_therm_func_pres == 0:#if double thermal function not already there
            
            state.init['amp_d_1']=1e10
            state.init['T_d_1']=3e6
            state.init['alpha_d_1']=1
            state.init['amp_d_2']=1e8
            state.init['T_d_2']=16e6
            state.init['alpha_d_2']=1
            
            state.vary['amp_d_1']=True
            state.vary['T_d_1']=True
            state.vary['alpha_d_1']=False
            state.vary['amp_d_2']=True
            state.vary['T_d_2']=True
            state.vary['alpha_d_2']=False
            
            state.minval['amp_d_1']=0
            state.minval['T_d_1']=0
            state.minval['alpha_d_1']=0
            state.minval['amp_d_2']=0
            state.minval['T_d_2']=0
            state.minval['alpha_d_2']=0
            
            
            state.maxval['amp_d_1']=1e10
            state.maxval['T_d_1']=1e8
            state.maxval['alpha_d_1']=5    
            state.maxval['amp_d_2']=1e10
            state.maxval['T_d_2']=1e8
            state.maxval['alpha_d_2']=5   
            
            
            #defining the part of the GUI window that contains the options for the thermal curve
            global frame_double_therm 
            frame_double_therm=tk.Frame(master=frame_params)
            tk.Label(frame_double_therm, text="Double Thermal", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_amp_d_1(): state.vary['amp_d_1'] = not state.vary['amp_d_1']
            def toggle_T_d_1(): state.vary['T_d_1'] = not state.vary['T_d_1']
            def toggle_alpha_d_1(): state.vary['alpha_d_1'] = not state.vary['alpha_d_1']
            def toggle_amp_d_2(): state.vary['amp_d_2'] = not state.vary['amp_d_2']
            def toggle_T_d_2(): state.vary['T_d_2'] = not state.vary['T_d_2']
            def toggle_alpha_d_2(): state.vary['alpha_d_2'] = not state.vary['alpha_d_2']

            add_param_row(frame_double_therm, 1, "amp_d_1", state.init['amp_d_1'], state.minval['amp_d_1'], state.maxval['amp_d_1'], state.vary['amp_d_1'], toggle_amp_d_1, "amp_d_1")
            add_param_row(frame_double_therm, 2, "T_d_1", state.init['T_d_1'], state.minval['T_d_1'], state.maxval['T_d_1'], state.vary['T_d_1'], toggle_T_d_1, "T_d_1")
            add_param_row(frame_double_therm, 3, "alpha_d_1", state.init['alpha_d_1'], state.minval['alpha_d_1'], state.maxval['alpha_d_1'], state.vary['alpha_d_1'], toggle_alpha_d_1, "alpha_d_1")
            add_param_row(frame_double_therm, 4, "amp_d_2", state.init['amp_d_2'], state.minval['amp_d_2'], state.maxval['amp_d_2'], state.vary['amp_d_2'], toggle_amp_d_2, "amp_d_2")
            add_param_row(frame_double_therm, 5, "T_d_2", state.init['T_d_2'], state.minval['T_d_2'], state.maxval['T_d_2'], state.vary['T_d_2'], toggle_T_d_2, "T_d_2")
            add_param_row(frame_double_therm, 6, "alpha_d_2", state.init['alpha_d_2'], state.minval['alpha_d_2'], state.maxval['alpha_d_2'], state.vary['alpha_d_2'], toggle_alpha_d_2, "alpha_d_2")

            def hndl_remove_double_therm_btn():
                frame_double_therm.destroy()
                state.double_therm_func_pres = 0

            tk.Button(frame_double_therm, text='Remove Double Thermal component', command=hndl_remove_double_therm_btn)\
                .grid(row=7, column=0, columnspan=5, pady=10)

            frame_double_therm.grid(row=8, column=0, sticky="ew")
            for i in range(8):
                frame_double_therm.grid_columnconfigure(i, weight=1)
            state.double_therm_func_pres = 1 #set the thermal function as present
    
    
    
    def add_tpl():#function to add the triple power law
        if state.tpl_pres == 0:
    
            
            state.init['x1_tpl']=11
            state.init['x2_tpl']=40
            state.init['A_tpl']=1e5
            state.init['B_tpl']=-2
            state.init['A2_tpl']=1e5
            state.init['B2_tpl']=-1    
            state.init['A3_tpl']=1e5
            state.init['B3_tpl']=-2   
            state.init['x0_tpl']=1
            state.init['dx_tpl']=0.1  


            state.vary['x1_tpl']=True
            state.vary['x2_tpl']=True
            state.vary['A_tpl']=True
            state.vary['B_tpl']=True
            state.vary['A2_tpl']=True
            state.vary['B2_tpl']=True            
            state.vary['A3_tpl']=True
            state.vary['B3_tpl']=True            
            state.vary['x0_tpl']=True
            state.vary['dx_tpl']=True
            

            state.maxval['x1_tpl']=50
            state.maxval['x2_tpl']=50
            state.maxval['A_tpl']=1e10
            state.maxval['B_tpl']=0
            state.maxval['A2_tpl']=1e10
            state.maxval['B2_tpl']=0            
            state.maxval['A3_tpl']=1e10
            state.maxval['B3_tpl']=0           
            state.maxval['x0_tpl']=10
            state.maxval['dx_tpl']=1


            

            state.minval['x1_tpl']=5
            state.minval['x2_tpl']=15
            state.minval['A_tpl']=0
            state.minval['B_tpl']=-10
            state.minval['A2_tpl']=0
            state.minval['B2_tpl']=-10            
            state.minval['A3_tpl']=0
            state.minval['B3_tpl']=-10            
            state.minval['x0_tpl']=-1
            state.minval['dx_tpl']=0.01


            global frame_tpl#defining gui section to handle tpl param options
            frame_tpl=tk.Frame(master=frame_params)
            tk.Label(frame_tpl, text="Triple Power Law", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))
                

            def toggle_A_tpl(): state.vary['A_tpl'] = not state.vary['A_tpl']
            def toggle_B_tpl(): state.vary['B_tpl'] = not state.vary['B_tpl']
            
            def toggle_x1_tpl(): state.vary['x1_tpl'] = not state.vary['x1_tpl']
            def toggle_A2_tpl(): state.vary['A2_tpl'] = not state.vary['A2_tpl']
            def toggle_B2_tpl(): state.vary['B2_tpl'] = not state.vary['B2_tpl']
            
            def toggle_x2_tpl(): state.vary['x2_tpl'] = not state.vary['x2_tpl']
            def toggle_A3_tpl(): state.vary['A3_tpl'] = not state.vary['A3_tpl']
            def toggle_B3_tpl(): state.vary['B3_tpl'] = not state.vary['B3_tpl']
            
            def toggle_x0_tpl(): state.vary['x0_tpl'] = not state.vary['x0_tpl']
            def toggle_dx_tpl(): state.vary['dx_tpl'] = not state.vary['dx_tpl']


            add_param_row(frame_tpl, 1, "A_tpl", state.init['A_tpl'], state.minval['A_tpl'], state.maxval['A_tpl'], state.vary['A_tpl'], toggle_A_tpl, "A_tpl")
            add_param_row(frame_tpl, 2, "B_tpl", state.init['B_tpl'], state.minval['B_tpl'], state.maxval['B_tpl'], state.vary['B_tpl'], toggle_B_tpl, "B_tpl")
            add_param_row(frame_tpl, 3, "x1_tpl", state.init['x1_tpl'], state.minval['x1_tpl'], state.maxval['x1_tpl'], state.vary['x1_tpl'], toggle_x1_tpl, "x1_tpl")
            add_param_row(frame_tpl, 4, "A2_tpl", state.init['A2_tpl'], state.minval['A2_tpl'], state.maxval['A2_tpl'], state.vary['A2_tpl'], toggle_A2_tpl, "A2_tpl")
            add_param_row(frame_tpl, 5, "B2_tpl", state.init['B2_tpl'], state.minval['B2_tpl'], state.maxval['B2_tpl'], state.vary['B2_tpl'], toggle_B2_tpl, "B2_tpl")
            add_param_row(frame_tpl, 6, "x2_tpl", state.init['x2_tpl'], state.minval['x2_tpl'], state.maxval['x2_tpl'], state.vary['x2_tpl'], toggle_x2_tpl, "x2_tpl")
            add_param_row(frame_tpl, 7, "A3_tpl", state.init['A3_tpl'], state.minval['A3_tpl'], state.maxval['A3_tpl'], state.vary['A3_tpl'], toggle_A3_tpl, "A3_tpl")
            add_param_row(frame_tpl, 8, "B3_tpl", state.init['B3_tpl'], state.minval['B3_tpl'], state.maxval['B3_tpl'], state.vary['B3_tpl'], toggle_B3_tpl, "B3_tpl")
            add_param_row(frame_tpl, 9, "x0_tpl", state.init['x0_tpl'], state.minval['x0_tpl'], state.maxval['x0_tpl'], state.vary['x0_tpl'], toggle_x0_tpl, "x0_tpl")
            add_param_row(frame_tpl, 10, "dx_tpl", state.init['dx_tpl'], state.minval['dx_tpl'], state.maxval['dx_tpl'], state.vary['dx_tpl'], toggle_dx_tpl, "dx_tpl")

            

            def hndl_remove_tpl_btn():
                frame_tpl.destroy()
                state.tpl_pres = 0

            tk.Button(frame_tpl, text='Remove TPL component', command=hndl_remove_tpl_btn)\
                .grid(row=11, column=0, columnspan=5, pady=10)

            frame_tpl.grid(row=9, column=0, sticky="ew")
            for i in range(12):
                frame_tpl.grid_columnconfigure(i, weight=1)
            state.tpl_pres = 1
    
    
    def add_qpl():#function to add the quad power law
        if state.qpl_pres == 0:
    
            
            state.init['x1_qpl']=5
            state.init['x2_qpl']=11
            state.init['x3_qpl']=40
            state.init['A_qpl']=1e8
            state.init['B_qpl']=-1
            state.init['A2_qpl']=1e9
            state.init['B2_qpl']=-2    
            state.init['A3_qpl']=1.5e8
            state.init['B3_qpl']=-1   
            state.init['A4_qpl']=1e9
            state.init['B4_qpl']=-2 
            state.init['x0_qpl']=1
            state.init['dx_qpl']=0.1    

    

            state.vary['x1_qpl']=True
            state.vary['x2_qpl']=True
            state.vary['x3_qpl']=True
            state.vary['A_qpl']=True
            state.vary['B_qpl']=True
            state.vary['A2_qpl']=True
            state.vary['B2_qpl']=True            
            state.vary['A3_qpl']=True
            state.vary['B3_qpl']=True
            state.vary['A4_qpl']=True
            state.vary['B4_qpl']=True            
            state.vary['x0_qpl']=True
            state.vary['dx_qpl']=True
            

            state.maxval['x1_qpl']=10
            state.maxval['x2_qpl']=20
            state.maxval['x3_qpl']=50
            state.maxval['A_qpl']=1e10
            state.maxval['B_qpl']=0
            state.maxval['A2_qpl']=1e10
            state.maxval['B2_qpl']=0            
            state.maxval['A3_qpl']=1e10
            state.maxval['B3_qpl']=0           
            state.maxval['A4_qpl']=1e10
            state.maxval['B4_qpl']=0           
            state.maxval['x0_qpl']=10
            state.maxval['dx_qpl']=1

            

            state.minval['x1_qpl']=0
            state.minval['x2_qpl']=5
            state.minval['x3_qpl']=30
            state.minval['A_qpl']=0
            state.minval['B_qpl']=-10
            state.minval['A2_qpl']=0
            state.minval['B2_qpl']=-10            
            state.minval['A3_qpl']=0
            state.minval['B3_qpl']=-10
            state.minval['A4_qpl']=0
            state.minval['B4_qpl']=-10            
            state.minval['x0_qpl']=-1.1
            state.minval['dx_qpl']=0.01

            
            global frame_qpl#defining gui section to handle qpl param options
            frame_qpl=tk.Frame(master=frame_params)
            
            tk.Label(frame_qpl, text="Quadruple Power Law", font=("Arial", 12, "bold"))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_A_qpl(): state.vary['A_qpl'] = not state.vary['A_qpl']
            def toggle_B_qpl(): state.vary['B_qpl'] = not state.vary['B_qpl']
            
            def toggle_x1_qpl(): state.vary['x1_qpl'] = not state.vary['x1_qpl']
            def toggle_A2_qpl(): state.vary['A2_qpl'] = not state.vary['A2_qpl']
            def toggle_B2_qpl(): state.vary['B2_qpl'] = not state.vary['B2_qpl']
            
            def toggle_x2_qpl(): state.vary['x2_qpl'] = not state.vary['x2_qpl']
            def toggle_A3_qpl(): state.vary['A3_qpl'] = not state.vary['A3_qpl']
            def toggle_B3_qpl(): state.vary['B3_qpl'] = not state.vary['B3_qpl']
            
            def toggle_x3_qpl(): state.vary['x3_qpl'] = not state.vary['x3_qpl']
            def toggle_A4_qpl(): state.vary['A4_qpl'] = not state.vary['A4_qpl']
            def toggle_B4_qpl(): state.vary['B4_qpl'] = not state.vary['B4_qpl']
            def toggle_x0_qpl(): state.vary['x0_qpl'] = not state.vary['x0_qpl']
            def toggle_dx_qpl(): state.vary['dx_qpl'] = not state.vary['dx_qpl']            

            


            add_param_row(frame_qpl, 1, "A_qpl", state.init['A_qpl'], state.minval['A_qpl'], state.maxval['A_qpl'], state.vary['A_qpl'], toggle_A_qpl, "A_qpl")
            add_param_row(frame_qpl, 2, "B_qpl", state.init['B_qpl'], state.minval['B_qpl'], state.maxval['B_qpl'], state.vary['B_qpl'], toggle_B_qpl, "B_qpl")
            
            add_param_row(frame_qpl, 3, "x1_qpl", state.init['x1_qpl'], state.minval['x1_qpl'], state.maxval['x1_qpl'], state.vary['x1_qpl'], toggle_x1_qpl, "x1_qpl")
            add_param_row(frame_qpl, 4, "A2_qpl", state.init['A2_qpl'], state.minval['A2_qpl'], state.maxval['A2_qpl'], state.vary['A2_qpl'], toggle_A2_qpl, "A2_qpl")
            add_param_row(frame_qpl, 5, "B2_qpl", state.init['B2_qpl'], state.minval['B2_qpl'], state.maxval['B2_qpl'], state.vary['B2_qpl'], toggle_B2_qpl, "B2_qpl")
            
            add_param_row(frame_qpl, 6, "x2_qpl", state.init['x2_qpl'], state.minval['x2_qpl'], state.maxval['x2_qpl'], state.vary['x2_qpl'], toggle_x2_qpl, "x2_qpl")
            add_param_row(frame_qpl, 7, "A3_qpl", state.init['A3_qpl'], state.minval['A3_qpl'], state.maxval['A3_qpl'], state.vary['A3_qpl'], toggle_A3_qpl, "A3_qpl")
            add_param_row(frame_qpl, 8, "B3_qpl", state.init['B3_qpl'], state.minval['B3_qpl'], state.maxval['B3_qpl'], state.vary['B3_qpl'], toggle_B3_qpl, "B3_qpl")
            
            add_param_row(frame_qpl, 9, "x3_qpl", state.init['x3_qpl'], state.minval['x3_qpl'], state.maxval['x3_qpl'], state.vary['x3_qpl'], toggle_x3_qpl, "x3_qpl")
            add_param_row(frame_qpl, 10, "A4_qpl", state.init['A4_qpl'], state.minval['A4_qpl'], state.maxval['A4_qpl'], state.vary['A4_qpl'], toggle_A4_qpl, "A4_qpl")
            add_param_row(frame_qpl, 11, "B4_qpl", state.init['B4_qpl'], state.minval['B4_qpl'], state.maxval['B4_qpl'], state.vary['B4_qpl'], toggle_B4_qpl, "B4_qpl")
            
            add_param_row(frame_qpl, 12, "x0_qpl", state.init['x0_qpl'], state.minval['x0_qpl'], state.maxval['x0_qpl'], state.vary['x0_qpl'], toggle_x0_qpl, "x0_qpl")
            add_param_row(frame_qpl, 13, "dx_qpl", state.init['dx_qpl'], state.minval['dx_qpl'], state.maxval['dx_qpl'], state.vary['dx_qpl'], toggle_dx_qpl, "dx_qpl")



            def hndl_remove_qpl_btn():
                frame_qpl.destroy()
                state.qpl_pres = 0

            tk.Button(frame_qpl, text='Remove QPL component', command=hndl_remove_qpl_btn)\
                .grid(row=14, column=0, columnspan=5, pady=10)

            frame_qpl.grid(row=10, column=0, sticky="ew")
            for i in range(15):
                frame_qpl.grid_columnconfigure(i, weight=1)
            state.qpl_pres = 1
    
    def add_quint_pl():#function to add the quad power law
        if state.quint_pl_pres == 0:
    
            
            state.init['x1_5pl']=2
            state.init['x2_5pl']=5
            state.init['x3_5pl']=11
            state.init['x4_5pl']=40
            state.init['A_5pl']=1e8
            state.init['B_5pl']=-1
            state.init['A2_5pl']=1e9
            state.init['B2_5pl']=-2    
            state.init['A3_5pl']=1.5e8
            state.init['B3_5pl']=-1   
            state.init['A4_5pl']=1e9
            state.init['B4_5pl']=-2 
            state.init['A5_5pl']=1e9
            state.init['B5_5pl']=-2 
            state.init['x0_5pl']=1
            state.init['dx_5pl']=0.1    

    

            state.vary['x1_5pl']=True
            state.vary['x2_5pl']=True
            state.vary['x3_5pl']=True
            state.vary['x4_5pl']=True
            state.vary['A_5pl']=True
            state.vary['B_5pl']=True
            state.vary['A2_5pl']=True
            state.vary['B2_5pl']=True            
            state.vary['A3_5pl']=True
            state.vary['B3_5pl']=True
            state.vary['A4_5pl']=True
            state.vary['B4_5pl']=True
            state.vary['A5_5pl']=True
            state.vary['B5_5pl']=True            
            state.vary['x0_5pl']=True
            state.vary['dx_5pl']=True
            

            state.maxval['x1_5pl']=10
            state.maxval['x2_5pl']=20
            state.maxval['x3_5pl']=50
            state.maxval['x4_5pl']=50
            state.maxval['A_5pl']=1e10
            state.maxval['B_5pl']=0
            state.maxval['A2_5pl']=1e10
            state.maxval['B2_5pl']=0            
            state.maxval['A3_5pl']=1e10
            state.maxval['B3_5pl']=0           
            state.maxval['A4_5pl']=1e10
            state.maxval['B4_5pl']=0     
            state.maxval['A5_5pl']=1e10
            state.maxval['B5_5pl']=0 
            state.maxval['x0_5pl']=10
            state.maxval['dx_5pl']=1

            

            state.minval['x1_5pl']=0
            state.minval['x2_5pl']=5
            state.minval['x3_5pl']=10
            state.minval['x4_5pl']=30
            state.minval['A_5pl']=0
            state.minval['B_5pl']=-10
            state.minval['A2_5pl']=0
            state.minval['B2_5pl']=-10            
            state.minval['A3_5pl']=0
            state.minval['B3_5pl']=-10
            state.minval['A4_5pl']=0
            state.minval['B4_5pl']=-10
            state.minval['A5_5pl']=0
            state.minval['B5_5pl']=-10            
            state.minval['x0_5pl']=-1
            state.minval['dx_5pl']=0.01


            global frame_quint_pl#defining gui section to handle quint_pl param options
            frame_quint_pl=tk.Frame(master=frame_params)
            
            tk.Label(frame_quint_pl, text="Quintuple Power Law", font=("Arial", 12, "bold"))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle(var):
                state.vary[var] = not state.vary[var]

            row_counter = 1
            for name in ["A_5pl", "B_5pl","x1_5pl","A2_5pl", "B2_5pl","x2_5pl", "A3_5pl","B3_5pl","x3_5pl","A4_5pl", "B4_5pl", "x4_5pl","A5_5pl", "B5_5pl",'x0_5pl','dx_5pl']:
                add_param_row(
                    frame_quint_pl,
                    row_counter,
                    name,
                    state.init[name],
                    state.minval[name],
                    state.maxval[name],
                    state.vary[name],
                    lambda n=name: toggle(n),
                    name
                )
                row_counter += 1

            def hndl_remove_quint_btn():
                frame_quint_pl.destroy()
                state.quint_pl_pres = 0

            tk.Button(frame_quint_pl, text='Remove Quintuple Power Law component', command=hndl_remove_quint_btn)\
                .grid(row=row_counter, column=0, columnspan=5, pady=10)

            frame_quint_pl.grid(row=11, column=0, sticky="ew")
            for i in range(row_counter+1):
                frame_quint_pl.grid_columnconfigure(i, weight=1)
            state.quint_pl_pres = 1
    
    

    
    
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
            
            state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
            
        elif selected_func == 'broken power law':
            add_bpl()
            
            state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
            
        elif selected_func == 'Gaussian':
            add_gauss()
            
            state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
            
        elif selected_func == 'power law':
            add_power()
            
            state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
            
        elif selected_func == 'kappa function':
            add_kappa()
            
            state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
            
        elif selected_func == 'broken power law + thermal':
            add_bpl_and_therm()
            
            state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
            
        elif selected_func == 'double thermal':
            add_double_therm()
            
            state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
            
        elif selected_func == 'triple power law':
            add_tpl()
            
            state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
            
        elif selected_func == "quadruple power law":
            add_qpl()
            
            state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
            
        elif selected_func == "quintuple power law":
            add_quint_pl()
            
            state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
            
 
    

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
    
    def validate_minmaxval(min_val,max_val): #max must be float greater than state.minval or none
        if (type(min_val)==float and type(max_val)==float and max_val>min_val) or (min_val==None and type(max_val)==float) or (type(min_val)==float and max_val==None) or (min_val==None and max_val==None):
            return True
        return False
    
    def validate_finite(min_val,max_val): #max must be float greater than state.minval or none
        if (type(min_val)==float and type(max_val)==float):
            return True
        return False

    def validate_init(init_val,min_val,max_val):#must be float between max and min val
        if type(state.init)==float and init_val>min_val and  max_val>init_val:
            return True
        return False
    def validate_lims(min_val,max_val):
        if (type(min_val)==float and type(max_val)==float and max_val>min_val):
            return True
        return False
        
    def fit_btn_hndl():#function to handle button to perform fit
        if state.fit_window is not None:
            #close any open figues
            try: state.fit_window.destroy()
            except tk.TclError: pass
            state.fit_window=None
            
        if state.preview_window is not None:
            #close any open figues
            try: state.preview_window.destroy()
            except tk.TclError: pass
            state.preview_window=None
            
        if state.resid_window is not None:
            #close any open figues
            try: state.resid_window.destroy()
            except tk.TclError: pass
            state.resid_window=None
        
        state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
        try:#try excpet statement is to validate inputs as integers
            if state.therm_func_pres==1:#if thermal function present, save parameter options from the gui for that function
                
                global frame_therm
                
                state.init['T']=None if state.entries["init_T_entry"].get()=='None' else float(state.entries["init_T_entry"].get())
                state.minval['T']=None if state.entries["minval_T_entry"].get()=='None' else float(state.entries["minval_T_entry"].get())
                state.maxval['T']=None if state.entries["maxval_T_entry"].get()=='None' else float(state.entries["maxval_T_entry"].get())
                
                state.init['amp']=None if state.entries["init_amp_entry"].get()=='None' else float(state.entries["init_amp_entry"].get())
                state.minval['amp']=None if state.entries["minval_amp_entry"].get()=='None' else float(state.entries["minval_amp_entry"].get())
                state.maxval['amp']=None if state.entries["maxval_amp_entry"].get()=='None' else float(state.entries["maxval_amp_entry"].get())
                
                state.init['alpha']=None if state.entries["init_alpha_entry"].get()=='None' else float(state.entries["init_alpha_entry"].get())
                state.minval['alpha']=None if state.entries["minval_alpha_entry"].get()=='None' else float(state.entries["minval_alpha_entry"].get())
                state.maxval['alpha']=None if state.entries["maxval_alpha_entry"].get()=='None' else float(state.entries["maxval_alpha_entry"].get())
                
            if state.bpl_pres==1:#if bpl function present, save parameter options from the gui for that function
                
                
                
                state.init['x1']=None if state.entries["init_x1_entry"].get()=='None' else float(state.entries["init_x1_entry"].get())
                state.minval['x1']=None if state.entries["minval_x1_entry"].get()=='None' else float(state.entries["minval_x1_entry"].get())
                state.maxval['x1']=None if state.entries["maxval_x1_entry"].get()=='None' else float(state.entries["maxval_x1_entry"].get())
                
                
                state.init['A2']=None if state.entries["init_A2_entry"].get()=='None' else float(state.entries["init_A2_entry"].get())
                state.minval['A2']=None if state.entries["minval_A2_entry"].get()=='None' else float(state.entries["minval_A2_entry"].get())
                state.maxval['A2']=None if state.entries["maxval_A2_entry"].get()=='None' else float(state.entries["maxval_A2_entry"].get())
                
                
                state.init['B2']=None if state.entries["init_B2_entry"].get()=='None' else float(state.entries["init_B2_entry"].get())
                state.minval['B2']=None if state.entries["minval_B2_entry"].get()=='None' else float(state.entries["minval_B2_entry"].get())
                state.maxval['B2']=None if state.entries["maxval_B2_entry"].get()=='None' else float(state.entries["maxval_B2_entry"].get())
                
                
                state.init['B']=None if state.entries["init_B_entry"].get()=='None' else float(state.entries["init_B_entry"].get())
                state.minval['B']=None if state.entries["minval_B_entry"].get()=='None' else float(state.entries["minval_B_entry"].get())
                state.maxval['B']=None if state.entries["maxval_B_entry"].get()=='None' else float( state.entries["maxval_B_entry"].get())
    
    
                state.init['A']=None if state.entries["init_A_entry"].get()=='None' else float(state.entries["init_A_entry"].get())
                state.minval['A']=None if state.entries["minval_A_entry"].get()=='None' else float(state.entries["minval_A_entry"].get())
                state.maxval['A']=None if state.entries["maxval_A_entry"].get()=='None' else float(state.entries["maxval_A_entry"].get())
                
                
                state.init['x0_bpl']=None if state.entries["init_x0_bpl_entry"].get()=='None' else float(state.entries["init_x0_bpl_entry"].get())
                state.minval['x0_bpl']=None if state.entries["minval_x0_bpl_entry"].get()=='None' else float(state.entries["minval_x0_bpl_entry"].get())
                state.maxval['x0_bpl']=None if state.entries["maxval_x0_bpl_entry"].get()=='None' else float( state.entries["maxval_x0_bpl_entry"].get())
    
    
                state.init['dx_bpl']=None if state.entries["init_dx_bpl_entry"].get()=='None' else float(state.entries["init_dx_bpl_entry"].get())
                state.minval['dx_bpl']=None if state.entries["minval_dx_bpl_entry"].get()=='None' else float(state.entries["minval_dx_bpl_entry"].get())
                state.maxval['dx_bpl']=None if state.entries["maxval_dx_bpl_entry"].get()=='None' else float(state.entries["maxval_dx_bpl_entry"].get())
                
            if state.gauss_pres==1:#if gaussian function present, save parameter options from the gui for that function
                state.init['gauss_centre']=None if state.entries["init_gauss_centre_entry"].get()=='None' else float(state.entries["init_gauss_centre_entry"].get())
                state.minval['gauss_centre']=None if state.entries["minval_gauss_centre_entry"].get()=='None' else float(state.entries["minval_gauss_centre_entry"].get())
                state.maxval['gauss_centre']=None if state.entries["maxval_gauss_centre_entry"].get()=='None' else float(state.entries["maxval_gauss_centre_entry"].get())
                
                
                state.init['gauss_amp']=None if state.entries["init_gauss_amp_entry"].get()=='None' else float(state.entries["init_gauss_amp_entry"].get())
                state.minval['gauss_amp']=None if state.entries["minval_gauss_amp_entry"].get()=='None' else float(state.entries["minval_gauss_amp_entry"].get())
                state.maxval['gauss_amp']=None if state.entries["maxval_gauss_amp_entry"].get()=='None' else float( state.entries["maxval_gauss_amp_entry"].get())
    
    
                state.init['sigma']=None if state.entries["init_sigma_entry"].get()=='None' else float(state.entries["init_sigma_entry"].get())
                state.minval['sigma']=None if state.entries["minval_sigma_entry"].get()=='None' else float(state.entries["minval_sigma_entry"].get())
                state.maxval['sigma']=None if state.entries["maxval_sigma_entry"].get()=='None' else float(state.entries["maxval_sigma_entry"].get())
               
               
               
               
               
            if state.power_pres==1:#if single power law function present, save parameter options from the gui for that function
                state.init['B_sing']=None if state.entries["init_B_sing_entry"].get()=='None' else float(state.entries["init_B_sing_entry"].get())
                state.minval['B_sing']=None if state.entries["minval_B_sing_entry"].get()=='None' else float(state.entries["minval_B_sing_entry"].get())
                state.maxval['B_sing']=None if state.entries["maxval_B_sing_entry"].get()=='None' else float( state.entries["maxval_B_sing_entry"].get())
    
    
                state.init['A_sing']=None if state.entries["init_A_sing_entry"].get()=='None' else float(state.entries["init_A_sing_entry"].get())
                state.minval['A_sing']=None if state.entries["minval_A_sing_entry"].get()=='None' else float(state.entries["minval_A_sing_entry"].get())
                state.maxval['A_sing']=None if state.entries["maxval_A_sing_entry"].get()=='None' else float(state.entries["maxval_A_sing_entry"].get())
               
                state.init['x0_sing']=None if state.entries["init_x0_sing_entry"].get()=='None' else float(state.entries["init_x0_sing_entry"].get())
                state.minval['x0_sing']=None if state.entries["minval_x0_sing_entry"].get()=='None' else float(state.entries["minval_x0_sing_entry"].get())
                state.maxval['x0_sing']=None if state.entries["maxval_x0_sing_entry"].get()=='None' else float( state.entries["maxval_x0_sing_entry"].get())
    
    
                state.init['dx_sing']=None if state.entries["init_dx_sing_entry"].get()=='None' else float(state.entries["init_dx_sing_entry"].get())
                state.minval['dx_sing']=None if state.entries["minval_dx_sing_entry"].get()=='None' else float(state.entries["minval_dx_sing_entry"].get())
                state.maxval['dx_sing']=None if state.entries["maxval_dx_sing_entry"].get()=='None' else float(state.entries["maxval_dx_sing_entry"].get())

               
            if state.kappa_pres==1:#if kappa function present, save parameter options from the gui for that function
   
                state.init['A_k']=None if state.entries["init_A_k_entry"].get()=='None' else float(state.entries["init_A_k_entry"].get())
                state.minval['A_k']=None if state.entries["minval_A_k_entry"].get()=='None' else float(state.entries["minval_A_k_entry"].get())
                state.maxval['A_k']=None if state.entries["maxval_A_k_entry"].get()=='None' else float(state.entries["maxval_A_k_entry"].get())
                
                state.init['T_k']=None if state.entries["init_T_k_entry"].get()=='None' else float(state.entries["init_T_k_entry"].get())
                state.minval['T_k']=None if state.entries["minval_T_k_entry"].get()=='None' else float(state.entries["minval_T_k_entry"].get())
                state.maxval['T_k']=None if state.entries["maxval_T_k_entry"].get()=='None' else float( state.entries["maxval_T_k_entry"].get())
                
                state.init['m_i']=None if state.entries["init_m_i_entry"].get()=='None' else float(state.entries["init_m_i_entry"].get())
                state.minval['m_i']=None if state.entries["minval_m_i_entry"].get()=='None' else float(state.entries["minval_m_i_entry"].get())
                state.maxval['m_i']=None if state.entries["maxval_m_i_entry"].get()=='None' else float( state.entries["maxval_m_i_entry"].get())
                
                state.init['n_i']=None if state.entries["init_n_i_entry"].get()=='None' else float(state.entries["init_n_i_entry"].get())
                state.minval['n_i']=None if state.entries["minval_n_i_entry"].get()=='None' else float(state.entries["minval_n_i_entry"].get())
                state.maxval['n_i']=None if state.entries["maxval_n_i_entry"].get()=='None' else float( state.entries["maxval_n_i_entry"].get())    
  
                state.init['kappa']=None if state.entries["init_kappa_entry"].get()=='None' else float(state.entries["init_kappa_entry"].get())
                state.minval['kappa']=None if state.entries["minval_kappa_entry"].get()=='None' else float(state.entries["minval_kappa_entry"].get())
                state.maxval['kappa']=None if state.entries["maxval_kappa_entry"].get()=='None' else float( state.entries["maxval_kappa_entry"].get())
               
               
            if state.bpl_and_therm_pres==1:
                state.init['T_c']=None if state.entries["init_T_c_entry"].get()=='None' else float(state.entries["init_T_c_entry"].get())
                state.minval['T_c']=None if state.entries["minval_T_c_entry"].get()=='None' else float(state.entries["minval_T_c_entry"].get())
                state.maxval['T_c']=None if state.entries["maxval_T_c_entry"].get()=='None' else float(state.entries["maxval_T_c_entry"].get())
                
                state.init['amp_c']=None if state.entries["init_amp_c_entry"].get()=='None' else float(state.entries["init_amp_c_entry"].get())
                state.minval['amp_c']=None if state.entries["minval_amp_c_entry"].get()=='None' else float(state.entries["minval_amp_c_entry"].get())
                state.maxval['amp_c']=None if state.entries["maxval_amp_c_entry"].get()=='None' else float(state.entries["maxval_amp_c_entry"].get())
                
                state.init['alpha_c']=None if state.entries["init_alpha_c_entry"].get()=='None' else float(state.entries["init_alpha_c_entry"].get())
                state.minval['alpha_c']=None if state.entries["minval_alpha_c_entry"].get()=='None' else float(state.entries["minval_alpha_c_entry"].get())
                state.maxval['alpha_c']=None if state.entries["maxval_alpha_c_entry"].get()=='None' else float(state.entries["maxval_alpha_c_entry"].get())
                
                state.init['x1_c']=None if state.entries["init_x1_c_entry"].get()=='None' else float(state.entries["init_x1_c_entry"].get())
                state.minval['x1_c']=None if state.entries["minval_x1_c_entry"].get()=='None' else float(state.entries["minval_x1_c_entry"].get())
                state.maxval['x1_c']=None if state.entries["maxval_x1_c_entry"].get()=='None' else float(state.entries["maxval_x1_c_entry"].get())
                
                state.init['x0_c']=None if state.entries["init_x0_c_entry"].get()=='None' else float(state.entries["init_x0_c_entry"].get())
                state.minval['x0_c']=None if state.entries["minval_x0_c_entry"].get()=='None' else float(state.entries["minval_x0_c_entry"].get())
                state.maxval['x0_c']=None if state.entries["maxval_x0_c_entry"].get()=='None' else float(state.entries["maxval_x0_c_entry"].get())
                
                state.init['B2_c']=None if state.entries["init_B2_c_entry"].get()=='None' else float(state.entries["init_B2_c_entry"].get())
                state.minval['B2_c']=None if state.entries["minval_B2_c_entry"].get()=='None' else float(state.entries["minval_B2_c_entry"].get())
                state.maxval['B2_c']=None if state.entries["maxval_B2_c_entry"].get()=='None' else float(state.entries["maxval_B2_c_entry"].get())
                
                
                state.init['B_c']=None if state.entries["init_B_c_entry"].get()=='None' else float(state.entries["init_B_c_entry"].get())
                state.minval['B_c']=None if state.entries["minval_B_c_entry"].get()=='None' else float(state.entries["minval_B_c_entry"].get())
                state.maxval['B_c']=None if state.entries["maxval_B_c_entry"].get()=='None' else float( state.entries["maxval_B_c_entry"].get())
                
            if state.double_therm_func_pres==1:#if double thermal function present, save parameter options from the gui for that function
                
                
                
                state.init['T_d_1']=None if state.entries["init_T_d_1_entry"].get()=='None' else float(state.entries["init_T_d_1_entry"].get())
                state.minval['T_d_1']=None if state.entries["minval_T_d_1_entry"].get()=='None' else float(state.entries["minval_T_d_1_entry"].get())
                state.maxval['T_d_1']=None if state.entries["maxval_T_d_1_entry"].get()=='None' else float(state.entries["maxval_T_d_1_entry"].get())
                
                state.init['amp_d_1']=None if state.entries["init_amp_d_1_entry"].get()=='None' else float(state.entries["init_amp_d_1_entry"].get())
                state.minval['amp_d_1']=None if state.entries["minval_amp_d_1_entry"].get()=='None' else float(state.entries["minval_amp_d_1_entry"].get())
                state.maxval['amp_d_1']=None if state.entries["maxval_amp_d_1_entry"].get()=='None' else float(state.entries["maxval_amp_d_1_entry"].get())
                
                state.init['alpha_d_1']=None if state.entries["init_alpha_d_1_entry"].get()=='None' else float(state.entries["init_alpha_d_1_entry"].get())
                state.minval['alpha_d_1']=None if state.entries["minval_alpha_d_1_entry"].get()=='None' else float(state.entries["minval_alpha_d_1_entry"].get())
                state.maxval['alpha_d_1']=None if state.entries["maxval_alpha_d_1_entry"].get()=='None' else float(state.entries["maxval_alpha_d_1_entry"].get())

                state.init['T_d_2']=None if state.entries["init_T_d_2_entry"].get()=='None' else float(state.entries["init_T_d_2_entry"].get())
                state.minval['T_d_2']=None if state.entries["minval_T_d_2_entry"].get()=='None' else float(state.entries["minval_T_d_2_entry"].get())
                state.maxval['T_d_2']=None if state.entries["maxval_T_d_2_entry"].get()=='None' else float(state.entries["maxval_T_d_2_entry"].get())
                
                state.init['amp_d_2']=None if state.entries["init_amp_d_2_entry"].get()=='None' else float(state.entries["init_amp_d_2_entry"].get())
                state.minval['amp_d_2']=None if state.entries["minval_amp_d_2_entry"].get()=='None' else float(state.entries["minval_amp_d_2_entry"].get())
                state.maxval['amp_d_2']=None if state.entries["maxval_amp_d_2_entry"].get()=='None' else float(state.entries["maxval_amp_d_2_entry"].get())
                
                state.init['alpha_d_2']=None if state.entries["init_alpha_d_2_entry"].get()=='None' else float(state.entries["init_alpha_d_2_entry"].get())
                state.minval['alpha_d_2']=None if state.entries["minval_alpha_d_2_entry"].get()=='None' else float(state.entries["minval_alpha_d_2_entry"].get())
                state.maxval['alpha_d_2']=None if state.entries["maxval_alpha_d_2_entry"].get()=='None' else float(state.entries["maxval_alpha_d_2_entry"].get())
            
            
            if state.tpl_pres==1:#if tpl function present, save parameter options from the gui for that function
                state.init['x1_tpl']=None if state.entries["init_x1_tpl_entry"].get()=='None' else float(state.entries["init_x1_tpl_entry"].get())
                state.minval['x1_tpl']=None if state.entries["minval_x1_tpl_entry"].get()=='None' else float(state.entries["minval_x1_tpl_entry"].get())
                state.maxval['x1_tpl']=None if state.entries["maxval_x1_tpl_entry"].get()=='None' else float(state.entries["maxval_x1_tpl_entry"].get())
                
                                
                state.init['x2_tpl']=None if state.entries["init_x2_tpl_entry"].get()=='None' else float(state.entries["init_x2_tpl_entry"].get())
                state.minval['x2_tpl']=None if state.entries["minval_x2_tpl_entry"].get()=='None' else float(state.entries["minval_x2_tpl_entry"].get())
                state.maxval['x2_tpl']=None if state.entries["maxval_x2_tpl_entry"].get()=='None' else float(state.entries["maxval_x2_tpl_entry"].get())
                
                
                state.init['A2_tpl']=None if state.entries["init_A2_tpl_entry"].get()=='None' else float(state.entries["init_A2_tpl_entry"].get())
                state.minval['A2_tpl']=None if state.entries["minval_A2_tpl_entry"].get()=='None' else float(state.entries["minval_A2_tpl_entry"].get())
                state.maxval['A2_tpl']=None if state.entries["maxval_A2_tpl_entry"].get()=='None' else float(state.entries["maxval_A2_tpl_entry"].get())
                
                
                state.init['B2_tpl']=None if state.entries["init_B2_tpl_entry"].get()=='None' else float(state.entries["init_B2_tpl_entry"].get())
                state.minval['B2_tpl']=None if state.entries["minval_B2_tpl_entry"].get()=='None' else float(state.entries["minval_B2_tpl_entry"].get())
                state.maxval['B2_tpl']=None if state.entries["maxval_B2_tpl_entry"].get()=='None' else float(state.entries["maxval_B2_tpl_entry"].get())
                
                
                state.init['B_tpl']=None if state.entries["init_B_tpl_entry"].get()=='None' else float(state.entries["init_B_tpl_entry"].get())
                state.minval['B_tpl']=None if state.entries["minval_B_tpl_entry"].get()=='None' else float(state.entries["minval_B_tpl_entry"].get())
                state.maxval['B_tpl']=None if state.entries["maxval_B_tpl_entry"].get()=='None' else float( state.entries["maxval_B_tpl_entry"].get())
            
            
                state.init['A_tpl']=None if state.entries["init_A_tpl_entry"].get()=='None' else float(state.entries["init_A_tpl_entry"].get())
                state.minval['A_tpl']=None if state.entries["minval_A_tpl_entry"].get()=='None' else float(state.entries["minval_A_tpl_entry"].get())
                state.maxval['A_tpl']=None if state.entries["maxval_A_tpl_entry"].get()=='None' else float(state.entries["maxval_A_tpl_entry"].get())
                
                state.init['A3_tpl']=None if state.entries["init_A3_tpl_entry"].get()=='None' else float(state.entries["init_A3_tpl_entry"].get())
                state.minval['A3_tpl']=None if state.entries["minval_A3_tpl_entry"].get()=='None' else float(state.entries["minval_A3_tpl_entry"].get())
                state.maxval['A3_tpl']=None if state.entries["maxval_A3_tpl_entry"].get()=='None' else float(state.entries["maxval_A3_tpl_entry"].get())
                
                
                state.init['B3_tpl']=None if state.entries["init_B3_tpl_entry"].get()=='None' else float(state.entries["init_B3_tpl_entry"].get())
                state.minval['B3_tpl']=None if state.entries["minval_B3_tpl_entry"].get()=='None' else float(state.entries["minval_B3_tpl_entry"].get())
                state.maxval['B3_tpl']=None if state.entries["maxval_B3_tpl_entry"].get()=='None' else float(state.entries["maxval_B3_tpl_entry"].get())
                
                state.init['x0_tpl']=None if state.entries["init_x0_tpl_entry"].get()=='None' else float(state.entries["init_x0_tpl_entry"].get())
                state.minval['x0_tpl']=None if state.entries["minval_x0_tpl_entry"].get()=='None' else float(state.entries["minval_x0_tpl_entry"].get())
                state.maxval['x0_tpl']=None if state.entries["maxval_x0_tpl_entry"].get()=='None' else float( state.entries["maxval_x0_tpl_entry"].get())
    
    
                state.init['dx_tpl']=None if state.entries["init_dx_tpl_entry"].get()=='None' else float(state.entries["init_dx_tpl_entry"].get())
                state.minval['dx_tpl']=None if state.entries["minval_dx_tpl_entry"].get()=='None' else float(state.entries["minval_dx_tpl_entry"].get())
                state.maxval['dx_tpl']=None if state.entries["maxval_dx_tpl_entry"].get()=='None' else float(state.entries["maxval_dx_tpl_entry"].get())
            
            if state.qpl_pres==1:#if qpl function present, save parameter options from the gui for that function
                state.init['x1_qpl']=None if state.entries["init_x1_qpl_entry"].get()=='None' else float(state.entries["init_x1_qpl_entry"].get())
                state.minval['x1_qpl']=None if state.entries["minval_x1_qpl_entry"].get()=='None' else float(state.entries["minval_x1_qpl_entry"].get())
                state.maxval['x1_qpl']=None if state.entries["maxval_x1_qpl_entry"].get()=='None' else float(state.entries["maxval_x1_qpl_entry"].get())
                
                                
                state.init['x2_qpl']=None if state.entries["init_x2_qpl_entry"].get()=='None' else float(state.entries["init_x2_qpl_entry"].get())
                state.minval['x2_qpl']=None if state.entries["minval_x2_qpl_entry"].get()=='None' else float(state.entries["minval_x2_qpl_entry"].get())
                state.maxval['x2_qpl']=None if state.entries["maxval_x2_qpl_entry"].get()=='None' else float(state.entries["maxval_x2_qpl_entry"].get())
                
                state.init['x3_qpl']=None if state.entries["init_x3_qpl_entry"].get()=='None' else float(state.entries["init_x3_qpl_entry"].get())
                state.minval['x3_qpl']=None if state.entries["minval_x3_qpl_entry"].get()=='None' else float(state.entries["minval_x3_qpl_entry"].get())
                state.maxval['x3_qpl']=None if state.entries["maxval_x3_qpl_entry"].get()=='None' else float(state.entries["maxval_x3_qpl_entry"].get())
                
                state.init['A2_qpl']=None if state.entries["init_A2_qpl_entry"].get()=='None' else float(state.entries["init_A2_qpl_entry"].get())
                state.minval['A2_qpl']=None if state.entries["minval_A2_qpl_entry"].get()=='None' else float(state.entries["minval_A2_qpl_entry"].get())
                state.maxval['A2_qpl']=None if state.entries["maxval_A2_qpl_entry"].get()=='None' else float(state.entries["maxval_A2_qpl_entry"].get())
                
                
                state.init['B2_qpl']=None if state.entries["init_B2_qpl_entry"].get()=='None' else float(state.entries["init_B2_qpl_entry"].get())
                state.minval['B2_qpl']=None if state.entries["minval_B2_qpl_entry"].get()=='None' else float(state.entries["minval_B2_qpl_entry"].get())
                state.maxval['B2_qpl']=None if state.entries["maxval_B2_qpl_entry"].get()=='None' else float(state.entries["maxval_B2_qpl_entry"].get())
                
                
                state.init['B_qpl']=None if state.entries["init_B_qpl_entry"].get()=='None' else float(state.entries["init_B_qpl_entry"].get())
                state.minval['B_qpl']=None if state.entries["minval_B_qpl_entry"].get()=='None' else float(state.entries["minval_B_qpl_entry"].get())
                state.maxval['B_qpl']=None if state.entries["maxval_B_qpl_entry"].get()=='None' else float( state.entries["maxval_B_qpl_entry"].get())
            
            
                state.init['A_qpl']=None if state.entries["init_A_qpl_entry"].get()=='None' else float(state.entries["init_A_qpl_entry"].get())
                state.minval['A_qpl']=None if state.entries["minval_A_qpl_entry"].get()=='None' else float(state.entries["minval_A_qpl_entry"].get())
                state.maxval['A_qpl']=None if state.entries["maxval_A_qpl_entry"].get()=='None' else float(state.entries["maxval_A_qpl_entry"].get())
                
                state.init['A3_qpl']=None if state.entries["init_A3_qpl_entry"].get()=='None' else float(state.entries["init_A3_qpl_entry"].get())
                state.minval['A3_qpl']=None if state.entries["minval_A3_qpl_entry"].get()=='None' else float(state.entries["minval_A3_qpl_entry"].get())
                state.maxval['A3_qpl']=None if state.entries["maxval_A3_qpl_entry"].get()=='None' else float(state.entries["maxval_A3_qpl_entry"].get())
                
                
                state.init['B3_qpl']=None if state.entries["init_B3_qpl_entry"].get()=='None' else float(state.entries["init_B3_qpl_entry"].get())
                state.minval['B3_qpl']=None if state.entries["minval_B3_qpl_entry"].get()=='None' else float(state.entries["minval_B3_qpl_entry"].get())
                state.maxval['B3_qpl']=None if state.entries["maxval_B3_qpl_entry"].get()=='None' else float(state.entries["maxval_B3_qpl_entry"].get())
                
                state.init['A4_qpl']=None if state.entries["init_A4_qpl_entry"].get()=='None' else float(state.entries["init_A4_qpl_entry"].get())
                state.minval['A4_qpl']=None if state.entries["minval_A4_qpl_entry"].get()=='None' else float(state.entries["minval_A4_qpl_entry"].get())
                state.maxval['A4_qpl']=None if state.entries["maxval_A4_qpl_entry"].get()=='None' else float(state.entries["maxval_A4_qpl_entry"].get())
                
                state.init['B4_qpl']=None if state.entries["init_B4_qpl_entry"].get()=='None' else float(state.entries["init_B4_qpl_entry"].get())
                state.minval['B4_qpl']=None if state.entries["minval_B4_qpl_entry"].get()=='None' else float(state.entries["minval_B4_qpl_entry"].get())
                state.maxval['B4_qpl']=None if state.entries["maxval_B4_qpl_entry"].get()=='None' else float(state.entries["maxval_B4_qpl_entry"].get())
                
                state.init['x0_qpl']=None if state.entries["init_x0_qpl_entry"].get()=='None' else float(state.entries["init_x0_qpl_entry"].get())
                state.minval['x0_qpl']=None if state.entries["minval_x0_qpl_entry"].get()=='None' else float(state.entries["minval_x0_qpl_entry"].get())
                state.maxval['x0_qpl']=None if state.entries["maxval_x0_qpl_entry"].get()=='None' else float( state.entries["maxval_x0_qpl_entry"].get())
    
    
                state.init['dx_qpl']=None if state.entries["init_dx_qpl_entry"].get()=='None' else float(state.entries["init_dx_qpl_entry"].get())
                state.minval['dx_qpl']=None if state.entries["minval_dx_qpl_entry"].get()=='None' else float(state.entries["minval_dx_qpl_entry"].get())
                state.maxval['dx_qpl']=None if state.entries["maxval_dx_qpl_entry"].get()=='None' else float(state.entries["maxval_dx_qpl_entry"].get())
                
            if state.quint_pl_pres==1:#if 5pl function present, save parameter options from the gui for that function
                state.init['x1_5pl']=None if state.entries["init_x1_5pl_entry"].get()=='None' else float(state.entries["init_x1_5pl_entry"].get())
                state.minval['x1_5pl']=None if state.entries["minval_x1_5pl_entry"].get()=='None' else float(state.entries["minval_x1_5pl_entry"].get())
                state.maxval['x1_5pl']=None if state.entries["maxval_x1_5pl_entry"].get()=='None' else float(state.entries["maxval_x1_5pl_entry"].get())
                
                                
                state.init['x2_5pl']=None if state.entries["init_x2_5pl_entry"].get()=='None' else float(state.entries["init_x2_5pl_entry"].get())
                state.minval['x2_5pl']=None if state.entries["minval_x2_5pl_entry"].get()=='None' else float(state.entries["minval_x2_5pl_entry"].get())
                state.maxval['x2_5pl']=None if state.entries["maxval_x2_5pl_entry"].get()=='None' else float(state.entries["maxval_x2_5pl_entry"].get())
                
                state.init['x3_5pl']=None if state.entries["init_x3_5pl_entry"].get()=='None' else float(state.entries["init_x3_5pl_entry"].get())
                state.minval['x3_5pl']=None if state.entries["minval_x3_5pl_entry"].get()=='None' else float(state.entries["minval_x3_5pl_entry"].get())
                state.maxval['x3_5pl']=None if state.entries["maxval_x3_5pl_entry"].get()=='None' else float(state.entries["maxval_x3_5pl_entry"].get())
                
                state.init['x4_5pl']=None if state.entries["init_x4_5pl_entry"].get()=='None' else float(state.entries["init_x4_5pl_entry"].get())
                state.minval['x4_5pl']=None if state.entries["minval_x4_5pl_entry"].get()=='None' else float(state.entries["minval_x4_5pl_entry"].get())
                state.maxval['x4_5pl']=None if state.entries["maxval_x4_5pl_entry"].get()=='None' else float(state.entries["maxval_x4_5pl_entry"].get())
                
                state.init['A2_5pl']=None if state.entries["init_A2_5pl_entry"].get()=='None' else float(state.entries["init_A2_5pl_entry"].get())
                state.minval['A2_5pl']=None if state.entries["minval_A2_5pl_entry"].get()=='None' else float(state.entries["minval_A2_5pl_entry"].get())
                state.maxval['A2_5pl']=None if state.entries["maxval_A2_5pl_entry"].get()=='None' else float(state.entries["maxval_A2_5pl_entry"].get())
                
                
                state.init['B2_5pl']=None if state.entries["init_B2_5pl_entry"].get()=='None' else float(state.entries["init_B2_5pl_entry"].get())
                state.minval['B2_5pl']=None if state.entries["minval_B2_5pl_entry"].get()=='None' else float(state.entries["minval_B2_5pl_entry"].get())
                state.maxval['B2_5pl']=None if state.entries["maxval_B2_5pl_entry"].get()=='None' else float(state.entries["maxval_B2_5pl_entry"].get())
                
                
                state.init['B_5pl']=None if state.entries["init_B_5pl_entry"].get()=='None' else float(state.entries["init_B_5pl_entry"].get())
                state.minval['B_5pl']=None if state.entries["minval_B_5pl_entry"].get()=='None' else float(state.entries["minval_B_5pl_entry"].get())
                state.maxval['B_5pl']=None if state.entries["maxval_B_5pl_entry"].get()=='None' else float( state.entries["maxval_B_5pl_entry"].get())
            
            
                state.init['A_5pl']=None if state.entries["init_A_5pl_entry"].get()=='None' else float(state.entries["init_A_5pl_entry"].get())
                state.minval['A_5pl']=None if state.entries["minval_A_5pl_entry"].get()=='None' else float(state.entries["minval_A_5pl_entry"].get())
                state.maxval['A_5pl']=None if state.entries["maxval_A_5pl_entry"].get()=='None' else float(state.entries["maxval_A_5pl_entry"].get())
                
                state.init['A3_5pl']=None if state.entries["init_A3_5pl_entry"].get()=='None' else float(state.entries["init_A3_5pl_entry"].get())
                state.minval['A3_5pl']=None if state.entries["minval_A3_5pl_entry"].get()=='None' else float(state.entries["minval_A3_5pl_entry"].get())
                state.maxval['A3_5pl']=None if state.entries["maxval_A3_5pl_entry"].get()=='None' else float(state.entries["maxval_A3_5pl_entry"].get())
                
                
                state.init['B3_5pl']=None if state.entries["init_B3_5pl_entry"].get()=='None' else float(state.entries["init_B3_5pl_entry"].get())
                state.minval['B3_5pl']=None if state.entries["minval_B3_5pl_entry"].get()=='None' else float(state.entries["minval_B3_5pl_entry"].get())
                state.maxval['B3_5pl']=None if state.entries["maxval_B3_5pl_entry"].get()=='None' else float(state.entries["maxval_B3_5pl_entry"].get())
                
                state.init['A4_5pl']=None if state.entries["init_A4_5pl_entry"].get()=='None' else float(state.entries["init_A4_5pl_entry"].get())
                state.minval['A4_5pl']=None if state.entries["minval_A4_5pl_entry"].get()=='None' else float(state.entries["minval_A4_5pl_entry"].get())
                state.maxval['A4_5pl']=None if state.entries["maxval_A4_5pl_entry"].get()=='None' else float(state.entries["maxval_A4_5pl_entry"].get())
                
                state.init['B4_5pl']=None if state.entries["init_B4_5pl_entry"].get()=='None' else float(state.entries["init_B4_5pl_entry"].get())
                state.minval['B4_5pl']=None if state.entries["minval_B4_5pl_entry"].get()=='None' else float(state.entries["minval_B4_5pl_entry"].get())
                state.maxval['B4_5pl']=None if state.entries["maxval_B4_5pl_entry"].get()=='None' else float(state.entries["maxval_B4_5pl_entry"].get()) 
                
                state.init['A5_5pl']=None if state.entries["init_A5_5pl_entry"].get()=='None' else float(state.entries["init_A5_5pl_entry"].get())
                state.minval['A5_5pl']=None if state.entries["minval_A5_5pl_entry"].get()=='None' else float(state.entries["minval_A5_5pl_entry"].get())
                state.maxval['A5_5pl']=None if state.entries["maxval_A5_5pl_entry"].get()=='None' else float(state.entries["maxval_A5_5pl_entry"].get())
                
                state.init['B5_5pl']=None if state.entries["init_B5_5pl_entry"].get()=='None' else float(state.entries["init_B5_5pl_entry"].get())
                state.minval['B5_5pl']=None if state.entries["minval_B5_5pl_entry"].get()=='None' else float(state.entries["minval_B5_5pl_entry"].get())
                state.maxval['B5_5pl']=None if state.entries["maxval_B5_5pl_entry"].get()=='None' else float(state.entries["maxval_B5_5pl_entry"].get()) 
            
                state.init['x0_5pl']=None if state.entries["init_x0_5pl_entry"].get()=='None' else float(state.entries["init_x0_5pl_entry"].get())
                state.minval['x0_5pl']=None if state.entries["minval_x0_5pl_entry"].get()=='None' else float(state.entries["minval_x0_5pl_entry"].get())
                state.maxval['x0_5pl']=None if state.entries["maxval_x0_5pl_entry"].get()=='None' else float( state.entries["maxval_x0_5pl_entry"].get())
    
    
                state.init['dx_5pl']=None if state.entries["init_dx_5pl_entry"].get()=='None' else float(state.entries["init_dx_5pl_entry"].get())
                state.minval['dx_5pl']=None if state.entries["minval_dx_5pl_entry"].get()=='None' else float(state.entries["minval_dx_5pl_entry"].get())
                state.maxval['dx_5pl']=None if state.entries["maxval_dx_5pl_entry"].get()=='None' else float(state.entries["maxval_dx_5pl_entry"].get())
                
            #pull the min/max energy (x) values to fit to

            state.fitmin=float(fitmin_entry.get())
            state.fitmax=float(fitmax_entry.get())
            
            #validate limits
            if not validate_lims(state.fitmin,state.fitmax):
                tk.messagebox.showerror("Invalid Input","Fit limits should be floats with max greater than min")
            else:
                
                #validate entries
                validity=dict()
                for ind in state.minval.keys():
                    min_val=state.minval[ind]
                    max_val=state.maxval[ind]
                    valid = validate_minmaxval(min_val,max_val)
                    validity[ind]=(valid)
                    
                    #ensure finit limits
                    finite=dict()
                for ind in state.minval.keys():
                    min_val=state.minval[ind]
                    max_val=state.maxval[ind]
                    finiteness = validate_finite(min_val,max_val)
                    finite[ind]=(finiteness)
                if False in validity:#show where error is !!!!!
                    false_keys=list()            
                    for key, value in validity.items():
                        if value is False:
                            false_keys.append(key)
                    
                    tk.messagebox.showerror("Invalid Input",f"Parameter limits should be floats with max greater than min for parameter(s) {false_keys}")
                else:
                    validity=dict()
                    for ind in state.init.keys():
                        min_val=state.minval[ind]
                        max_val=state.maxval[ind]
                        init_val=state.init[ind]
                        validity[ind]=validate_init(init_val,min_val,max_val)
                    if False in validity:#show where error is !!!!!
                        false_keys=list()            
                        for key, value in validity.items():
                            if value is False:
                                false_keys.append(key)
                        
                        tk.messagebox.showerror("Invalid Input",f"Parameter initial values should be floats between their max and min values for parameter(s) {false_keys}")
                    
                    elif False in finite:
                        inf_keys=list()            
                        for key, value in finite.items():
                            if value is False:
                                inf_keys.append(key)
                        
                        tk.messagebox.showerror("Invalid Input",f"Parameter limits should be finite value floats for parameter(s) {inf_keys}")
                    
                    
                    else:
#%%fit outputs                        #perform the fitting function defined above to obtain the minimised parameters
                        


                        #conduct fitting process
                        global x_data_E_sliced
                        state.parvals,state.param_uncert_calced,x_data_E_sliced=fitting(state.header,state.init,state.vary,state.minval,state.maxval,x_data,y_data,uncert,state.fitmin,state.fitmax,spec_type)
                        

                        #add the results into the entry boxes
                        if state.bpl_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui

                            state.entries["init_x1_entry"].delete(0, tk.END)
                            state.entries["init_x1_entry"].insert(0,state.parvals["x1"])
                            state.entries["init_A_entry"].delete(0, tk.END)
                            state.entries["init_A_entry"].insert(0,state.parvals["A"])
                            state.entries["init_B_entry"].delete(0, tk.END)
                            state.entries["init_B_entry"].insert(0,state.parvals["B"])
                            state.entries["init_A2_entry"].delete(0, tk.END)
                            state.entries["init_A2_entry"].insert(0,state.parvals["A2"])
                            state.entries["init_B2_entry"].delete(0, tk.END)
                            state.entries["init_B2_entry"].insert(0,state.parvals["B2"])
                            state.entries["init_x0_bpl_entry"].delete(0, tk.END)
                            state.entries["init_x0_bpl_entry"].insert(0,state.parvals["x0_bpl"])
                            state.entries["init_dx_bpl_entry"].delete(0, tk.END)
                            state.entries["init_dx_bpl_entry"].insert(0,state.parvals["dx_bpl"])
                            
                        if state.therm_func_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui
                            state.entries["init_amp_entry"].delete(0, tk.END)
                            state.entries["init_amp_entry"].insert(0,state.parvals["amp"])
                            state.entries["init_T_entry"].delete(0, tk.END)
                            state.entries["init_T_entry"].insert(0,state.parvals["T"])
                            state.entries["init_alpha_entry"].delete(0, tk.END)
                            state.entries["init_alpha_entry"].insert(0,state.parvals["alpha"])
                
                        if state.gauss_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui
                            state.entries["init_gauss_amp_entry"].delete(0, tk.END)
                            state.entries["init_gauss_amp_entry"].insert(0,state.parvals["gauss_amp"])
                            state.entries["init_gauss_centre_entry"].delete(0, tk.END)
                            state.entries["init_gauss_centre_entry"].insert(0,state.parvals["gauss_centre"])
                            state.entries["init_sigma_entry"].delete(0, tk.END)
                            state.entries["init_sigma_entry"].insert(0,state.parvals["sigma"]) 
                
                
                        if state.power_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui
                            state.entries["init_A_sing_entry"].delete(0, tk.END)
                            state.entries["init_A_sing_entry"].insert(0,state.parvals["A_sing"])
                            state.entries["init_B_sing_entry"].delete(0, tk.END)
                            state.entries["init_B_sing_entry"].insert(0,state.parvals["B_sing"])
                            state.entries["init_x0_sing_entry"].delete(0, tk.END)
                            state.entries["init_x0_sing_entry"].insert(0,state.parvals["x0_sing"])
                            state.entries["init_dx_sing_entry"].delete(0, tk.END)
                            state.entries["init_dx_sing_entry"].insert(0,state.parvals["dx_sing"])
                            
                        if state.kappa_pres==1:
                            
                            state.entries["init_A_k_entry"].delete(0, tk.END)
                            state.entries["init_A_k_entry"].insert(0,state.parvals["A_k"])
                            state.entries["init_T_k_entry"].delete(0, tk.END)
                            state.entries["init_T_k_entry"].insert(0,state.parvals["T_k"])
                            state.entries["init_m_i_entry"].delete(0, tk.END)
                            state.entries["init_m_i_entry"].insert(0,state.parvals["m_i"])
                            state.entries["init_n_i_entry"].delete(0, tk.END)
                            state.entries["init_n_i_entry"].insert(0,state.parvals["n_i"])
                            state.entries["init_kappa_entry"].delete(0, tk.END)                            
                            state.entries["init_kappa_entry"].insert(0,state.parvals["kappa"])
                        
                        
                        if state.bpl_and_therm_pres==1:
                            state.entries["init_amp_c_entry"].delete(0, tk.END)
                            state.entries["init_amp_c_entry"].insert(0,state.parvals["amp_c"])
                            state.entries["init_T_c_entry"].delete(0, tk.END)
                            state.entries["init_T_c_entry"].insert(0,state.parvals["T_c"])
                            state.entries["init_alpha_c_entry"].delete(0, tk.END)
                            state.entries["init_alpha_c_entry"].insert(0,state.parvals["alpha_c"])
                            state.entries["init_x0_c_entry"].delete(0, tk.END)
                            state.entries["init_x0_c_entry"].insert(0,state.parvals["x0_c"])
                            state.entries["init_x1_c_entry"].delete(0, tk.END)
                            state.entries["init_x1_c_entry"].insert(0,state.parvals["x1_c"])
                            state.entries["init_B_c_entry"].delete(0, tk.END)
                            state.entries["init_B_c_entry"].insert(0,state.parvals["B_c"])
                            state.entries["init_B2_c_entry"].delete(0, tk.END)
                            state.entries["init_B2_c_entry"].insert(0,state.parvals["B2_c"])
                            
                            
                            
                            
                        if state.double_therm_func_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui
                            state.entries["init_amp_d_1_entry"].delete(0, tk.END)
                            state.entries["init_amp_d_1_entry"].insert(0,state.parvals["amp_d_1"])
                            state.entries["init_T_d_1_entry"].delete(0, tk.END)
                            state.entries["init_T_d_1_entry"].insert(0,state.parvals["T_d_1"])
                            state.entries["init_alpha_d_1_entry"].delete(0, tk.END)
                            state.entries["init_alpha_d_1_entry"].insert(0,state.parvals["alpha_d_1"])
                            state.entries["init_amp_d_2_entry"].delete(0, tk.END)
                            state.entries["init_amp_d_2_entry"].insert(0,state.parvals["amp_d_2"])
                            state.entries["init_T_d_2_entry"].delete(0, tk.END)
                            state.entries["init_T_d_2_entry"].insert(0,state.parvals["T_d_2"])
                            state.entries["init_alpha_d_2_entry"].delete(0, tk.END)
                            state.entries["init_alpha_d_2_entry"].insert(0,state.parvals["alpha_d_2"])
                
                        if state.tpl_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui

                            state.entries["init_x1_tpl_entry"].delete(0, tk.END)
                            state.entries["init_x1_tpl_entry"].insert(0,state.parvals["x1_tpl"])
                            state.entries["init_x2_tpl_entry"].delete(0, tk.END)
                            state.entries["init_x2_tpl_entry"].insert(0,state.parvals["x2_tpl"])
                            state.entries["init_A_tpl_entry"].delete(0, tk.END)
                            state.entries["init_A_tpl_entry"].insert(0,state.parvals["A_tpl"])
                            state.entries["init_B_tpl_entry"].delete(0, tk.END)
                            state.entries["init_B_tpl_entry"].insert(0,state.parvals["B_tpl"])
                            state.entries["init_A2_tpl_entry"].delete(0, tk.END)
                            state.entries["init_A2_tpl_entry"].insert(0,state.parvals["A2_tpl"])
                            state.entries["init_B2_tpl_entry"].delete(0, tk.END)
                            state.entries["init_B2_tpl_entry"].insert(0,state.parvals["B2_tpl"])
                            state.entries["init_A3_tpl_entry"].delete(0, tk.END)
                            state.entries["init_A3_tpl_entry"].insert(0,state.parvals["A3_tpl"])
                            state.entries["init_B3_tpl_entry"].delete(0, tk.END)
                            state.entries["init_B3_tpl_entry"].insert(0,state.parvals["B3_tpl"])
                            state.entries["init_x0_tpl_entry"].delete(0, tk.END)
                            state.entries["init_x0_tpl_entry"].insert(0,state.parvals["x0_tpl"])
                            state.entries["init_dx_tpl_entry"].delete(0, tk.END)
                            state.entries["init_dx_tpl_entry"].insert(0,state.parvals["dx_tpl"])

                        if state.qpl_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui

                            state.entries["init_x1_qpl_entry"].delete(0, tk.END)
                            state.entries["init_x1_qpl_entry"].insert(0,state.parvals["x1_qpl"])
                            state.entries["init_x2_qpl_entry"].delete(0, tk.END)
                            state.entries["init_x2_qpl_entry"].insert(0,state.parvals["x2_qpl"])
                            state.entries["init_x3_qpl_entry"].delete(0, tk.END)
                            state.entries["init_x3_qpl_entry"].insert(0,state.parvals["x3_qpl"])
                            state.entries["init_A_qpl_entry"].delete(0, tk.END)
                            state.entries["init_A_qpl_entry"].insert(0,state.parvals["A_qpl"])
                            state.entries["init_B_qpl_entry"].delete(0, tk.END)
                            state.entries["init_B_qpl_entry"].insert(0,state.parvals["B_qpl"])
                            state.entries["init_A2_qpl_entry"].delete(0, tk.END)
                            state.entries["init_A2_qpl_entry"].insert(0,state.parvals["A2_qpl"])
                            state.entries["init_B2_qpl_entry"].delete(0, tk.END)
                            state.entries["init_B2_qpl_entry"].insert(0,state.parvals["B2_qpl"])
                            state.entries["init_A3_qpl_entry"].delete(0, tk.END)
                            state.entries["init_A3_qpl_entry"].insert(0,state.parvals["A3_qpl"])
                            state.entries["init_B3_qpl_entry"].delete(0, tk.END)
                            state.entries["init_B3_qpl_entry"].insert(0,state.parvals["B3_qpl"])
                            state.entries["init_A4_qpl_entry"].delete(0, tk.END)
                            state.entries["init_A4_qpl_entry"].insert(0,state.parvals["A4_qpl"])
                            state.entries["init_B4_qpl_entry"].delete(0, tk.END)
                            state.entries["init_B4_qpl_entry"].insert(0,state.parvals["B4_qpl"])
                            state.entries["init_x0_qpl_entry"].delete(0, tk.END)
                            state.entries["init_x0_qpl_entry"].insert(0,state.parvals["x0_qpl"])
                            state.entries["init_dx_qpl_entry"].delete(0, tk.END)
                            state.entries["init_dx_qpl_entry"].insert(0,state.parvals["dx_qpl"])
                            
                        if state.quint_pl_pres==1:#if this function present, clear the initial values and replace with the newly fitted ones in the gui

                            state.entries["init_x1_5pl_entry"].delete(0, tk.END)
                            state.entries["init_x1_5pl_entry"].insert(0,state.parvals["x1_5pl"])
                            state.entries["init_x2_5pl_entry"].delete(0, tk.END)
                            state.entries["init_x2_5pl_entry"].insert(0,state.parvals["x2_5pl"])
                            state.entries["init_x3_5pl_entry"].delete(0, tk.END)
                            state.entries["init_x3_5pl_entry"].insert(0,state.parvals["x3_5pl"])
                            state.entries["init_x4_5pl_entry"].delete(0, tk.END)
                            state.entries["init_x4_5pl_entry"].insert(0,state.parvals["x4_5pl"])
                            state.entries["init_A_5pl_entry"].delete(0, tk.END)
                            state.entries["init_A_5pl_entry"].insert(0,state.parvals["A_5pl"])
                            state.entries["init_B_5pl_entry"].delete(0, tk.END)
                            state.entries["init_B_5pl_entry"].insert(0,state.parvals["B_5pl"])
                            state.entries["init_A2_5pl_entry"].delete(0, tk.END)
                            state.entries["init_A2_5pl_entry"].insert(0,state.parvals["A2_5pl"])
                            state.entries["init_B2_5pl_entry"].delete(0, tk.END)
                            state.entries["init_B2_5pl_entry"].insert(0,state.parvals["B2_5pl"])
                            state.entries["init_A3_5pl_entry"].delete(0, tk.END)
                            state.entries["init_A3_5pl_entry"].insert(0,state.parvals["A3_5pl"])
                            state.entries["init_B3_5pl_entry"].delete(0, tk.END)
                            state.entries["init_B3_5pl_entry"].insert(0,state.parvals["B3_5pl"])
                            state.entries["init_A4_5pl_entry"].delete(0, tk.END)
                            state.entries["init_A4_5pl_entry"].insert(0,state.parvals["A4_5pl"])
                            state.entries["init_B4_5pl_entry"].delete(0, tk.END)
                            state.entries["init_B4_5pl_entry"].insert(0,state.parvals["B4_5pl"])
                            state.entries["init_A5_5pl_entry"].delete(0, tk.END)
                            state.entries["init_A5_5pl_entry"].insert(0,state.parvals["A5_5pl"])
                            state.entries["init_B5_5pl_entry"].delete(0, tk.END)
                            state.entries["init_B5_5pl_entry"].insert(0,state.parvals["B5_5pl"])
                            state.entries["init_x0_5pl_entry"].delete(0, tk.END)
                            state.entries["init_x0_5pl_entry"].insert(0,state.parvals["x0_5pl"])
                            state.entries["init_dx_5pl_entry"].delete(0, tk.END)
                            state.entries["init_dx_5pl_entry"].insert(0,state.parvals["dx_5pl"])
            
                        parvals_new=state.parvals
               
                        print("uncerts")
                        print(state.param_uncert_calced)
                        
                        
                        #the percentage uncerts 
                        print('percent uncerts')
                        for key in list(state.param_uncert_calced.keys()):
                           frac=state.param_uncert_calced[key]/parvals_new[key]
                           print(str(key)+":"+str(frac*100))               
        
        
        
        
            
        except ValueError as e:
               tk.messagebox.showerror("Invalid Input","Inputs should be floating point intergers")
               print(e)
       

#%%preview buttns handling   

    def preview_btn_hndl():#function to handle preview button
        if state.preview_window is not None:# and state.preview_window.winfo_exists():
            #close any open figues
            try: state.preview_window.destroy()
            except tk.TclError: pass
            state.preview_window=None
            
        if state.fit_window is not None:# and state.fit_window.winfo_exists():
            #close any open figues
            try: state.fit_window.destroy()
            except tk.TclError: pass
            state.fit_window=None
        if state.resid_window is not None:# and state.resid_window.winfo_exists():
            #close any open figues
            try: state.resid_window.destroy()
            except tk.TclError: pass
            state.resid_window=None
        
        
        state.header=f"bpl_pres={state.bpl_pres}; therm_func_pres={state.therm_func_pres}; gauss_pres={state.gauss_pres}; power_pres={state.power_pres}; kappa_pres={state.kappa_pres}; bpl_and_therm_pres={state.bpl_and_therm_pres}; double_therm_func_pres={state.double_therm_func_pres}; tpl_pres={state.tpl_pres}; qpl_pres={state.qpl_pres}; quint_pl_pres={state.quint_pl_pres};"#defines state.header (header) according to what functions are currently present in the gui
        try:#try excpet statement is to validate inputs as integers
            if state.therm_func_pres==1:#if thermal function present, save parameter options from the gui for that function
                
                global frame_therm
                
                state.init['T']=None if state.entries["init_T_entry"].get()=='None' else float(state.entries["init_T_entry"].get())
                state.minval['T']=None if state.entries["minval_T_entry"].get()=='None' else float(state.entries["minval_T_entry"].get())
                state.maxval['T']=None if state.entries["maxval_T_entry"].get()=='None' else float(state.entries["maxval_T_entry"].get())
                
                state.init['amp']=None if state.entries["init_amp_entry"].get()=='None' else float(state.entries["init_amp_entry"].get())
                state.minval['amp']=None if state.entries["minval_amp_entry"].get()=='None' else float(state.entries["minval_amp_entry"].get())
                state.maxval['amp']=None if state.entries["maxval_amp_entry"].get()=='None' else float(state.entries["maxval_amp_entry"].get())
                
                state.init['alpha']=None if state.entries["init_alpha_entry"].get()=='None' else float(state.entries["init_alpha_entry"].get())
                state.minval['alpha']=None if state.entries["minval_alpha_entry"].get()=='None' else float(state.entries["minval_alpha_entry"].get())
                state.maxval['alpha']=None if state.entries["maxval_alpha_entry"].get()=='None' else float(state.entries["maxval_alpha_entry"].get())
                
            if state.bpl_pres==1:#if bpl function present, save parameter options from the gui for that function
                
                
                
                state.init['x1']=None if state.entries["init_x1_entry"].get()=='None' else float(state.entries["init_x1_entry"].get())
                state.minval['x1']=None if state.entries["minval_x1_entry"].get()=='None' else float(state.entries["minval_x1_entry"].get())
                state.maxval['x1']=None if state.entries["maxval_x1_entry"].get()=='None' else float(state.entries["maxval_x1_entry"].get())
                
                
                
                
                state.init['A2']=None if state.entries["init_A2_entry"].get()=='None' else float(state.entries["init_A2_entry"].get())
                state.minval['A2']=None if state.entries["minval_A2_entry"].get()=='None' else float(state.entries["minval_A2_entry"].get())
                state.maxval['A2']=None if state.entries["maxval_A2_entry"].get()=='None' else float(state.entries["maxval_A2_entry"].get())
                
                
                state.init['B2']=None if state.entries["init_B2_entry"].get()=='None' else float(state.entries["init_B2_entry"].get())
                state.minval['B2']=None if state.entries["minval_B2_entry"].get()=='None' else float(state.entries["minval_B2_entry"].get())
                state.maxval['B2']=None if state.entries["maxval_B2_entry"].get()=='None' else float(state.entries["maxval_B2_entry"].get())
                
                
                state.init['B']=None if state.entries["init_B_entry"].get()=='None' else float(state.entries["init_B_entry"].get())
                state.minval['B']=None if state.entries["minval_B_entry"].get()=='None' else float(state.entries["minval_B_entry"].get())
                state.maxval['B']=None if state.entries["maxval_B_entry"].get()=='None' else float( state.entries["maxval_B_entry"].get())
    
    
                state.init['A']=None if state.entries["init_A_entry"].get()=='None' else float(state.entries["init_A_entry"].get())
                state.minval['A']=None if state.entries["minval_A_entry"].get()=='None' else float(state.entries["minval_A_entry"].get())
                state.maxval['A']=None if state.entries["maxval_A_entry"].get()=='None' else float(state.entries["maxval_A_entry"].get())
                
                state.init['x0_bpl']=None if state.entries["init_x0_bpl_entry"].get()=='None' else float(state.entries["init_x0_bpl_entry"].get())
                state.minval['x0_bpl']=None if state.entries["minval_x0_bpl_entry"].get()=='None' else float(state.entries["minval_x0_bpl_entry"].get())
                state.maxval['x0_bpl']=None if state.entries["maxval_x0_bpl_entry"].get()=='None' else float( state.entries["maxval_x0_bpl_entry"].get())
    
    
                state.init['dx_bpl']=None if state.entries["init_dx_bpl_entry"].get()=='None' else float(state.entries["init_dx_bpl_entry"].get())
                state.minval['dx_bpl']=None if state.entries["minval_dx_bpl_entry"].get()=='None' else float(state.entries["minval_dx_bpl_entry"].get())
                state.maxval['dx_bpl']=None if state.entries["maxval_dx_bpl_entry"].get()=='None' else float(state.entries["maxval_dx_bpl_entry"].get())
                
                
            if state.gauss_pres==1:#if gaussian function present, save parameter options from the gui for that function
                state.init['gauss_centre']=None if state.entries["init_gauss_centre_entry"].get()=='None' else float(state.entries["init_gauss_centre_entry"].get())
                state.minval['gauss_centre']=None if state.entries["minval_gauss_centre_entry"].get()=='None' else float(state.entries["minval_gauss_centre_entry"].get())
                state.maxval['gauss_centre']=None if state.entries["maxval_gauss_centre_entry"].get()=='None' else float(state.entries["maxval_gauss_centre_entry"].get())
                
                
                state.init['gauss_amp']=None if state.entries["init_gauss_amp_entry"].get()=='None' else float(state.entries["init_gauss_amp_entry"].get())
                state.minval['gauss_amp']=None if state.entries["minval_gauss_amp_entry"].get()=='None' else float(state.entries["minval_gauss_amp_entry"].get())
                state.maxval['gauss_amp']=None if state.entries["maxval_gauss_amp_entry"].get()=='None' else float( state.entries["maxval_gauss_amp_entry"].get())
    
    
                state.init['sigma']=None if state.entries["init_sigma_entry"].get()=='None' else float(state.entries["init_sigma_entry"].get())
                state.minval['sigma']=None if state.entries["minval_sigma_entry"].get()=='None' else float(state.entries["minval_sigma_entry"].get())
                state.maxval['sigma']=None if state.entries["maxval_sigma_entry"].get()=='None' else float(state.entries["maxval_sigma_entry"].get())
               
               
               
               
               
            if state.power_pres==1:#if single power law function present, save parameter options from the gui for that function
                state.init['B_sing']=None if state.entries["init_B_sing_entry"].get()=='None' else float(state.entries["init_B_sing_entry"].get())
                state.minval['B_sing']=None if state.entries["minval_B_sing_entry"].get()=='None' else float(state.entries["minval_B_sing_entry"].get())
                state.maxval['B_sing']=None if state.entries["maxval_B_sing_entry"].get()=='None' else float( state.entries["maxval_B_sing_entry"].get())
     
     
                state.init['A_sing']=None if state.entries["init_A_sing_entry"].get()=='None' else float(state.entries["init_A_sing_entry"].get())
                state.minval['A_sing']=None if state.entries["minval_A_sing_entry"].get()=='None' else float(state.entries["minval_A_sing_entry"].get())
                state.maxval['A_sing']=None if state.entries["maxval_A_sing_entry"].get()=='None' else float(state.entries["maxval_A_sing_entry"].get())
               
                state.init['x0_sing']=None if state.entries["init_x0_sing_entry"].get()=='None' else float(state.entries["init_x0_sing_entry"].get())
                state.minval['x0_sing']=None if state.entries["minval_x0_sing_entry"].get()=='None' else float(state.entries["minval_x0_sing_entry"].get())
                state.maxval['x0_sing']=None if state.entries["maxval_x0_sing_entry"].get()=='None' else float( state.entries["maxval_x0_sing_entry"].get())
    
    
                state.init['dx_sing']=None if state.entries["init_dx_sing_entry"].get()=='None' else float(state.entries["init_dx_sing_entry"].get())
                state.minval['dx_sing']=None if state.entries["minval_dx_sing_entry"].get()=='None' else float(state.entries["minval_dx_sing_entry"].get())
                state.maxval['dx_sing']=None if state.entries["maxval_dx_sing_entry"].get()=='None' else float(state.entries["maxval_dx_sing_entry"].get())


            if state.kappa_pres==1:#if kappa function present, save parameter options from the gui for that function

    
                state.init['A_k']=None if state.entries["init_A_k_entry"].get()=='None' else float(state.entries["init_A_k_entry"].get())
                state.minval['A_k']=None if state.entries["minval_A_k_entry"].get()=='None' else float(state.entries["minval_A_k_entry"].get())
                state.maxval['A_k']=None if state.entries["maxval_A_k_entry"].get()=='None' else float(state.entries["maxval_A_k_entry"].get())
                
                state.init['T_k']=None if state.entries["init_T_k_entry"].get()=='None' else float(state.entries["init_T_k_entry"].get())
                state.minval['T_k']=None if state.entries["minval_T_k_entry"].get()=='None' else float(state.entries["minval_T_k_entry"].get())
                state.maxval['T_k']=None if state.entries["maxval_T_k_entry"].get()=='None' else float( state.entries["maxval_T_k_entry"].get())
                
                state.init['m_i']=None if state.entries["init_m_i_entry"].get()=='None' else float(state.entries["init_m_i_entry"].get())
                state.minval['m_i']=None if state.entries["minval_m_i_entry"].get()=='None' else float(state.entries["minval_m_i_entry"].get())
                state.maxval['m_i']=None if state.entries["maxval_m_i_entry"].get()=='None' else float( state.entries["maxval_m_i_entry"].get())
                
                state.init['n_i']=None if state.entries["init_n_i_entry"].get()=='None' else float(state.entries["init_n_i_entry"].get())
                state.minval['n_i']=None if state.entries["minval_n_i_entry"].get()=='None' else float(state.entries["minval_n_i_entry"].get())
                state.maxval['n_i']=None if state.entries["maxval_n_i_entry"].get()=='None' else float( state.entries["maxval_n_i_entry"].get())    
                
                state.init['kappa']=None if state.entries["init_kappa_entry"].get()=='None' else float(state.entries["init_kappa_entry"].get())
                state.minval['kappa']=None if state.entries["minval_kappa_entry"].get()=='None' else float(state.entries["minval_kappa_entry"].get())
                state.maxval['kappa']=None if state.entries["maxval_kappa_entry"].get()=='None' else float( state.entries["maxval_kappa_entry"].get())
               
            if state.bpl_and_therm_pres==1:
                state.init['T_c']=None if state.entries["init_T_c_entry"].get()=='None' else float(state.entries["init_T_c_entry"].get())
                state.minval['T_c']=None if state.entries["minval_T_c_entry"].get()=='None' else float(state.entries["minval_T_c_entry"].get())
                state.maxval['T_c']=None if state.entries["maxval_T_c_entry"].get()=='None' else float(state.entries["maxval_T_c_entry"].get())
                
                state.init['amp_c']=None if state.entries["init_amp_c_entry"].get()=='None' else float(state.entries["init_amp_c_entry"].get())
                state.minval['amp_c']=None if state.entries["minval_amp_c_entry"].get()=='None' else float(state.entries["minval_amp_c_entry"].get())
                state.maxval['amp_c']=None if state.entries["maxval_amp_c_entry"].get()=='None' else float(state.entries["maxval_amp_c_entry"].get())
                
                state.init['alpha_c']=None if state.entries["init_alpha_c_entry"].get()=='None' else float(state.entries["init_alpha_c_entry"].get())
                state.minval['alpha_c']=None if state.entries["minval_alpha_c_entry"].get()=='None' else float(state.entries["minval_alpha_c_entry"].get())
                state.maxval['alpha_c']=None if state.entries["maxval_alpha_c_entry"].get()=='None' else float(state.entries["maxval_alpha_c_entry"].get())
                
                state.init['x1_c']=None if state.entries["init_x1_c_entry"].get()=='None' else float(state.entries["init_x1_c_entry"].get())
                state.minval['x1_c']=None if state.entries["minval_x1_c_entry"].get()=='None' else float(state.entries["minval_x1_c_entry"].get())
                state.maxval['x1_c']=None if state.entries["maxval_x1_c_entry"].get()=='None' else float(state.entries["maxval_x1_c_entry"].get())
                
                state.init['x0_c']=None if state.entries["init_x0_c_entry"].get()=='None' else float(state.entries["init_x0_c_entry"].get())
                state.minval['x0_c']=None if state.entries["minval_x0_c_entry"].get()=='None' else float(state.entries["minval_x0_c_entry"].get())
                state.maxval['x0_c']=None if state.entries["maxval_x0_c_entry"].get()=='None' else float(state.entries["maxval_x0_c_entry"].get())
                
                state.init['B2_c']=None if state.entries["init_B2_c_entry"].get()=='None' else float(state.entries["init_B2_c_entry"].get())
                state.minval['B2_c']=None if state.entries["minval_B2_c_entry"].get()=='None' else float(state.entries["minval_B2_c_entry"].get())
                state.maxval['B2_c']=None if state.entries["maxval_B2_c_entry"].get()=='None' else float(state.entries["maxval_B2_c_entry"].get())
                
                
                state.init['B_c']=None if state.entries["init_B_c_entry"].get()=='None' else float(state.entries["init_B_c_entry"].get())
                state.minval['B_c']=None if state.entries["minval_B_c_entry"].get()=='None' else float(state.entries["minval_B_c_entry"].get())
                state.maxval['B_c']=None if state.entries["maxval_B_c_entry"].get()=='None' else float( state.entries["maxval_B_c_entry"].get())
                
                
                
            if state.double_therm_func_pres==1:#if double thermal function present, save parameter options from the gui for that function
                
                global frame_double_therm
                
                state.init['T_d_1']=None if state.entries["init_T_d_1_entry"].get()=='None' else float(state.entries["init_T_d_1_entry"].get())
                state.minval['T_d_1']=None if state.entries["minval_T_d_1_entry"].get()=='None' else float(state.entries["minval_T_d_1_entry"].get())
                state.maxval['T_d_1']=None if state.entries["maxval_T_d_1_entry"].get()=='None' else float(state.entries["maxval_T_d_1_entry"].get())
                
                state.init['amp_d_1']=None if state.entries["init_amp_d_1_entry"].get()=='None' else float(state.entries["init_amp_d_1_entry"].get())
                state.minval['amp_d_1']=None if state.entries["minval_amp_d_1_entry"].get()=='None' else float(state.entries["minval_amp_d_1_entry"].get())
                state.maxval['amp_d_1']=None if state.entries["maxval_amp_d_1_entry"].get()=='None' else float(state.entries["maxval_amp_d_1_entry"].get())
                
                state.init['alpha_d_1']=None if state.entries["init_alpha_d_1_entry"].get()=='None' else float(state.entries["init_alpha_d_1_entry"].get())
                state.minval['alpha_d_1']=None if state.entries["minval_alpha_d_1_entry"].get()=='None' else float(state.entries["minval_alpha_d_1_entry"].get())
                state.maxval['alpha_d_1']=None if state.entries["maxval_alpha_d_1_entry"].get()=='None' else float(state.entries["maxval_alpha_d_1_entry"].get())

                state.init['T_d_2']=None if state.entries["init_T_d_2_entry"].get()=='None' else float(state.entries["init_T_d_2_entry"].get())
                state.minval['T_d_2']=None if state.entries["minval_T_d_2_entry"].get()=='None' else float(state.entries["minval_T_d_2_entry"].get())
                state.maxval['T_d_2']=None if state.entries["maxval_T_d_2_entry"].get()=='None' else float(state.entries["maxval_T_d_2_entry"].get())
                
                state.init['amp_d_2']=None if state.entries["init_amp_d_2_entry"].get()=='None' else float(state.entries["init_amp_d_2_entry"].get())
                state.minval['amp_d_2']=None if state.entries["minval_amp_d_2_entry"].get()=='None' else float(state.entries["minval_amp_d_2_entry"].get())
                state.maxval['amp_d_2']=None if state.entries["maxval_amp_d_2_entry"].get()=='None' else float(state.entries["maxval_amp_d_2_entry"].get())
                
                state.init['alpha_d_2']=None if state.entries["init_alpha_d_2_entry"].get()=='None' else float(state.entries["init_alpha_d_2_entry"].get())
                state.minval['alpha_d_2']=None if state.entries["minval_alpha_d_2_entry"].get()=='None' else float(state.entries["minval_alpha_d_2_entry"].get())
                state.maxval['alpha_d_2']=None if state.entries["maxval_alpha_d_2_entry"].get()=='None' else float(state.entries["maxval_alpha_d_2_entry"].get())
            
            
            if state.tpl_pres==1:#if tpl function present, save parameter options from the gui for that function
                
                
                
                state.init['x1_tpl']=None if state.entries["init_x1_tpl_entry"].get()=='None' else float(state.entries["init_x1_tpl_entry"].get())
                state.minval['x1_tpl']=None if state.entries["minval_x1_tpl_entry"].get()=='None' else float(state.entries["minval_x1_tpl_entry"].get())
                state.maxval['x1_tpl']=None if state.entries["maxval_x1_tpl_entry"].get()=='None' else float(state.entries["maxval_x1_tpl_entry"].get())
                
                                
                state.init['x2_tpl']=None if state.entries["init_x2_tpl_entry"].get()=='None' else float(state.entries["init_x2_tpl_entry"].get())
                state.minval['x2_tpl']=None if state.entries["minval_x2_tpl_entry"].get()=='None' else float(state.entries["minval_x2_tpl_entry"].get())
                state.maxval['x2_tpl']=None if state.entries["maxval_x2_tpl_entry"].get()=='None' else float(state.entries["maxval_x2_tpl_entry"].get())
                
                
                state.init['A2_tpl']=None if state.entries["init_A2_tpl_entry"].get()=='None' else float(state.entries["init_A2_tpl_entry"].get())
                state.minval['A2_tpl']=None if state.entries["minval_A2_tpl_entry"].get()=='None' else float(state.entries["minval_A2_tpl_entry"].get())
                state.maxval['A2_tpl']=None if state.entries["maxval_A2_tpl_entry"].get()=='None' else float(state.entries["maxval_A2_tpl_entry"].get())
                
                
                state.init['B2_tpl']=None if state.entries["init_B2_tpl_entry"].get()=='None' else float(state.entries["init_B2_tpl_entry"].get())
                state.minval['B2_tpl']=None if state.entries["minval_B2_tpl_entry"].get()=='None' else float(state.entries["minval_B2_tpl_entry"].get())
                state.maxval['B2_tpl']=None if state.entries["maxval_B2_tpl_entry"].get()=='None' else float(state.entries["maxval_B2_tpl_entry"].get())
                
                
                state.init['B_tpl']=None if state.entries["init_B_tpl_entry"].get()=='None' else float(state.entries["init_B_tpl_entry"].get())
                state.minval['B_tpl']=None if state.entries["minval_B_tpl_entry"].get()=='None' else float(state.entries["minval_B_tpl_entry"].get())
                state.maxval['B_tpl']=None if state.entries["maxval_B_tpl_entry"].get()=='None' else float( state.entries["maxval_B_tpl_entry"].get())
    
    
                state.init['A_tpl']=None if state.entries["init_A_tpl_entry"].get()=='None' else float(state.entries["init_A_tpl_entry"].get())
                state.minval['A_tpl']=None if state.entries["minval_A_tpl_entry"].get()=='None' else float(state.entries["minval_A_tpl_entry"].get())
                state.maxval['A_tpl']=None if state.entries["maxval_A_tpl_entry"].get()=='None' else float(state.entries["maxval_A_tpl_entry"].get())
                
                state.init['A3_tpl']=None if state.entries["init_A3_tpl_entry"].get()=='None' else float(state.entries["init_A3_tpl_entry"].get())
                state.minval['A3_tpl']=None if state.entries["minval_A3_tpl_entry"].get()=='None' else float(state.entries["minval_A3_tpl_entry"].get())
                state.maxval['A3_tpl']=None if state.entries["maxval_A3_tpl_entry"].get()=='None' else float(state.entries["maxval_A3_tpl_entry"].get())
                
                
                state.init['B3_tpl']=None if state.entries["init_B3_tpl_entry"].get()=='None' else float(state.entries["init_B3_tpl_entry"].get())
                state.minval['B3_tpl']=None if state.entries["minval_B3_tpl_entry"].get()=='None' else float(state.entries["minval_B3_tpl_entry"].get())
                state.maxval['B3_tpl']=None if state.entries["maxval_B3_tpl_entry"].get()=='None' else float(state.entries["maxval_B3_tpl_entry"].get())
                
                state.init['x0_tpl']=None if state.entries["init_x0_tpl_entry"].get()=='None' else float(state.entries["init_x0_tpl_entry"].get())
                state.minval['x0_tpl']=None if state.entries["minval_x0_tpl_entry"].get()=='None' else float(state.entries["minval_x0_tpl_entry"].get())
                state.maxval['x0_tpl']=None if state.entries["maxval_x0_tpl_entry"].get()=='None' else float( state.entries["maxval_x0_tpl_entry"].get())
    
    
                state.init['dx_tpl']=None if state.entries["init_dx_tpl_entry"].get()=='None' else float(state.entries["init_dx_tpl_entry"].get())
                state.minval['dx_tpl']=None if state.entries["minval_dx_tpl_entry"].get()=='None' else float(state.entries["minval_dx_tpl_entry"].get())
                state.maxval['dx_tpl']=None if state.entries["maxval_dx_tpl_entry"].get()=='None' else float(state.entries["maxval_dx_tpl_entry"].get())
            
            if state.qpl_pres==1:#if qpl function present, save parameter options from the gui for that function
                state.init['x1_qpl']=None if state.entries["init_x1_qpl_entry"].get()=='None' else float(state.entries["init_x1_qpl_entry"].get())
                state.minval['x1_qpl']=None if state.entries["minval_x1_qpl_entry"].get()=='None' else float(state.entries["minval_x1_qpl_entry"].get())
                state.maxval['x1_qpl']=None if state.entries["maxval_x1_qpl_entry"].get()=='None' else float(state.entries["maxval_x1_qpl_entry"].get())
                
                                
                state.init['x2_qpl']=None if state.entries["init_x2_qpl_entry"].get()=='None' else float(state.entries["init_x2_qpl_entry"].get())
                state.minval['x2_qpl']=None if state.entries["minval_x2_qpl_entry"].get()=='None' else float(state.entries["minval_x2_qpl_entry"].get())
                state.maxval['x2_qpl']=None if state.entries["maxval_x2_qpl_entry"].get()=='None' else float(state.entries["maxval_x2_qpl_entry"].get())
                
                state.init['x3_qpl']=None if state.entries["init_x3_qpl_entry"].get()=='None' else float(state.entries["init_x3_qpl_entry"].get())
                state.minval['x3_qpl']=None if state.entries["minval_x3_qpl_entry"].get()=='None' else float(state.entries["minval_x3_qpl_entry"].get())
                state.maxval['x3_qpl']=None if state.entries["maxval_x3_qpl_entry"].get()=='None' else float(state.entries["maxval_x3_qpl_entry"].get())
                
                state.init['A2_qpl']=None if state.entries["init_A2_qpl_entry"].get()=='None' else float(state.entries["init_A2_qpl_entry"].get())
                state.minval['A2_qpl']=None if state.entries["minval_A2_qpl_entry"].get()=='None' else float(state.entries["minval_A2_qpl_entry"].get())
                state.maxval['A2_qpl']=None if state.entries["maxval_A2_qpl_entry"].get()=='None' else float(state.entries["maxval_A2_qpl_entry"].get())
                
                
                state.init['B2_qpl']=None if state.entries["init_B2_qpl_entry"].get()=='None' else float(state.entries["init_B2_qpl_entry"].get())
                state.minval['B2_qpl']=None if state.entries["minval_B2_qpl_entry"].get()=='None' else float(state.entries["minval_B2_qpl_entry"].get())
                state.maxval['B2_qpl']=None if state.entries["maxval_B2_qpl_entry"].get()=='None' else float(state.entries["maxval_B2_qpl_entry"].get())
                
                
                state.init['B_qpl']=None if state.entries["init_B_qpl_entry"].get()=='None' else float(state.entries["init_B_qpl_entry"].get())
                state.minval['B_qpl']=None if state.entries["minval_B_qpl_entry"].get()=='None' else float(state.entries["minval_B_qpl_entry"].get())
                state.maxval['B_qpl']=None if state.entries["maxval_B_qpl_entry"].get()=='None' else float( state.entries["maxval_B_qpl_entry"].get())
            
            
                state.init['A_qpl']=None if state.entries["init_A_qpl_entry"].get()=='None' else float(state.entries["init_A_qpl_entry"].get())
                state.minval['A_qpl']=None if state.entries["minval_A_qpl_entry"].get()=='None' else float(state.entries["minval_A_qpl_entry"].get())
                state.maxval['A_qpl']=None if state.entries["maxval_A_qpl_entry"].get()=='None' else float(state.entries["maxval_A_qpl_entry"].get())
                
                state.init['A3_qpl']=None if state.entries["init_A3_qpl_entry"].get()=='None' else float(state.entries["init_A3_qpl_entry"].get())
                state.minval['A3_qpl']=None if state.entries["minval_A3_qpl_entry"].get()=='None' else float(state.entries["minval_A3_qpl_entry"].get())
                state.maxval['A3_qpl']=None if state.entries["maxval_A3_qpl_entry"].get()=='None' else float(state.entries["maxval_A3_qpl_entry"].get())
                
                
                state.init['B3_qpl']=None if state.entries["init_B3_qpl_entry"].get()=='None' else float(state.entries["init_B3_qpl_entry"].get())
                state.minval['B3_qpl']=None if state.entries["minval_B3_qpl_entry"].get()=='None' else float(state.entries["minval_B3_qpl_entry"].get())
                state.maxval['B3_qpl']=None if state.entries["maxval_B3_qpl_entry"].get()=='None' else float(state.entries["maxval_B3_qpl_entry"].get())
                
                state.init['A4_qpl']=None if state.entries["init_A4_qpl_entry"].get()=='None' else float(state.entries["init_A4_qpl_entry"].get())
                state.minval['A4_qpl']=None if state.entries["minval_A4_qpl_entry"].get()=='None' else float(state.entries["minval_A4_qpl_entry"].get())
                state.maxval['A4_qpl']=None if state.entries["maxval_A4_qpl_entry"].get()=='None' else float(state.entries["maxval_A4_qpl_entry"].get())
                
                state.init['B4_qpl']=None if state.entries["init_B4_qpl_entry"].get()=='None' else float(state.entries["init_B4_qpl_entry"].get())
                state.minval['B4_qpl']=None if state.entries["minval_B4_qpl_entry"].get()=='None' else float(state.entries["minval_B4_qpl_entry"].get())
                state.maxval['B4_qpl']=None if state.entries["maxval_B4_qpl_entry"].get()=='None' else float(state.entries["maxval_B4_qpl_entry"].get())
                
                state.init['x0_qpl']=None if state.entries["init_x0_qpl_entry"].get()=='None' else float(state.entries["init_x0_qpl_entry"].get())
                state.minval['x0_qpl']=None if state.entries["minval_x0_qpl_entry"].get()=='None' else float(state.entries["minval_x0_qpl_entry"].get())
                state.maxval['x0_qpl']=None if state.entries["maxval_x0_qpl_entry"].get()=='None' else float( state.entries["maxval_x0_qpl_entry"].get())
    
    
                state.init['dx_qpl']=None if state.entries["init_dx_qpl_entry"].get()=='None' else float(state.entries["init_dx_qpl_entry"].get())
                state.minval['dx_qpl']=None if state.entries["minval_dx_qpl_entry"].get()=='None' else float(state.entries["minval_dx_qpl_entry"].get())
                state.maxval['dx_qpl']=None if state.entries["maxval_dx_qpl_entry"].get()=='None' else float(state.entries["maxval_dx_qpl_entry"].get())
                
                
            if state.quint_pl_pres==1:#if 5pl function present, save parameter options from the gui for that function
                state.init['x1_5pl']=None if state.entries["init_x1_5pl_entry"].get()=='None' else float(state.entries["init_x1_5pl_entry"].get())
                state.minval['x1_5pl']=None if state.entries["minval_x1_5pl_entry"].get()=='None' else float(state.entries["minval_x1_5pl_entry"].get())
                state.maxval['x1_5pl']=None if state.entries["maxval_x1_5pl_entry"].get()=='None' else float(state.entries["maxval_x1_5pl_entry"].get())
                
                                
                state.init['x2_5pl']=None if state.entries["init_x2_5pl_entry"].get()=='None' else float(state.entries["init_x2_5pl_entry"].get())
                state.minval['x2_5pl']=None if state.entries["minval_x2_5pl_entry"].get()=='None' else float(state.entries["minval_x2_5pl_entry"].get())
                state.maxval['x2_5pl']=None if state.entries["maxval_x2_5pl_entry"].get()=='None' else float(state.entries["maxval_x2_5pl_entry"].get())
                
                state.init['x3_5pl']=None if state.entries["init_x3_5pl_entry"].get()=='None' else float(state.entries["init_x3_5pl_entry"].get())
                state.minval['x3_5pl']=None if state.entries["minval_x3_5pl_entry"].get()=='None' else float(state.entries["minval_x3_5pl_entry"].get())
                state.maxval['x3_5pl']=None if state.entries["maxval_x3_5pl_entry"].get()=='None' else float(state.entries["maxval_x3_5pl_entry"].get())
                
                state.init['x4_5pl']=None if state.entries["init_x4_5pl_entry"].get()=='None' else float(state.entries["init_x4_5pl_entry"].get())
                state.minval['x4_5pl']=None if state.entries["minval_x4_5pl_entry"].get()=='None' else float(state.entries["minval_x4_5pl_entry"].get())
                state.maxval['x4_5pl']=None if state.entries["maxval_x4_5pl_entry"].get()=='None' else float(state.entries["maxval_x4_5pl_entry"].get())
                
                state.init['A2_5pl']=None if state.entries["init_A2_5pl_entry"].get()=='None' else float(state.entries["init_A2_5pl_entry"].get())
                state.minval['A2_5pl']=None if state.entries["minval_A2_5pl_entry"].get()=='None' else float(state.entries["minval_A2_5pl_entry"].get())
                state.maxval['A2_5pl']=None if state.entries["maxval_A2_5pl_entry"].get()=='None' else float(state.entries["maxval_A2_5pl_entry"].get())
                
                
                state.init['B2_5pl']=None if state.entries["init_B2_5pl_entry"].get()=='None' else float(state.entries["init_B2_5pl_entry"].get())
                state.minval['B2_5pl']=None if state.entries["minval_B2_5pl_entry"].get()=='None' else float(state.entries["minval_B2_5pl_entry"].get())
                state.maxval['B2_5pl']=None if state.entries["maxval_B2_5pl_entry"].get()=='None' else float(state.entries["maxval_B2_5pl_entry"].get())
                
                
                state.init['B_5pl']=None if state.entries["init_B_5pl_entry"].get()=='None' else float(state.entries["init_B_5pl_entry"].get())
                state.minval['B_5pl']=None if state.entries["minval_B_5pl_entry"].get()=='None' else float(state.entries["minval_B_5pl_entry"].get())
                state.maxval['B_5pl']=None if state.entries["maxval_B_5pl_entry"].get()=='None' else float( state.entries["maxval_B_5pl_entry"].get())
            
            
                state.init['A_5pl']=None if state.entries["init_A_5pl_entry"].get()=='None' else float(state.entries["init_A_5pl_entry"].get())
                state.minval['A_5pl']=None if state.entries["minval_A_5pl_entry"].get()=='None' else float(state.entries["minval_A_5pl_entry"].get())
                state.maxval['A_5pl']=None if state.entries["maxval_A_5pl_entry"].get()=='None' else float(state.entries["maxval_A_5pl_entry"].get())
                
                state.init['A3_5pl']=None if state.entries["init_A3_5pl_entry"].get()=='None' else float(state.entries["init_A3_5pl_entry"].get())
                state.minval['A3_5pl']=None if state.entries["minval_A3_5pl_entry"].get()=='None' else float(state.entries["minval_A3_5pl_entry"].get())
                state.maxval['A3_5pl']=None if state.entries["maxval_A3_5pl_entry"].get()=='None' else float(state.entries["maxval_A3_5pl_entry"].get())
                
                
                state.init['B3_5pl']=None if state.entries["init_B3_5pl_entry"].get()=='None' else float(state.entries["init_B3_5pl_entry"].get())
                state.minval['B3_5pl']=None if state.entries["minval_B3_5pl_entry"].get()=='None' else float(state.entries["minval_B3_5pl_entry"].get())
                state.maxval['B3_5pl']=None if state.entries["maxval_B3_5pl_entry"].get()=='None' else float(state.entries["maxval_B3_5pl_entry"].get())
                
                state.init['A4_5pl']=None if state.entries["init_A4_5pl_entry"].get()=='None' else float(state.entries["init_A4_5pl_entry"].get())
                state.minval['A4_5pl']=None if state.entries["minval_A4_5pl_entry"].get()=='None' else float(state.entries["minval_A4_5pl_entry"].get())
                state.maxval['A4_5pl']=None if state.entries["maxval_A4_5pl_entry"].get()=='None' else float(state.entries["maxval_A4_5pl_entry"].get())
                
                state.init['B4_5pl']=None if state.entries["init_B4_5pl_entry"].get()=='None' else float(state.entries["init_B4_5pl_entry"].get())
                state.minval['B4_5pl']=None if state.entries["minval_B4_5pl_entry"].get()=='None' else float(state.entries["minval_B4_5pl_entry"].get())
                state.maxval['B4_5pl']=None if state.entries["maxval_B4_5pl_entry"].get()=='None' else float(state.entries["maxval_B4_5pl_entry"].get())               
                
                state.init['A5_5pl']=None if state.entries["init_A5_5pl_entry"].get()=='None' else float(state.entries["init_A5_5pl_entry"].get())
                state.minval['A5_5pl']=None if state.entries["minval_A5_5pl_entry"].get()=='None' else float(state.entries["minval_A5_5pl_entry"].get())
                state.maxval['A5_5pl']=None if state.entries["maxval_A5_5pl_entry"].get()=='None' else float(state.entries["maxval_A5_5pl_entry"].get())
                
                state.init['B5_5pl']=None if state.entries["init_B5_5pl_entry"].get()=='None' else float(state.entries["init_B5_5pl_entry"].get())
                state.minval['B5_5pl']=None if state.entries["minval_B5_5pl_entry"].get()=='None' else float(state.entries["minval_B5_5pl_entry"].get())
                state.maxval['B5_5pl']=None if state.entries["maxval_B5_5pl_entry"].get()=='None' else float(state.entries["maxval_B5_5pl_entry"].get())  
                            
                state.init['x0_5pl']=None if state.entries["init_x0_5pl_entry"].get()=='None' else float(state.entries["init_x0_5pl_entry"].get())
                state.minval['x0_5pl']=None if state.entries["minval_x0_5pl_entry"].get()=='None' else float(state.entries["minval_x0_5pl_entry"].get())
                state.maxval['x0_5pl']=None if state.entries["maxval_x0_5pl_entry"].get()=='None' else float( state.entries["maxval_x0_5pl_entry"].get())
    
    
                state.init['dx_5pl']=None if state.entries["init_dx_5pl_entry"].get()=='None' else float(state.entries["init_dx_5pl_entry"].get())
                state.minval['dx_5pl']=None if state.entries["minval_dx_5pl_entry"].get()=='None' else float(state.entries["minval_dx_5pl_entry"].get())
                state.maxval['dx_5pl']=None if state.entries["maxval_dx_5pl_entry"].get()=='None' else float(state.entries["maxval_dx_5pl_entry"].get())
                
                
            #pull the min/max energy (x) values to fit to
            state.fitmin=float(fitmin_entry.get())
            state.fitmax=float(fitmax_entry.get())
            #validate limits
            if not validate_lims(state.fitmin,state.fitmax):
                tk.messagebox.showerror("Invalid Input","Fit limits should be floats with max greater than min")
            else:
                
                #validate entries
                validity=dict()
                for ind in state.minval.keys():
                    min_val=state.minval[ind]
                    max_val=state.maxval[ind]
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
                    for ind in state.init.keys():
                        min_val=state.minval[ind]
                        max_val=state.maxval[ind]
                        init_val=state.init[ind]
                        validity[ind]=validate_init(init_val,min_val,max_val)
                    if False in validity:#show where error is !!!!!
                        false_keys=list()            
                        for key, value in validity.items():
                            if value is False:
                                false_keys.append(key)
                        
                        tk.messagebox.showerror("Invalid Input",f"Parameter initial values should be floats between their max and min values for parameter(s) {false_keys}")                
        
        
        
        
        
        
        except ValueError:
               tk.messagebox.showerror("Invalid Input","inputs should be floating point intergers preview")
        param_preview(x_data,y_data,state.init,state.header)#calls previously defined save function
    

    
    
    #option to save the spectrum
    def spec_save_hndl():
        #organise into dataframe
        spec_dict={'energies':list(x_data) ,'fluxes': list(y_data),'errors':list(uncert),'date':[str(date) for i in list(x_data)],'inst':[inst for i in list(x_data)],'spec_type':[spec_type for i in list(x_data)]}
    
        spec_frame=pd.DataFrame(spec_dict)
        files = [('Text Document','*.txt')]
        file_obj=tk.filedialog.asksaveasfile(filetypes = files, defaultextension=".txt")
        if file_obj is None:  #user cancelled the dialog
            return
        spec_frame.to_csv(file_obj)

    
    
    def close_btn_hndl():
        try: window_buttons.destroy()
        except tk.TclError: pass
        if state.fit_window != None:
            try: state.fit_window.destroy()
            except tk.TclError: pass
            state.fit_window=None
    



#%%load savehandling
    def save_btn_hndl():#function to handle save button
        param_save(date,inst,spec_type, state.bpl_pres, state.therm_func_pres, state.gauss_pres, state.power_pres, state.kappa_pres,state.bpl_and_therm_pres, state.double_therm_func_pres, state.tpl_pres, state.qpl_pres,state.quint_pl_pres,state.redchi)#calls previously defined save function
    #when loading, must adjust bounds to fit loaded pars

    def widen_bounds_for_loaded_value(name, loaded_val, margin_frac=0.2, abs_floor=1e-6):
        """
        Ensure a value loaded from a previous fit sits safely INSIDE the current
        min/max bounds (not on the edge), widening whichever bound needs it.
        Bounds are only ever widened, never narrowed, and both state.minval/maxval
        and the visible entry boxes are updated so the change is visible and the
        user can override it if they don't want it.
        """
        cur_min = state.minval[name]
        cur_max = state.maxval[name]
        pad = max(abs(loaded_val) * margin_frac, abs_floor)
    
        new_min = min(cur_min, loaded_val - pad)
        new_max = max(cur_max, loaded_val + pad)
    
        if new_min != cur_min or new_max != cur_max:
            state.minval[name] = new_min
            state.maxval[name] = new_max
    
            min_entry = state.entries[f"minval_{name}_entry"]
            min_entry.delete(0, tk.END)
            min_entry.insert(0, str(new_min))
    
            max_entry = state.entries[f"maxval_{name}_entry"]
            max_entry.delete(0, tk.END)
            max_entry.insert(0, str(new_max))
    
            print(f"Widened bounds for '{name}' to [{new_min:.4g}, {new_max:.4g}] "
                  f"to fit loaded value {loaded_val:.4g}")


    #loading each param
    def load_param(name, value):
        """Populate the init entry for `name`, widening bounds if needed so the
        loaded value can't be silently clamped."""
        entry = state.entries[f"init_{name}_entry"]
        entry.delete(0, tk.END)
        entry.insert(0, value)
        widen_bounds_for_loaded_value(name, value)

    def load_btn_hndl():#function to handle load button
        state.header,parvals_ld=param_load(date,inst,spec_type)
        
        
        if state.header[9] == '1':  # bpl
           add_bpl()
           for name in ["x1", "A", "B", "A2", "B2", "x0_bpl", "dx_bpl"]:
               load_param(name, parvals_ld[name])
    
        if state.header[28] == '1':  # therm
           add_therm()
           for name in ["amp", "T", "alpha"]:
               load_param(name, parvals_ld[name])
    
        if state.header[42] == '1':  # gauss
           add_gauss()
           for name in ["gauss_amp", "gauss_centre", "sigma"]:
               load_param(name, parvals_ld[name])
    
        if state.header[56] == '1':  # single power law
           add_power()
           for name in ["A_sing", "B_sing", "x0_sing", "dx_sing"]:
               load_param(name, parvals_ld[name])
    
        if state.header[70] == '1':  # kappa
           add_kappa()
           for name in ["A_k", "T_k", "m_i", "n_i", "kappa"]:
               load_param(name, parvals_ld[name])
    
        if state.header[92] == '1':  # combined bpl + thermal
           add_bpl_and_therm()
           for name in ["amp_c", "T_c", "alpha_c", "x0_c", "x1_c", "B_c", "B2_c"]:
               load_param(name, parvals_ld[name])
    
        if state.header[118] == '1':  # double thermal
           add_double_therm()
           for name in ["amp_d_1", "T_d_1", "alpha_d_1", "amp_d_2", "T_d_2", "alpha_d_2"]:
               load_param(name, parvals_ld[name])
    
        if state.header[130] == '1':  # triple power law
           add_tpl()
           for name in ["x1_tpl", "x2_tpl", "B_tpl", "B2_tpl", "B3_tpl",
                        "A_tpl", "A2_tpl", "A3_tpl", "x0_tpl", "dx_tpl"]:
               load_param(name, parvals_ld[name])
    
        if state.header[142] == '1':  # quad power law
           add_qpl()
           for name in ["x1_qpl", "x2_qpl", "x3_qpl", "B_qpl", "B2_qpl", "B3_qpl", "B4_qpl",
                        "A_qpl", "A2_qpl", "A3_qpl", "A4_qpl", "x0_qpl", "dx_qpl"]:
               load_param(name, parvals_ld[name])
    
        if state.header[159] == '1':  # quintuple power law
           add_quint_pl()
           for name in ["x1_5pl", "x2_5pl", "x3_5pl", "x4_5pl",
                        "B_5pl", "B2_5pl", "B3_5pl", "B4_5pl", "B5_5pl",
                        "A_5pl", "A2_5pl", "A3_5pl", "A4_5pl", "A5_5pl",
                        "x0_5pl", "dx_5pl"]:
               load_param(name, parvals_ld[name])
                   
    def fit_sum_hndl():
        try:
            state.fit_summary
            
        except NameError:tk.messagebox.showwarning("No Results", "No uncertainties and statistics yet, run a fit first")

        
        else:
            summ_window=tk.Toplevel()
            summ_window.title("Parameter Uncertainty Summary")
            summary_text=tk.Message(summ_window, text=state.fit_summary)
            summary_text.pack(padx=10, pady=10)
            
            tk.Label(summ_window, text="Absolute Uncertainties").pack()
            uncerts_text=tk.Message(summ_window, text=state.param_uncert_calced)
            uncerts_text.pack(padx=10, pady=10)
            
            

            
            summ_window.mainloop()
     
    def resid_save_hndl():
        try:
            state.fit_summary
            
        except NameError:tk.messagebox.showwarning("No Results", "No residuals yet, run a fit first")

        
        else:
            #organise into dataframe
            spec_dict={'energies':list(x_data_E_sliced) ,'state.resids': list(state.resids),'date':[str(date) for i in list(x_data_E_sliced)],'inst':[inst for i in list(x_data_E_sliced)],'spec_type':[spec_type for i in list(x_data_E_sliced)]}
            spec_frame=pd.DataFrame(spec_dict)
            files = [('Text Document','*.txt')]
            file_obj=tk.filedialog.asksaveasfile(filetypes = files, defaultextension=".txt")
            if file_obj is None:  #user cancelled the dialog
                return
            spec_frame.to_csv(file_obj)

            
    def on_selection(event):
        selection = combo.get()
        option_handlers={"Load Parameters":load_btn_hndl,
                        "Save Parameters":save_btn_hndl,
                        "Close (and proceed to next interval if set)":close_btn_hndl,
                        "Save Spectrum":spec_save_hndl,
                        "Preview Parameters":preview_btn_hndl,
                        "Perform Fit":fit_btn_hndl,
                        "Summary of fit statistics and uncerainties":fit_sum_hndl,
                        "Save residuals":resid_save_hndl}
        option_handlers[selection]()
        
    fit_window_options=["Load Parameters","Save Parameters","Close (and proceed to next interval if set)","Save Spectrum","Save residuals","Preview Parameters","Perform Fit","Summary of fit statistics and uncerainties"]
    combo=ttk.Combobox(window_buttons, values=fit_window_options)
    combo.bind('<<ComboboxSelected>>', on_selection)
    combo.pack()
    
    window_buttons.mainloop()#this creates the gui window as defined above