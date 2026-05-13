#%%Initial set up
import sys#for file path handling
import os#has general functions for file manipulation

from . import param_preview,param_load,param_save


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

        
        globals()[f"btn_vary_{name_prefix}"] = tk.Checkbutton(frame, text=f"Vary {label_text}", command=callback, variable=tk.IntVar())
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
            init['x0_bpl']=1
            init['dx_bpl']=0.1    

    
            global vary#define global if vary values for the params of the function

            vary['x1']=True
            vary['A']=True
            vary['B']=True
            vary['A2']=True
            vary['B2']=True            
            vary['x0_bpl']=True
            vary['dx_bpl']=True
            
            global maxval##define global maximum values for the params of the function

            maxval['x1']=50
            maxval['A']=None
            maxval['B']=0
            maxval['A2']=None
            maxval['B2']=0            
            maxval['x0_bpl']=10
            maxval['dx_bpl']=1


            
            global minval##define global minimum values for the params of the function

            minval['x1']=15
            minval['A']=0
            minval['B']=-10
            minval['A2']=0
            minval['B2']=-10            
            minval['x0_bpl']=-1
            minval['dx_bpl']=0.01



            global frame_bpl#defining gui section to handle bpl param options
            frame_bpl=tk.Frame(master=frame_params)
            
            tk.Label(frame_bpl, text="Broken Power Law", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_A(): vary['A'] = not vary['A']
            def toggle_A2(): vary['A2'] = not vary['A2']
            def toggle_x1(): vary['x1'] = not vary['x1']
            def toggle_B(): vary['B'] = not vary['B']
            def toggle_B2(): vary['B2'] = not vary['B2']
            def toggle_x0_bpl(): vary['x0_bpl'] = not vary['x0_bpl']
            def toggle_dx_bpl(): vary['dx_bpl'] = not vary['dx_bpl']

            add_param_row(frame_bpl, 1, "x1", init['x1'], minval['x1'], maxval['x1'], vary['x1'], toggle_x1, "x1")
            add_param_row(frame_bpl, 2, "A", init['A'], minval['A'], maxval['A'], vary['A'], toggle_A, "A")           
            add_param_row(frame_bpl, 3, "B", init['B'], minval['B'], maxval['B'], vary['B'], toggle_B, "B")
            add_param_row(frame_bpl, 4, "A2", init['A2'], minval['A2'], maxval['A2'], vary['A2'], toggle_A2, "A2")
            add_param_row(frame_bpl, 5, "B2", init['B2'], minval['B2'], maxval['B2'], vary['B2'], toggle_B2, "B2")
            add_param_row(frame_bpl, 6, "x0_bpl", init['x0_bpl'], minval['x0_bpl'], maxval['x0_bpl'], vary['x0_bpl'], toggle_x0_bpl, "x0_bpl")
            add_param_row(frame_bpl, 7, "dx_bpl", init['dx_bpl'], minval['dx_bpl'], maxval['dx_bpl'], vary['dx_bpl'], toggle_dx_bpl, "dx_bpl")

            def hndl_remove_bpl_btn():
                global bpl_pres
                frame_bpl.grid_forget()
                bpl_pres = 0

            tk.Button(frame_bpl, text='Remove BPL component', command=hndl_remove_bpl_btn)\
                .grid(row=8, column=0, columnspan=5, pady=10)

            frame_bpl.grid(row=3, column=0, sticky="ew")
            for i in range(8):
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
            init['x0_sing']=1
            init['dx_sing']=0.1    

            
            global vary#define global if vary values for the params of the function
            vary['A_sing']=True
            vary['B_sing']=True
            vary['x0_sing']=True
            vary['dx_sing']=True
            
            global minval##define global minimum values for the params of the function
            minval['A_sing']=0
            minval['B_sing']=None
            minval['x0_sing']=-1
            minval['dx_sing']=0.01

            
            global maxval##define global maximum values for the params of the function
            maxval['A_sing']=None
            maxval['B_sing']=0
            maxval['x0_sing']=10
            maxval['dx_sing']=1

            
            global frame_power
            frame_power=tk.Frame(master=frame_params)
            tk.Label(frame_power, text="Power Law", font=("Arial", 12))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle_A_sing(): vary['A_sing'] = not vary['A_sing']
            def toggle_B_sing(): vary['B_sing'] = not vary['B_sing']
            def toggle_x0_sing(): vary['x0_sing'] = not vary['x0_sing']
            def toggle_dx_sing(): vary['dx_sing'] = not vary['dx_sing']

            add_param_row(frame_power, 1, "A_sing", init['A_sing'], minval['A_sing'], maxval['A_sing'], vary['A_sing'], toggle_A_sing, "A_sing")
            add_param_row(frame_power, 2, "B_sing", init['B_sing'], minval['B_sing'], maxval['B_sing'], vary['B_sing'], toggle_B_sing, "B_sing")
            add_param_row(frame_power, 3, "x0_sing", init['x0_sing'], minval['x0_sing'], maxval['x0_sing'], vary['x0_sing'], toggle_x0_sing, "x0_sing")
            add_param_row(frame_power, 4, "dx_sing", init['dx_sing'], minval['dx_sing'], maxval['dx_sing'], vary['dx_sing'], toggle_dx_sing, "dx_sing")

            def hndl_remove_power_btn():
                global power_pres
                frame_power.grid_forget()
                power_pres = 0

            tk.Button(frame_power, text='Remove Power Law component', command=hndl_remove_power_btn)\
                .grid(row=5, column=0, columnspan=5, pady=10)

            frame_power.grid(row=5, column=0, sticky="ew")
            for i in range(6):
                frame_power.grid_columnconfigure(i, weight=1)
            power_pres = 1
            
    def add_kappa():#function to add kappa law to gui/test function
        
        global kappa_pres#defining gui section to handle kappa law param options
        if kappa_pres ==0:
            
            global init#define global initial values for the params of the function
            
            init['A_k']=10**-20
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
            
            minval['A_k']=1e-22
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
            for i in range(7):
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
            for i in range(9):
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
                global double_therm_func_pres
                frame_double_therm.grid_forget()
                double_therm_func_pres = 0

            tk.Button(frame_double_therm, text='Remove Double Thermal component', command=hndl_remove_double_therm_btn)\
                .grid(row=7, column=0, columnspan=5, pady=10)

            frame_double_therm.grid(row=8, column=0, sticky="ew")
            for i in range(8):
                frame_double_therm.grid_columnconfigure(i, weight=1)
            double_therm_func_pres = 1 #set the thermal function as present
    
    
    
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
            init['x0_tpl']=1
            init['dx_tpl']=0.1  

            global vary#define global if vary values for the params of the function

            vary['x1_tpl']=True
            vary['x2_tpl']=True
            vary['A_tpl']=True
            vary['B_tpl']=True
            vary['A2_tpl']=True
            vary['B2_tpl']=True            
            vary['A3_tpl']=True
            vary['B3_tpl']=True            
            vary['x0_tpl']=True
            vary['dx_tpl']=True
            
            global maxval##define global maximum values for the params of the function

            maxval['x1_tpl']=50
            maxval['x2_tpl']=50
            maxval['A_tpl']=None
            maxval['B_tpl']=0
            maxval['A2_tpl']=None
            maxval['B2_tpl']=0            
            maxval['A3_tpl']=None
            maxval['B3_tpl']=0           
            maxval['x0_tpl']=10
            maxval['dx_tpl']=1


            
            global minval##define global minimum values for the params of the function

            minval['x1_tpl']=5
            minval['x2_tpl']=15
            minval['A_tpl']=0
            minval['B_tpl']=-10
            minval['A2_tpl']=0
            minval['B2_tpl']=-10            
            minval['A3_tpl']=0
            minval['B3_tpl']=-10            
            minval['x0_tpl']=-1
            minval['dx_tpl']=0.01


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
            
            def toggle_x0_tpl(): vary['x0_tpl'] = not vary['x0_tpl']
            def toggle_dx_tpl(): vary['dx_tpl'] = not vary['dx_tpl']


            add_param_row(frame_tpl, 1, "A_tpl", init['A_tpl'], minval['A_tpl'], maxval['A_tpl'], vary['A_tpl'], toggle_A_tpl, "A_tpl")
            add_param_row(frame_tpl, 2, "B_tpl", init['B_tpl'], minval['B_tpl'], maxval['B_tpl'], vary['B_tpl'], toggle_B_tpl, "B_tpl")
            add_param_row(frame_tpl, 3, "x1_tpl", init['x1_tpl'], minval['x1_tpl'], maxval['x1_tpl'], vary['x1_tpl'], toggle_x1_tpl, "x1_tpl")
            add_param_row(frame_tpl, 4, "A2_tpl", init['A2_tpl'], minval['A2_tpl'], maxval['A2_tpl'], vary['A2_tpl'], toggle_A2_tpl, "A2_tpl")
            add_param_row(frame_tpl, 5, "B2_tpl", init['B2_tpl'], minval['B2_tpl'], maxval['B2_tpl'], vary['B2_tpl'], toggle_B2_tpl, "B2_tpl")
            add_param_row(frame_tpl, 6, "x2_tpl", init['x2_tpl'], minval['x2_tpl'], maxval['x2_tpl'], vary['x2_tpl'], toggle_x2_tpl, "x2_tpl")
            add_param_row(frame_tpl, 7, "A3_tpl", init['A3_tpl'], minval['A3_tpl'], maxval['A3_tpl'], vary['A3_tpl'], toggle_A3_tpl, "A3_tpl")
            add_param_row(frame_tpl, 8, "B3_tpl", init['B3_tpl'], minval['B3_tpl'], maxval['B3_tpl'], vary['B3_tpl'], toggle_B3_tpl, "B3_tpl")
            add_param_row(frame_tpl, 9, "x0_tpl", init['x0_tpl'], minval['x0_tpl'], maxval['x0_tpl'], vary['x0_tpl'], toggle_x0_tpl, "x0_tpl")
            add_param_row(frame_tpl, 10, "dx_tpl", init['dx_tpl'], minval['dx_tpl'], maxval['dx_tpl'], vary['dx_tpl'], toggle_dx_tpl, "dx_tpl")

            

            def hndl_remove_tpl_btn():
                global tpl_pres
                frame_tpl.grid_forget()
                tpl_pres = 0

            tk.Button(frame_tpl, text='Remove TPL component', command=hndl_remove_tpl_btn)\
                .grid(row=11, column=0, columnspan=5, pady=10)

            frame_tpl.grid(row=9, column=0, sticky="ew")
            for i in range(12):
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
            init['x0_qpl']=1
            init['dx_qpl']=0.1    

    
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
            vary['x0_qpl']=True
            vary['dx_qpl']=True
            
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
            maxval['x0_qpl']=10
            maxval['dx_qpl']=1

            
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
            minval['x0_qpl']=-1.1
            minval['dx_qpl']=0.01

            
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
            def toggle_x0_qpl(): vary['x0_qpl'] = not vary['x0_qpl']
            def toggle_dx_qpl(): vary['dx_qpl'] = not vary['dx_qpl']            

            


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
            
            add_param_row(frame_qpl, 12, "x0_qpl", init['x0_qpl'], minval['x0_qpl'], maxval['x0_qpl'], vary['x0_qpl'], toggle_x0_qpl, "x0_qpl")
            add_param_row(frame_qpl, 13, "dx_qpl", init['dx_qpl'], minval['dx_qpl'], maxval['dx_qpl'], vary['dx_qpl'], toggle_dx_qpl, "dx_qpl")



            def hndl_remove_qpl_btn():
                global qpl_pres
                frame_qpl.grid_forget()
                qpl_pres = 0

            tk.Button(frame_qpl, text='Remove QPL component', command=hndl_remove_qpl_btn)\
                .grid(row=14, column=0, columnspan=5, pady=10)

            frame_qpl.grid(row=10, column=0, sticky="ew")
            for i in range(15):
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
            init['x0_5pl']=1
            init['dx_5pl']=0.1    

    
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
            vary['x0_5pl']=True
            vary['dx_5pl']=True
            
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
            maxval['x0_5pl']=10
            maxval['dx_5pl']=1

            
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
            minval['x0_5pl']=-1
            minval['dx_5pl']=0.01


            global frame_quint_pl#defining gui section to handle quint_pl param options
            frame_quint_pl=tk.Frame(master=frame_params)
            
            tk.Label(frame_quint_pl, text="Quintuple Power Law", font=("Arial", 12, "bold"))\
                .grid(row=0, column=0, columnspan=5, pady=(0, 10))

            def toggle(var):
                vary[var] = not vary[var]

            row_counter = 1
            for name in ["A_5pl", "B_5pl","x1_5pl","A2_5pl", "B2_5pl","x2_5pl", "A3_5pl","B3_5pl","x3_5pl","A4_5pl", "B4_5pl", "x4_5pl","A5_5pl", "B5_5pl",'x0_5pl','dx_5pl']:
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
            for i in range(row_counter+1):
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
                
                
                init['x0_bpl']=None if init_x0_bpl_entry.get()=='None' else float(init_x0_bpl_entry.get())
                minval['x0_bpl']=None if minval_x0_bpl_entry.get()=='None' else float(minval_x0_bpl_entry.get())
                maxval['x0_bpl']=None if maxval_x0_bpl_entry.get()=='None' else float( maxval_x0_bpl_entry.get())
    
    
                init['dx_bpl']=None if init_dx_bpl_entry.get()=='None' else float(init_dx_bpl_entry.get())
                minval['dx_bpl']=None if minval_dx_bpl_entry.get()=='None' else float(minval_dx_bpl_entry.get())
                maxval['dx_bpl']=None if maxval_dx_bpl_entry.get()=='None' else float(maxval_dx_bpl_entry.get())
                
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
               
                init['x0_sing']=None if init_x0_sing_entry.get()=='None' else float(init_x0_sing_entry.get())
                minval['x0_sing']=None if minval_x0_sing_entry.get()=='None' else float(minval_x0_sing_entry.get())
                maxval['x0_sing']=None if maxval_x0_sing_entry.get()=='None' else float( maxval_x0_sing_entry.get())
    
    
                init['dx_sing']=None if init_dx_sing_entry.get()=='None' else float(init_dx_sing_entry.get())
                minval['dx_sing']=None if minval_dx_sing_entry.get()=='None' else float(minval_dx_sing_entry.get())
                maxval['dx_sing']=None if maxval_dx_sing_entry.get()=='None' else float(maxval_dx_sing_entry.get())

               
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
                
                init['x0_tpl']=None if init_x0_tpl_entry.get()=='None' else float(init_x0_tpl_entry.get())
                minval['x0_tpl']=None if minval_x0_tpl_entry.get()=='None' else float(minval_x0_tpl_entry.get())
                maxval['x0_tpl']=None if maxval_x0_tpl_entry.get()=='None' else float( maxval_x0_tpl_entry.get())
    
    
                init['dx_tpl']=None if init_dx_tpl_entry.get()=='None' else float(init_dx_tpl_entry.get())
                minval['dx_tpl']=None if minval_dx_tpl_entry.get()=='None' else float(minval_dx_tpl_entry.get())
                maxval['dx_tpl']=None if maxval_dx_tpl_entry.get()=='None' else float(maxval_dx_tpl_entry.get())
            
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
                
                init['x0_qpl']=None if init_x0_qpl_entry.get()=='None' else float(init_x0_qpl_entry.get())
                minval['x0_qpl']=None if minval_x0_qpl_entry.get()=='None' else float(minval_x0_qpl_entry.get())
                maxval['x0_qpl']=None if maxval_x0_qpl_entry.get()=='None' else float( maxval_x0_qpl_entry.get())
    
    
                init['dx_qpl']=None if init_dx_qpl_entry.get()=='None' else float(init_dx_qpl_entry.get())
                minval['dx_qpl']=None if minval_dx_qpl_entry.get()=='None' else float(minval_dx_qpl_entry.get())
                maxval['dx_qpl']=None if maxval_dx_qpl_entry.get()=='None' else float(maxval_dx_qpl_entry.get())
                
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
            
                init['x0_5pl']=None if init_x0_5pl_entry.get()=='None' else float(init_x0_5pl_entry.get())
                minval['x0_5pl']=None if minval_x0_5pl_entry.get()=='None' else float(minval_x0_5pl_entry.get())
                maxval['x0_5pl']=None if maxval_x0_5pl_entry.get()=='None' else float( maxval_x0_5pl_entry.get())
    
    
                init['dx_5pl']=None if init_dx_5pl_entry.get()=='None' else float(init_dx_5pl_entry.get())
                minval['dx_5pl']=None if minval_dx_5pl_entry.get()=='None' else float(minval_dx_5pl_entry.get())
                maxval['dx_5pl']=None if maxval_dx_5pl_entry.get()=='None' else float(maxval_dx_5pl_entry.get())
                
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
                        global x_data_E_sliced
                        parvals,param_uncert_calced,x_data_E_sliced=fitting(header,init,vary,minval,maxval,x_data,y_data,uncert,fitmin,fitmax,spec_type)
                        

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
                            init_x0_bpl_entry.delete(0, tk.END)
                            init_x0_bpl_entry.insert(0,parvals["x0_bpl"])
                            init_dx_bpl_entry.delete(0, tk.END)
                            init_dx_bpl_entry.insert(0,parvals["dx_bpl"])
                            
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
                            init_x0_sing_entry.delete(0, tk.END)
                            init_x0_sing_entry.insert(0,parvals["x0_sing"])
                            init_dx_sing_entry.delete(0, tk.END)
                            init_dx_sing_entry.insert(0,parvals["dx_sing"])
                            
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
                            init_x0_tpl_entry.delete(0, tk.END)
                            init_x0_tpl_entry.insert(0,parvals["x0_tpl"])
                            init_dx_tpl_entry.delete(0, tk.END)
                            init_dx_tpl_entry.insert(0,parvals["dx_tpl"])

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
                            init_x0_qpl_entry.delete(0, tk.END)
                            init_x0_qpl_entry.insert(0,parvals["x0_qpl"])
                            init_dx_qpl_entry.delete(0, tk.END)
                            init_dx_qpl_entry.insert(0,parvals["dx_qpl"])
                            
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
                            init_x0_5pl_entry.delete(0, tk.END)
                            init_x0_5pl_entry.insert(0,parvals["x0_5pl"])
                            init_dx_5pl_entry.delete(0, tk.END)
                            init_dx_5pl_entry.insert(0,parvals["dx_5pl"])
            
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
                
                init['x0_bpl']=None if init_x0_bpl_entry.get()=='None' else float(init_x0_bpl_entry.get())
                minval['x0_bpl']=None if minval_x0_bpl_entry.get()=='None' else float(minval_x0_bpl_entry.get())
                maxval['x0_bpl']=None if maxval_x0_bpl_entry.get()=='None' else float( maxval_x0_bpl_entry.get())
    
    
                init['dx_bpl']=None if init_dx_bpl_entry.get()=='None' else float(init_dx_bpl_entry.get())
                minval['dx_bpl']=None if minval_dx_bpl_entry.get()=='None' else float(minval_dx_bpl_entry.get())
                maxval['dx_bpl']=None if maxval_dx_bpl_entry.get()=='None' else float(maxval_dx_bpl_entry.get())
                
                
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
               
                init['x0_sing']=None if init_x0_sing_entry.get()=='None' else float(init_x0_sing_entry.get())
                minval['x0_sing']=None if minval_x0_sing_entry.get()=='None' else float(minval_x0_sing_entry.get())
                maxval['x0_sing']=None if maxval_x0_sing_entry.get()=='None' else float( maxval_x0_sing_entry.get())
    
    
                init['dx_sing']=None if init_dx_sing_entry.get()=='None' else float(init_dx_sing_entry.get())
                minval['dx_sing']=None if minval_dx_sing_entry.get()=='None' else float(minval_dx_sing_entry.get())
                maxval['dx_sing']=None if maxval_dx_sing_entry.get()=='None' else float(maxval_dx_sing_entry.get())


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
                
                init['x0_tpl']=None if init_x0_tpl_entry.get()=='None' else float(init_x0_tpl_entry.get())
                minval['x0_tpl']=None if minval_x0_tpl_entry.get()=='None' else float(minval_x0_tpl_entry.get())
                maxval['x0_tpl']=None if maxval_x0_tpl_entry.get()=='None' else float( maxval_x0_tpl_entry.get())
    
    
                init['dx_tpl']=None if init_dx_tpl_entry.get()=='None' else float(init_dx_tpl_entry.get())
                minval['dx_tpl']=None if minval_dx_tpl_entry.get()=='None' else float(minval_dx_tpl_entry.get())
                maxval['dx_tpl']=None if maxval_dx_tpl_entry.get()=='None' else float(maxval_dx_tpl_entry.get())
            
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
                
                init['x0_qpl']=None if init_x0_qpl_entry.get()=='None' else float(init_x0_qpl_entry.get())
                minval['x0_qpl']=None if minval_x0_qpl_entry.get()=='None' else float(minval_x0_qpl_entry.get())
                maxval['x0_qpl']=None if maxval_x0_qpl_entry.get()=='None' else float( maxval_x0_qpl_entry.get())
    
    
                init['dx_qpl']=None if init_dx_qpl_entry.get()=='None' else float(init_dx_qpl_entry.get())
                minval['dx_qpl']=None if minval_dx_qpl_entry.get()=='None' else float(minval_dx_qpl_entry.get())
                maxval['dx_qpl']=None if maxval_dx_qpl_entry.get()=='None' else float(maxval_dx_qpl_entry.get())
                
                
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
                            
                init['x0_5pl']=None if init_x0_5pl_entry.get()=='None' else float(init_x0_5pl_entry.get())
                minval['x0_5pl']=None if minval_x0_5pl_entry.get()=='None' else float(minval_x0_5pl_entry.get())
                maxval['x0_5pl']=None if maxval_x0_5pl_entry.get()=='None' else float( maxval_x0_5pl_entry.get())
    
    
                init['dx_5pl']=None if init_dx_5pl_entry.get()=='None' else float(init_dx_5pl_entry.get())
                minval['dx_5pl']=None if minval_dx_5pl_entry.get()=='None' else float(minval_dx_5pl_entry.get())
                maxval['dx_5pl']=None if maxval_dx_5pl_entry.get()=='None' else float(maxval_dx_5pl_entry.get())
                
                
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
            init_x0_bpl_entry.delete(0, tk.END)
            init_x0_bpl_entry.insert(0,parvals_ld["x0_bpl"])
            init_dx_bpl_entry.delete(0, tk.END)
            init_dx_bpl_entry.insert(0,parvals_ld["dx_bpl"])

            
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
            init_x0_sing_entry.delete(0, tk.END)
            init_x0_sing_entry.insert(0,parvals_ld["x0_sing"])
            init_dx_sing_entry.delete(0, tk.END)
            init_dx_sing_entry.insert(0,parvals_ld["dx_sing"])
            
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
            init_x0_tpl_entry.delete(0, tk.END)
            init_x0_tpl_entry.insert(0,parvals_ld["x0_tpl"])
            init_dx_tpl_entry.delete(0, tk.END)
            init_dx_tpl_entry.insert(0,parvals_ld["dx_tpl"])
            
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
            init_x0_qpl_entry.delete(0, tk.END)
            init_x0_qpl_entry.insert(0,parvals_ld["x0_qpl"])
            init_dx_qpl_entry.delete(0, tk.END)
            init_dx_qpl_entry.insert(0,parvals_ld["dx_qpl"])
        
            
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
            init_x0_5pl_entry.delete(0, tk.END)
            init_x0_5pl_entry.insert(0,parvals_ld["x0_5pl"])
            init_dx_5pl_entry.delete(0, tk.END)
            init_dx_5pl_entry.insert(0,parvals_ld["dx_5pl"])
                   
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
     
    def resid_save_hndl():
        try:
            fit_summary
            
        except NameError:tk.messagebox.showwarning("No Results", "No residuals yet, run a fit first")

        
        else:
            #organise into dataframe
            spec_dict={'energies':list(x_data_E_sliced) ,'resids': list(resids),'date':[str(date) for i in list(x_data_E_sliced)],'inst':[inst for i in list(x_data_E_sliced)],'spec_type':[spec_type for i in list(x_data_E_sliced)]}
            spec_frame=pd.DataFrame(spec_dict)
            files = [('Text Document','*.txt')]
            file_obj=tk.filedialog.asksaveasfile(filetypes = files, defaultextension=".txt")
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