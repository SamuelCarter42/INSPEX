#bootstrapping method for parameter uncertainties

import sys#for file path handling
import os#has general functions for file manipulation

sys.path.append(f"{os.getcwd()}/Dependencies")#ensures that dependencies folder is available at point that modules are loaded

import tkinter as tk #this module contains most of the functions to run the gui
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
import math
#%%functions for fitting
k_B=8.617333262*(10**-8) # Boltzmann constant in keV per kelvin
G=6.67430e-11#in N m^2 kg^-2
m_sun=1.989e30
r_sun=6.957e8

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
    #we use erfs to describe the transitions, thus making the changes more gentle than piecewise and allowing the fitter to vary them
    x=np.array(x)
    xlo=[ 1 if x_i<x1 else 0 for x_i in x] #below x1
    xhi=[ 1 if x_i>=x1 else 0 for x_i in x]#above x1


    y_bpl=(xlo*lin_func(x,A,B))+(xhi*lin_func2(x,A2,B2))
    

    return y_bpl

def gauss_func(x,gauss_amp,gauss_centre,sigma): #defines a gaussian function that can be added
    x=np.array(x)
    y_gauss=gauss_amp*np.exp((-(x-gauss_centre)**2)/(2*sigma**2))
    return y_gauss

def power_func(x,A_sing,B_sing,dx_sing,x0_sing): #defines a simgle power law that can be added
    x=np.array(x)
    xlo_sing=(erf(((x-x0_sing)/dx_sing))+1)/2#below x0
    y_pow=xlo_sing*(A_sing*(x**B_sing))
    return y_pow


def kappa_func(x, A_k, T_k, m_i, n_i, kappa):

    v_th=np.sqrt((2*x)/m_i)
    w=np.sqrt(((2*kappa-3)*k_B*T_k)/(kappa*m_i))
    term1=((v_th**2)/m_i)*(n_i/(2*np.pi*(kappa*w**2)**(3/2)))
    term2=math.gamma(kappa+1)/(math.gamma(kappa-1/2)*math.gamma(3/2))
    term3=(1+((v_th**2)/(kappa*(w**2))))**-(kappa+1)
    
    
    y_kappa=A_k*term1*term2*term3
    
    
    
    return y_kappa
    
    
#a combined bpl and thermal
def bpl_and_therm_func(x,therm_amp,T,alpha,x0,x1,B,B2):
    x=np.array(x)
    

    xmid=[ 1 if x_i<x1  else 0 for x_i in x] #below x1, above x0 'and x_i>=x0'
    xhi=[ 1 if x_i>=x1 else 0 for x_i in x]#above x1    
    
    y_therm=(therm_amp*(x**alpha)*np.exp(-x/(k_B*T)))
    
    
    
    A=therm_amp*(x0**(alpha-B))*np.exp(-x0/(k_B*T))
    A2=A * x1**(B-B2)
    

    y_bpl=(xmid*lin_func(x,A,B))+(xhi*lin_func2(x,A2,B2))
    
    
    y_combined=y_therm+y_bpl
    
    return y_combined
#%%fit curve
def resid_calc(pars,x_data,y_data,uncert,header): #defines the calculator for residuals that the fitting function needs to minimise
    #unpack params object
    parvals=pars.valuesdict() #converts the parameters to a dictionary form

    #calculate values
    calcd_vals=test_func(x_data,parvals,header)#uses the defined test function to get the calculated values
    #calc resids
    resids=(np.array(calcd_vals)-np.array(y_data))/(np.array(uncert)*1) #calculates the residuals

    return list(resids)
def fit_curve(header,init,vary,minval,maxval,x_data,y_data,uncert,fitmin,fitmax,method_1,method_2):


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
            dx_sing=parvals["dx_sing"]
            x0_sing=parvals["x0_sing"]
            y+=power_func(x, A_sing, B_sing,dx_sing,x0_sing)
            
        if header[70]=='1': #ie if kappa fn is present
            n_i=parvals["n_i"]
            A_k=parvals["A_k"]
            T_k=parvals["T_k"]
            m_i=parvals["m_i"]
            kappa=parvals["kappa"]
            y+=kappa_func(x, A_k, T_k, m_i,n_i,kappa)
        
        if header[92]==1:
            amp_c=parvals['amp_c']
            T_c=parvals['T_c']
            alpha_c=parvals['alpha_c']
            x0_c=parvals['x0_c']
            x1_c=parvals['x1_c']
            B_c=parvals['B_c']
            B2_c=parvals['B2_c']
            
            y+=bpl_and_therm_func(x,amp_c,T_c,alpha_c,x0_c,x1_c,B_c,B2_c)

        return y
    
    #define params with bounds and initial values
    params=lmfit.Parameters()
    
    #adding params depending on which functions user has selected
    
    #addwithtuples:(NAME VALUE VARY MIN MAX EXPR BRUTE_STEP) 



    if header[28]=='1':#ie if the therm func is present                   
        params.add_many(('amp',init['amp'],True,minval['amp'],maxval['amp'],None,None),
                    ('T',init['T'] ,True,minval['T'],maxval['T'],None,None),
                    ('alpha',init['alpha'],vary['alpha'],minval['alpha'],maxval['alpha'],None,None))
        
        
    if header[9]=='1':# ie if the broken power law is present
    
        params.add_many(('x1',init['x1'],True,minval['x1'],maxval['x1'],None,None),
                       ('B',init['B'] ,True,minval['B'],maxval['B'],None,None), 
                      ('B2',init['B2'],True,minval['B2'],maxval['B2'],None,None),
                      ('A',init['A'],True,minval['A'],maxval['A'],None,None),
                      ('A2',init['A2'] ,True,minval['A2'],maxval['A2'],'A * x1**(B-B2)',None))#must add after A is defined





    
    if header[42]=='1': #ie if gaussian is present
        params.add_many(('gauss_amp',init['gauss_amp'],True,minval['gauss_amp'],maxval['gauss_amp'],None,None),
                    ('gauss_centre',init['gauss_centre'] ,True,minval['gauss_centre'],maxval['gauss_centre'],None,None),
                    ('sigma',init['sigma'],True,minval['sigma'],maxval['sigma'],None,None))
    
    if header[56]=='1':#ie if the power law is present
        params.add_many(('A_sing',init['A_sing'],True,minval['A_sing'],maxval['A_sing'],None,None),
                    ('B_sing',init['B_sing'] ,True,minval['B_sing'],maxval['B_sing'],None,None),
                    ('x0_sing',init['x0_sing'] ,True,minval['x0_sing'],maxval['x0_sing'],None,None),
                    ('dx_sing',init['dx_sing'] ,True,minval['dx_sing'],maxval['dx_sing'],None,None))

    if header[70]=='1':#ie if the kappa func is present
        params.add_many(('n_i',init['n_i'],True,minval['n_i'],maxval['n_i'],None,None),
                    ('A_k',init['A_k'] ,True,minval['A_k'],maxval['A_k'],None,None),
                    ('T_k',init['T_k'] ,True,minval['T_k'],maxval['T_k'],None,None),
                    ('m_i',init['m_i'] ,vary['m_i'],minval['m_i'],maxval['m_i'],None,None),#cannot vary the mass of an electron
                    ('kappa',init['kappa'] ,True,minval['kappa'],maxval['kappa'],None,None))

    if header[92]=='1':#ie if the combined thermal and bpl is present
    
        
        params.add_many(('amp_c',init['amp_c'],True,minval['amp_c'],maxval['amp_c'],None,None),
                    ('T_c',init['T_c'] ,True,minval['T_c'],maxval['T_c'],None,None),
                    ('alpha_c',init['alpha_c'],vary['alpha_c'],minval['alpha_c'],maxval['alpha_c'],None,None),
                    ('x0_c',init['x0_c'],True,minval['x0_c'],maxval['x0_c'],None,None),
                    ('x1_c',init['x1_c'],True,minval['x1_c'],maxval['x1_c'],None,None),
                     ('B_c',init['B_c'] ,True,minval['B_c'],maxval['B_c'],None,None), #this one should be shallower than B2
                    ('B2_c',init['B2_c'],True,minval['B2_c'],maxval['B2_c'],None,None))

    #two stage fit, global and local minimisation but only if not kappa
    try:
        #setup fitter, with resid func, param starts, and x+y data
        
        if header[70]!='1':#ie if the kappa func is not present
            fitter=lmfit.Minimizer(resid_calc, params, fcn_kws={'x_data':x_data_sliced,'y_data':y_data_sliced,'uncert':uncert_sliced, 'header':header} )
            #do the global fit, give the output
            result_global=fitter.minimize(method=method_1,stepsize=0.000000001)        
            # Use the results from basinhopping as initial parameters for leastsq
            params.update(result_global.params)
        
        # Now, refine the fit using leastsq
        fitter_local = lmfit.Minimizer(resid_calc, params, fcn_kws={'x_data': x_data_sliced, 'y_data': y_data_sliced, 'uncert': uncert_sliced, 'header':header})
        result = fitter_local.minimize(method=method_2,options={'ftol': 1e-9, 'gtol': 1e-9, 'eps': 1e-10})
        
    except ValueError as err:print(err)
    #write error report (optional, un comment for print to console)
    #lmfit.report_fit(result)
    
    #unpack params object
    pars=result.params
    
    #covert params object to a dictionary
    parvals=pars.valuesdict()
    
    
    return parvals




#%%
def uncert_bootstrap(header,init,vary,minval,maxval,x_data,y_data,uncert,fitmin,fitmax,n_sets,method_1,method_2):
    
    #generating datasets by bootstrapping method
    print("bootstrapping begins")
    #must zip together x and y
    points=np.column_stack((x_data,y_data))
    global minval_uncert
    minval_uncert=dict()
    global maxval_uncert
    maxval_uncert=dict()
    #remove upper and lower limits for all params to allow free fitting, except for physical limits
    #define initial minimum values
    if header[28]=='1':#ie if the therm func is present 
        minval_uncert['amp']=None
        minval_uncert['T']=0#temperature in kelvin begins at 0
        minval_uncert['alpha']=None
    if header[9]=='1':# ie if the broken power law is present #x positions are nergy and therefore cannot be negative
        minval_uncert['x1']=0
        minval_uncert['A']=None
        minval_uncert['B']=-10#less would be unrealistic
        minval_uncert['A2']=None
        minval_uncert['B2']=-10#less would be unrealistic

    if header[42]=='1': #ie if gaussian is present
        minval_uncert['gauss_amp']=None
        minval_uncert['gauss_centre']=min(x_data)#gauss centre should not be allowed to take function out of the fframe
        minval_uncert['sigma']=None
    if header[56]=='1': #ie if single power law is present
        minval_uncert['A_sing']=None
        minval_uncert['B_sing']=None
        minval_uncert['x0_sing']=None
        minval_uncert['dx_sing']=0
        
        
    if header[70]=='1': #ie if kappa fn is present
        minval_uncert['A_k']=0
        minval_uncert['T_k']=1e6
        minval_uncert['m_i']=0
        minval_uncert['n_i']=None
        minval_uncert['kappa']=(3/2)+0.0001#must be greater than 3/2
        
    if header[92]=='1':
        minval_uncert['amp_c']=0
        minval_uncert['T_c']=0
        minval_uncert['alpha_c']=None
        minval_uncert['x0_c']=0
        minval_uncert['x1_c']=0
        minval_uncert['B_c']=None
        minval_uncert['B2_c']=None
        
        
        
    #define initial maximum values 
    if header[28]=='1':#ie if the therm func is present 
        maxval_uncert['amp']=None
        maxval_uncert['T']=None
        maxval_uncert['alpha']=None
    if header[9]=='1':# ie if the broken power law is present

        maxval_uncert['x1']=None
        maxval_uncert['A']=None
        maxval_uncert['B']=0
        maxval_uncert['A2']=None
        maxval_uncert['B2']=0

    if header[42]=='1': #ie if gaussian is present
        maxval_uncert['gauss_amp']=None
        maxval_uncert['gauss_centre']=max(x_data)#gauss centre should not be allowed to take function out of the fframe
        maxval_uncert['sigma']=None
    if header[56]=='1': #ie if single power law is present
        maxval_uncert['A_sing']=None
        maxval_uncert['B_sing']=0
        maxval_uncert['x0_sing']=None
        maxval_uncert['dx_sing']=None
        
        
    if header[70]=='1': #ie if kappa fn is present
        maxval_uncert['A_k']=1
        maxval_uncert['T_k']=None
        maxval_uncert['m_i']=None
        maxval_uncert['n_i']=None
        maxval_uncert['kappa']=1000
        
    if header[92]=='1':
        maxval_uncert['amp_c']=None
        maxval_uncert['T_c']=None
        maxval_uncert['alpha_c']=None
        maxval_uncert['x0_c']=None
        maxval_uncert['x1_c']=None
        maxval_uncert['B_c']=0
        maxval_uncert['B2_c']=0
    
    
    
    
    
    par_dicts=list()
    
    #need to fit curve to n_sets varied data sets and save the fitted params
    for i in tqdm(range(n_sets)):
        this_set=rn.choices(points,  k =len(points))
        this_set=np.array(this_set)
        
        x_set=this_set[:,0]
        y_set=this_set[:,0]
        #then i need to fit the sets
        
        
        these_pars=fit_curve(header,init,vary,minval_uncert,maxval_uncert,x_set,y_set,uncert,fitmin,fitmax,method_1,method_2)
        par_dicts.append(these_pars)
    
    # Initialize lists for each parameter
    param_lists = {param: [] for param in minval_uncert.keys()}
    #save the parameters for each function into a list    
    for par_dict in par_dicts:
        for param in param_lists.keys():
            if param in par_dict:
                param_lists[param].append(par_dict[param])

    # Calculate uncertainties
    param_uncert_calced = {param: np.std(values) for param, values in param_lists.items()}

    return param_uncert_calced
