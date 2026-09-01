#%%Initial set up
import sys#for file path handling
import os#has general functions for file manipulation




import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk
import lmfit #this module contains the functions for the curve fitting
import numpy as np #general mathematical operations
from scipy.special import erf #imports an erf function for use in some of the fitting operations
from matplotlib import pyplot as plt #general plotting operations
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)#allows plotting to a tkinter window
from scipy.stats.qmc import LatinHypercube
import datetime as dt#handles general datetime operations
import pandas as pd #module for dataframe and time series handling
import scipy #for reading in idl saves and other various functions

import re#for handling regexs to validate inputs
import random as rn#for random number and choice utility, particularly in uncertainty estimation
from tqdm import tqdm #for tracking progress of long iterables

import math
import numdifftools
import random#needed for random numbers

import threading  #required to allow gui updates during code
from . import state  #shared cross-module state
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


def broken_power_law(x,x1,A,B,A2,B2,x0_bpl,dx_bpl): #defines a broken power law to fit, like the thick target approx.
        
    if type(x)==int:
        xlo=((erf(((x-x0_bpl)/dx_bpl))+1)/2) if x<x1 else 0 #below x0 if x<x1 else 0
        xhi=1 if x>=x1 else 0#above x1  
    else:
        x=np.array(x)
        xlo=[ ((erf(((x_i-x0_bpl)/dx_bpl))+1)/2) if x_i<x1 else 0 for x_i in x] #below x1
        xhi=[ 1 if x_i>=x1 else 0 for x_i in x]#above x1    
    y_bpl=(xlo*lin_func(x,A,B))+(xhi*lin_func2(x,A2,B2))
    return y_bpl

def gauss_func(x,gauss_amp,gauss_centre,sigma): #defines a gaussian function that can be added
    x=np.array(x)
    y_gauss=gauss_amp*np.exp((-(x-gauss_centre)**2)/(2*sigma**2))
    return y_gauss

def power_func(x,A_sing,B_sing,x0_sing,dx_sing): #defines a simgle power law that can be added
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

def triple_power_law(x,x1,x2,A,B,A2,B2,A3,B3,x0_tpl,dx_tpl):
    
    if type(x)==int:
        xlo=((erf(((x-x0_tpl)/dx_tpl))+1)/2) if x<x1 else 0 #below x1
        xmid=1  if (x>=x1 and x<=x2) else 0 #between x1 and x2
        xhi=1 if x>x2 else 0#above x2  
    else:
        x=np.array(x)
        xlo=[ ((erf(((x_i-x0_tpl)/dx_tpl))+1)/2) if x_i<x1 else 0 for x_i in x] #below x1
        xmid =[ 1 if (x_i>=x1 and x_i<=x2) else 0 for x_i in x] #between x1 and x2
        xhi=[ 1 if x_i>=x2 else 0 for x_i in x]#above x2    
    
    
    y_tpl=(xlo*lin_func(x,A,B))+(xmid*lin_func2(x,A2,B2))+(xhi*lin_func2(x,A3,B3))
    return y_tpl

def quad_power_law(x,x1,x2,x3,A,B,A2,B2,A3,B3,A4,B4,x0_qpl,dx_qpl):
    
    if type(x)==int:
        xlo=((erf(((x-x0_qpl)/dx_qpl))+1)/2) if x<x1 else 0 #below x1
        xmid1=1  if (x>=x1 and x<=x2) else 0 #between x1 and x2
        xmid2=1  if (x>x2 and x<=x3) else 0 #between x2 and x3
        xhi=1 if x>x3 else 0#above x3  
    else:
        x=np.array(x)
        xlo=[ ((erf(((x_i-x0_qpl)/dx_qpl))+1)/2) if x_i<x1 else 0 for x_i in x] #below x1
        xmid1 =[ 1 if (x_i>=x1 and x_i<=x2) else 0 for x_i in x] #between x1 and x2
        xmid2 =[ 1 if (x_i>x2 and x_i<=x3) else 0 for x_i in x] #between x2 and x3
        xhi=[ 1 if x_i>x3 else 0 for x_i in x]#above x3    
    
    
    y_qpl=(xlo*lin_func2(x,A,B))+(xmid1*lin_func2(x,A2,B2))+(xmid2*lin_func2(x,A3,B3))+(xhi*lin_func2(x,A4,B4))
    return y_qpl

def quint_power_law(x,x1,x2,x3,x4,A,B,A2,B2,A3,B3,A4,B4,A5,B5,x0_5pl,dx_5pl):
    
    if type(x)==int:
        xlo=((erf(((x-x0_5pl)/dx_5pl))+1)/2) if x<x1 else 0 #below x1
        xmid1=1  if (x>=x1 and x<=x2) else 0 #between x1 and x2
        xmid2=1  if (x>x2 and x<=x3) else 0 #between x2 and x3
        xmid3=1  if (x>x3 and x<=x4) else 0 #between x3 and x4
        xhi=1 if x>x4 else 0#above x4  
    else:
        x=np.array(x)
        xlo=[ ((erf(((x_i-x0_5pl)/dx_5pl))+1)/2) if x_i<x1 else 0 for x_i in x] #below x1
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
    calcd_vals=state.test_func(x_data,parvals,header)#uses the defined test function to get the calculated values
    #calc state.resids
    resids=(np.array(calcd_vals)-np.array(y_data))/(np.array(uncert)) #calculates the residuals
    return list(resids)

def neg_max_like(pars,x_data,y_data,uncert,header):#the negative maximum log likelihood 
    #unpack params object
    parvals=pars.valuesdict() #converts the parameters to a dictionary form
    
    #prob of measuring counts given model
    
    #calculate values
    calcd_vals=state.test_func(x_data,parvals,header)#uses the defined test function to get the calculated values
    
    # numerical safety: mu must be positive
    uncert = np.maximum(uncert, 1e-12)

    state.resids=(np.array(calcd_vals)-np.array(y_data))/(np.array(uncert)) #calculates the residuals
    n=len(x_data)
    nll = 0.5 * (n * np.log(2*np.pi) + np.sum(np.log(uncert**2)) + np.sum(state.resids**2))
    return nll


def build_seeded_population(params, free_names, popsize, seed, guess_vec):#seeding populations from initial guess for use with DE
    n_free = len(free_names)
    pop_size_total = popsize * n_free
    sampler = LatinHypercube(d=n_free, seed=seed)
    unit_samples = sampler.random(n=pop_size_total)

    scaled_pop = np.zeros((pop_size_total, n_free))
    for i, name in enumerate(free_names):
        lo, hi = params[name].min, params[name].max
        # treat a lower bound of 0 (or anything <=0) as "12 orders below hi"
        floor = hi * 1e-12 if hi > 0 else lo
        floor = max(lo, floor)
        span_orders = (hi / floor) if floor > 0 else 1

        if lo >= 0 and floor > 0 and span_orders > 100:
            # wide, non-negative range -> sample log-uniformly
            log_lo, log_hi = np.log10(floor), np.log10(hi)
            scaled_pop[:, i] = 10 ** (log_lo + unit_samples[:, i] * (log_hi - log_lo))
        else:
            # narrow range, or genuinely spans negative values (e.g. spectral index) -> linear
            scaled_pop[:, i] = lo + unit_samples[:, i] * (hi - lo)

    scaled_pop[0] = guess_vec  # guarantee the known-good point is present
    return scaled_pop


def fitting(header,init,vary,minval,maxval,x_data,y_data,uncert,fitmin,fitmax,spec_type): #defines our fitting process
    
    if state.fit_window is not None:# and state.fit_window.winfo_exists():
        #close any open figues
        #state.fit_window.destroy()
        state.fit_window=None
        

    
    
    
    if state.resid_window is not None:# and state.resid_window.winfo_exists():
        #close any open figues
        #state.resid_window.destroy()
        state.resid_window=None
    
    if state.preview_window is not None:# and state.preview_window.winfo_exists():
        #close any open figues
        #state.preview_window.destroy()
        state.preview_window=None
    
    #set range to user defined fitting limits
    x_data_sliced=list()
    y_data_sliced=list()
    uncert_sliced=list()
    for pos,E in enumerate(x_data):
      if E>=fitmin  and E<=fitmax:
          x_data_sliced.append(E)
          y_data_sliced.append(y_data[pos])
          uncert_sliced.append(uncert[pos])
    #save fitted energy range so that it can be retireved
    #build test function according to the user set options
    def _test_func(x,parvals,header): # this function is the one we are trying to fit to the data
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
            x0_bpl=parvals["x0_bpl"]
            dx_bpl=parvals["dx_bpl"]   
            y+=broken_power_law(x,x1,A,B,A2,B2,x0_bpl,dx_bpl)
        
        
        
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
            x0_sing=parvals["x0_sing"]
            dx_sing=parvals["dx_sing"]   
            y+=power_func(x, A_sing, B_sing,x0_sing,dx_sing)
            
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
            x0_tpl=parvals["x0_tpl"]
            dx_tpl=parvals["dx_tpl"]   
            y+=triple_power_law(x,x1_tpl,x2_tpl,A_tpl,B_tpl,A2_tpl,B2_tpl,A3_tpl,B3_tpl,x0_tpl,dx_tpl)
            
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
            x0_qpl=parvals["x0_qpl"]
            dx_qpl=parvals["dx_qpl"]   
            y+=quad_power_law(x,x1_qpl,x2_qpl,x3_qpl,A_qpl,B_qpl,A2_qpl,B2_qpl,A3_qpl,B3_qpl,A4_qpl,B4_qpl,x0_qpl,dx_qpl)
            
        
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
                x0_5pl=parvals["x0_5pl"]
                dx_5pl=parvals["dx_5pl"]   
                y+=quint_power_law(x, x1_5pl, x2_5pl, x3_5pl, x4_5pl, A_5pl, B_5pl, A2_5pl, B2_5pl, A3_5pl, B3_5pl, A4_5pl, B4_5pl, A5_5pl, B5_5pl,x0_5pl,dx_5pl)
            
    
        return y
    state.test_func = _test_func  #store in shared state for other modules
    
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
                       ('A2',init['A2'] ,vary['A2'],minval['A2'],maxval['A2'],'A * x1**(B-B2)',None),#must add after A is defined
                       ('x0_bpl',init['x0_bpl'] ,vary['x0_bpl'],minval['x0_bpl'],maxval['x0_bpl'],None,None),
                       ('dx_bpl',init['dx_bpl'] ,vary['dx_bpl'],minval['dx_bpl'],maxval['dx_bpl'],None,None))
    
    
    if header[42]=='1': #ie if gaussian is present
        params.add_many(('gauss_amp',init['gauss_amp'],vary['gauss_amp'],minval['gauss_amp'],maxval['gauss_amp'],None,None),
                    ('gauss_centre',init['gauss_centre'] ,vary['gauss_centre'],minval['gauss_centre'],maxval['gauss_centre'],None,None),
                    ('sigma',init['sigma'],vary['sigma'],minval['sigma'],maxval['sigma'],None,None))
    
    if header[56]=='1':#ie if the power law is present
        params.add_many(('A_sing',init['A_sing'],vary['A_sing'],minval['A_sing'],maxval['A_sing'],None,None),
                        ('B_sing',init['B_sing'] ,vary['B_sing'],minval['B_sing'],maxval['B_sing'],None,None),
                        ('x0_sing',init['x0_sing'] ,vary['x0_sing'],minval['x0_sing'],maxval['x0_sing'],None,None),
                        ('dx_sing',init['dx_sing'] ,vary['dx_sing'],minval['dx_sing'],maxval['dx_sing'],None,None))

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
                  ('A3_tpl',init['A3_tpl'] ,vary['A3_tpl'],minval['A3_tpl'],maxval['A3_tpl'],'A2_tpl * x2_tpl**(B2_tpl-B3_tpl)',None),#must add after A_tpl is defined
                  ('x0_tpl',init['x0_tpl'] ,vary['x0_tpl'],minval['x0_tpl'],maxval['x0_tpl'],None,None),
                  ('dx_tpl',init['dx_tpl'] ,vary['dx_tpl'],minval['dx_tpl'],maxval['dx_tpl'],None,None))
    
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
                  ('A4_qpl',init['A4_qpl'] ,vary['A4_qpl'],minval['A4_qpl'],maxval['A4_qpl'],'A3_qpl * x3_qpl**(B3_qpl-B4_qpl)',None),#must add after A_qpl is defined
                  ('x0_qpl',init['x0_qpl'] ,vary['x0_qpl'],minval['x0_qpl'],maxval['x0_qpl'],None,None),
                  ('dx_qpl',init['dx_qpl'] ,vary['dx_qpl'],minval['dx_qpl'],maxval['dx_qpl'],None,None))
              
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
                  ('A2_5pl',init['A2_5pl'] ,vary['A2_5pl'],minval['A2_5pl'],maxval['A2_5pl'],'A_5pl * x1_5pl**(B_5pl-B2_5pl)',None),
                  ('A3_5pl',init['A3_5pl'] ,vary['A3_5pl'],minval['A3_5pl'],maxval['A3_5pl'],'A2_5pl * x2_5pl**(B2_5pl-B3_5pl)',None),
                  ('A4_5pl',init['A4_5pl'] ,vary['A4_5pl'],minval['A4_5pl'],maxval['A4_5pl'],'A3_5pl * x3_5pl**(B3_5pl-B4_5pl)',None),
                  ('A5_5pl',init['A5_5pl'] ,vary['A5_5pl'],minval['A5_5pl'],maxval['A5_5pl'],'A4_5pl * x4_5pl**(B4_5pl-B5_5pl)',None),#must add after A_5pl is defined
                  ('x0_5pl',init['x0_5pl'] ,vary['x0_5pl'],minval['x0_5pl'],maxval['x0_5pl'],None,None),
                  ('dx_5pl',init['dx_5pl'] ,vary['dx_5pl'],minval['dx_5pl'],maxval['dx_5pl'],None,None))
        
        
    
        
    #two stage fit, global and local minimisation# 
    #setup fitter, with resid func, param starts, and x+y data
    
    

    
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
    #breakpoint()
    def run_fit():
        best_result = None
        lowest_chisq = float("inf")
        n_seeds = 5
        

        
        #breakpoint()
        for seed in range(n_seeds):
            try:
                
                #initialise param space so can never do worse than initial guess
                free_names = [name for name, p in params.items() if p.vary]
                bounds = [(params[name].min, params[name].max) for name in free_names]
                n_free = len(free_names)
                popsize = 20
                pop_size_total = popsize * n_free
                
                # fill most of the population with Latin hypercube samples across bounds
                sampler = LatinHypercube(d=n_free, seed=seed)
                unit_samples = sampler.random(n=pop_size_total)
                scaled_pop = np.array([
                    lo + unit_samples[:, i] * (hi - lo) for i, (lo, hi) in enumerate(bounds)
                ]).T
                
                # overwrite the first row with actual initial guess
                guess_vec = np.array([params[name].value for name in free_names])
                scaled_pop[0] = guess_vec                
                
                desired_generations =1000
                popsize=20
                np.random.seed(seed)
                random.seed(seed)
                free_names = [name for name, p in params.items() if p.vary]
                n_free = len(free_names)
                guess_vec = np.array([params[name].value for name in free_names])
                
                scaled_pop = build_seeded_population(params, free_names, popsize, seed, guess_vec)
                
                trial = fitter.minimize(
                    method='differential_evolution',
                    strategy='best2bin',
                    max_nfev=popsize * (n_free + 1) * desired_generations,
                    popsize=popsize,
                    init=scaled_pop,          # <-- the missing piece
                    tol=1e-8,
                    mutation=(0.5, 1.5),
                    recombination=0.7,
                    seed=seed,
                    polish=False,
                    updating='immediate',
                )
                if trial.chisqr < lowest_chisq:
                    best_result = trial
                    lowest_chisq = trial.chisqr
            except Exception as e:
                print(f"Seed {seed} failed: {e}")
                #breakpoint()
            pb['value'] += 100 / n_seeds
            pb.update_idletasks()

        if best_result is not None:
            params.update(best_result.params)
            state.fitter_local = lmfit.Minimizer(
                resid_calc,
                params,
                fcn_kws={'x_data': x_data_sliced, 'y_data': y_data_sliced, 'uncert': uncert_sliced, 'header': header},
                scale_covar=True)
            try:
                state.result = state.fitter_local.minimize(
                    method='nelder',
                    max_nfev=50000,
                    #x_scale='jac',
                    #ftol=1e-9, xtol=1e-9, gtol=1e-9
                    )
                #nelder is the most resilient to nans for final stage
            except Exception as e:
                    print(f"Local failed: {e}")
                    #breakpoint()
        progress_win.destroy()  # Closes the window, allows wait_window to continue
        

    threading.Thread(target=run_fit, daemon=True).start()
    
    # Block until progress window is destroyed
    progress_win.wait_window()
    

    #write error report (optional, un comment for print to console)
    state.fit_summary=lmfit.fit_report(state.result)

    state.bic=state.result.bic
    
    #unpack params object
    pars=state.result.params
    #convert params object to a dictionary
    state.parvals=pars.valuesdict()
    parvals=state.parvals   #local alias for convenience in unpacking below
    
    #read the parameters dictionary, according to the functions that should be present
    if header[9]=='1':# ie if the bpl is present
        x1=parvals["x1"]
        A=parvals["A"]
        B=parvals["B"]
        A2=parvals["A2"]
        B2=parvals["B2"]
        x0_bpl=parvals["x0_bpl"]
        dx_bpl=parvals["dx_bpl"]   
    
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
        x0_sing=parvals["x0_sing"]
        dx_sing=parvals["dx_sing"]  
    
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
        x0_tpl=parvals["x0_tpl"]
        dx_tpl=parvals["dx_tpl"]   
   
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
        x0_qpl=parvals["x0_qpl"]
        dx_qpl=parvals["dx_qpl"]   

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
        x0_5pl=parvals["x0_5pl"]
        dx_5pl=parvals["dx_5pl"]   

    #print(B)
    
    #calculate the chi squared of the fit
    y_fit=state.test_func(x_data_sliced,state.parvals,header)#fitted y values
    chi_sq=sum(((y_fit-y_data_sliced)/uncert_sliced)**2)#chi squared

    #calc reduced chi sq
    dof=len(y_data_sliced)-len(state.parvals)#degrees of freedom
    state.redchi=chi_sq/dof
    

    #breakpoint()
    #return the parameter uncertainties as well
    native_uncert=state.result.errorbars#this determines if uncerts were generated natively in the fit
    if native_uncert:#if fitter generated uncerts, use those
        print('native uncerts use')
        state.param_uncert_calced={param_name: param.stderr for param_name, param in state.result.params.items()}#output requires some reprocessing to correct form by removing param values

    else:#if fitter has not generated uncerts, use bayesian posterior method
        print('bayes uncerts use')
        state.fitter_local = lmfit.Minimizer(
            resid_calc,
            params,
            fcn_kws={'x_data': x_data_sliced, 'y_data': y_data_sliced, 'uncert': uncert_sliced, 'header': header},
            scale_covar=True,nan_policy='omit')
        try:#try-except wrap in case emcee makes inf/nan
            posterior = state.fitter_local.minimize( method='emcee',params= state.result.params, burn=300, steps=5000, thin=20,
                              is_weighted=True,progress=True)
        except Exception as e:
            print(f"Bayes uncert failed: {e}")
            #breakpoint()
                    
                    
        #locate MLE value in the chain to get sigmas-not neccessary for operations
        highest_prob = np.argmax(posterior.lnprob)
        hp_loc = np.unravel_index(highest_prob, posterior.lnprob.shape)
        mle_soln = posterior.chain[hp_loc]#chain item at location of highest prob
        state.param_uncert_calced=dict()#set up uncert starage object 
        for name, param in state.parvals.items():#go through each parameter. if varying get stderr, if not is 0
            if vary[name]:#chain only has varying params! must make sure they only use these
                state.param_uncert_calced[name]=posterior.params[name].stderr
            else:#for non-varying params
                state.param_uncert_calced[name] =0
            
            
    

    #%%state.result plotting
    x_model=np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 1000000)#set up an x-model for plotting the fitted line
    fit=state.test_func(x_model,state.parvals,header)# y-values for our new modeled fit
    

    


    
    
    
    
    #fit window declared global above


    
    if state.fit_window is not None:# and state.fit_window.winfo_exists():
            #close any open figues
        state.fit_window.destroy()
        state.fit_window=None
        print("test") 
    
    #open a window, around which the main program section is built
    
    plot_wind_size=(6,4)#define the window size for the plots
    
    state.fit_window=tk.Toplevel()
    state.fit_window.title('Fit window')
    state.fit_window.rowconfigure(0, weight=1)
    state.fit_window.columnconfigure(0, weight=1)
    fig_fit =plt.Figure(figsize=plot_wind_size, dpi=200)
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
        xlo=[ ((erf(((x_i-x0_bpl)/dx_bpl))+1)/2) if x_i<x1 else 0 for x_i in x_model] #below x0
        xhi=[ 1 if x_i>=x1 else 0 for x_i in x_model]#above x
        
        fit3=lin_func(x_model,A,B)*xlo
        fit4=lin_func2(x_model,A2,B2)*xhi

        ax_fit.plot(x_model,fit3, 'g', label='Broken Power Law',linestyle='dotted')
        ax_fit.plot(x_model,fit4, 'g')
        ax_fit.scatter(x1,state.test_func(int(x1),state.parvals,header),zorder=100000,c='black')
        
    if header[42]=='1': #ie if gaussian is present
        fit5=gauss_func(x_model, gauss_amp, gauss_centre, sigma)
        ax_fit.plot(x_model,fit5, 'b', label='Gaussian',linestyle='dashdot')
    
    
    if header[56]=='1': #ie if power law is present
        fit6=power_func(x_model, A_sing, B_sing,x0_sing,dx_sing)
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
        xlo=[ ((erf(((x_i-x0_tpl)/dx_tpl))+1)/2) if x_i<x1_tpl else 0 for x_i in x_model] #below x1
        xmid =[ 1 if (x_i>=x1_tpl and x_i<=x2_tpl) else 0 for x_i in x_model] #between x1 and x2
        xhi=[ 1 if x_i>=x2_tpl else 0 for x_i in x_model]#above x2    
        
        fit12=lin_func(x_model,A_tpl,B_tpl)*xlo
        fit13=lin_func2(x_model,A2_tpl,B2_tpl)*xmid
        fit14=lin_func2(x_model,A3_tpl,B3_tpl)*xhi
        
        ax_fit.plot(x_model,fit12, 'g')
        ax_fit.plot(x_model,fit13, 'g', label='Triple Power Law',linestyle='dotted')
        ax_fit.plot(x_model,fit14, 'g')
        ax_fit.scatter(x1_tpl,state.test_func(int(x1_tpl),state.parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x2_tpl,state.test_func(int(x2_tpl),state.parvals,header),zorder=100000,c='black')
        
    if header[142]=='1':# ie if the qpl is present
        xlo=[ ((erf(((x_i-x0_qpl)/dx_qpl))+1)/2) if x_i<x1_qpl else 0 for x_i in x_model] #below x1
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
        ax_fit.scatter(x1_qpl,state.test_func(int(x1_qpl),state.parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x2_qpl,state.test_func(int(x2_qpl),state.parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x3_qpl,state.test_func(int(x3_qpl),state.parvals,header),zorder=100000,c='black')

    if header[159]=='1':# ie if the 5pl is present
        xlo=[ ((erf(((x_i-x0_5pl)/dx_5pl))+1)/2) if x_i<x1_5pl else 0 for x_i in x_model] #below x1
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
        ax_fit.scatter(x1_5pl,state.test_func(int(x1_5pl),state.parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x2_5pl,state.test_func(int(x2_5pl),state.parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x3_5pl,state.test_func(int(x3_5pl),state.parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x4_5pl,state.test_func(int(x4_5pl),state.parvals,header),zorder=100000,c='black')
        
    ax_fit.set_yscale("log")
    ax_fit.set_xscale("log")
    
    #set plot limits so that it is focussed on the data, to avoid scaling issues from fitted curve
    ax_fit.set_ylim(np.nanmin(y_data)/2,np.nanmax(y_data)*2) 
    ax_fit.set_xlim(np.nanmin(x_model)-2,np.nanmax(x_model)+20)
    
    #add legend to plot
    ax_fit.legend(title=f"Reduced Chi sq = {round(state.redchi,1)}")



    ax_fit.grid()
    canvas_fit = FigureCanvasTkAgg(fig_fit, master=state.fit_window) 
    canvas_fit.draw()  
    canvas_fit.get_tk_widget().pack(fill="both",expand=True)
    
    #add buttton to save figure
    def fig_save_hndl():
        file_obj=tk.filedialog.asksaveasfilename()
        if not file_obj:  #user cancelled the dialog (returns empty string)
            return
        fig_fit.savefig(file_obj,bbox_inches='tight')
    
    #create preview button
    fig_save_button=tk.Button(
    text="Save Plot",  width=25,  height=2,  bg="white",  fg="black",  command=fig_save_hndl,  master=state.fit_window)
    fig_save_button.pack(side=tk.BOTTOM) 
    
    #second plot showing the residuals of the fit
    
    state.resid_window=tk.Toplevel()
    state.resid_window.rowconfigure(0, weight=1)
    state.resid_window.columnconfigure(0, weight=1)
    fig_resids =plt.Figure(figsize=plot_wind_size, dpi=200)
    ax_resids= fig_resids.add_subplot(1, 1, 1)

    state.resids=resid_calc(pars,x_data_sliced,y_data_sliced,uncert_sliced,header)

    ax_resids.plot(list(x_data_sliced),state.resids,marker='o')
    ax_resids.set_ylabel('Residual')
    ax_resids.set_xscale('log')
    ax_resids.set_xlabel('Energy (keV)')
    ax_resids.grid()
    

     
     
    #fig_resids.savefig(fname=.png',bbox_inches='tight')
    canvas_resids = FigureCanvasTkAgg(fig_resids, master=state.resid_window) 
    canvas_resids.draw()  
    canvas_resids.get_tk_widget().pack(fill="both",expand=True)
    


    return(state.parvals,state.param_uncert_calced,x_data_sliced)
