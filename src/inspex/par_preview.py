#%%Initial set up
import sys#for file path handling
import os#has general functions for file manipulation

from .fitting_and_resids import broken_power_law,therm_func,lin_func,lin_func2,gauss_func,power_func,kappa_func,bpl_and_therm_func,double_therm_func,triple_power_law,quad_power_law,quint_power_law
import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk
import numpy as np #general mathematical operations
from scipy.special import erf #imports an erf function for use in some of the fitting operations
from matplotlib import pyplot as plt #general plotting operations
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)#allows plotting to a tkinter window







#%%preview initial parameter for fit

def param_preview(x_data,y_data,parvals,header):
    x_model=np.logspace(np.log10(min(x_data)), np.log10(max(x_data)), 1000000)#set up an x-model for plotting the fitted line

    
    #unpack parvals
    #defining what parameters to read in, depending on the header definiions of the function to be fitted
    if header[9]=='1':# ie if the broken power law is present

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
        
        
    if header[56]=='1': #ie if single power law is present
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
            y+=quint_power_law(x, x1_5pl, x2_5pl, x3_5pl, x4_5pl, A_5pl, B_5pl, A2_5pl, B2_5pl, A3_5pl, B3_5pl, A4_5pl, B4_5pl, A5_5pl, B5_5pl,x0_5pl,dx_5pl)
            
        return y
    #print(header)
    fit=test_func(x_model,parvals,header)# y-values for our new modeled fit

    #open a new figure in a new window
    
    plot_wind_size=(6,4)#define the window size for the plots
    
    global preview_window
    preview_window=tk.Toplevel()
    preview_window.title('Preview window')
    preview_window.rowconfigure(0, weight=1)
    preview_window.columnconfigure(0, weight=1)
    fig_fit =plt.Figure(figsize=plot_wind_size, dpi=200)
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
        xlo=[ ((erf(((x_i-x0_bpl)/dx_bpl))+1)/2) if x_i<x1 else 0 for x_i in x_model] #below x0
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
        fit6=power_func(x_model, A_sing, B_sing,x0_sing,dx_sing)
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
        xlo=[ ((erf(((x_i-x0_tpl)/dx_tpl))+1)/2) if x_i<x1_tpl else 0 for x_i in x_model] #below x1
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
        xlo=[ ((erf(((x_i-x0_qpl)/dx_qpl))+1)/2) if x_i<x1_qpl else 0 for x_i in x_model] #below x1
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
        ax_fit.scatter(x1_5pl,test_func(int(x1_5pl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x2_5pl,test_func(int(x2_5pl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x3_5pl,test_func(int(x3_5pl),parvals,header),zorder=100000,c='black')
        ax_fit.scatter(x4_5pl,test_func(int(x4_5pl),parvals,header),zorder=100000,c='black')
            
    ax_fit.set_yscale("log")
    ax_fit.set_xscale("log")
    
    #set plot limits so that it is focussed on the data, to avoid scaling issues from fitted curve
    ax_fit.set_ylim(np.nanmin(y_data)/2,np.nanmax(y_data)*2) 
    ax_fit.set_xlim(np.nanmin(x_model),np.nanmax(x_model))
    
    ax_fit.grid()
    canvas_fit = FigureCanvasTkAgg(fig_fit, master=preview_window) 
    canvas_fit.draw()  
    canvas_fit.get_tk_widget().pack(fill="both",expand=True)
