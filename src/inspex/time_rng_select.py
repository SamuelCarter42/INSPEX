

import sys#for file path handling
import os#has general functions for file manipulation

import inspex
#breakpoint()
sys.path.append(f"{os.getcwd()}/Dependencies")#ensures that dependencies folder is available at point that modules are loaded

import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk

import numpy as np #general mathematical operations
from scipy.special import erf #imports an erf function for use in some of the fitting operations
from matplotlib import pyplot as plt #general plotting operations
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)#allows plotting to a tkinter window

import datetime as dt#handles general datetime operations
import pandas as pd #module for dataframe and time series handling




def time_rng_select(inst, start_time, end_time,spec_type_sel,resample_dur):#function for the time range window
    
    

#    window_inst.destroy()#closes instrument window window

    #load in data for selected probe. list of times, list of energies in keV, array of data in (times by energies), array of uncerts in (times by energies)
    if inst=="SolO-STEP":
        if dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S")<dt.datetime.strptime('2021/10/22',"%Y/%m/%d"):#time before recalibration:
            time_series_time,time_series_energies,time_series_data,time_series_uncert,epd_xyz_sectors,energy_lims=inspex.load_early_step_data(start_time, end_time)
        else:#post-recalibration, later data recalibrated and changed-must have different routines to interpret
            time_series_time,time_series_energies,time_series_data,time_series_uncert,epd_xyz_sectors,energy_lims=inspex.load_late_step_data(start_time, end_time)
                            
        if resample_dur!=None:
            time_series_time,time_series_data,time_series_uncert=inspex.resample_func(time_series_time,time_series_data,time_series_uncert,resample_dur)
    
    if inst=="STEREO STE":
        time_series_time,time_series_energies,time_series_data,time_series_uncert=inspex.stereo_data_load(start_time, end_time)
        if resample_dur!=None:
            time_series_time,time_series_data,time_series_uncert=inspex.resample_func(time_series_time,time_series_data,time_series_uncert,resample_dur)
    
    if inst=="SolO-EAS":
        low_e_cutoff=0.5
        date_for_spec=dt.datetime.strftime(dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S").date(),'%Y/%m/%d')
        epd_xyz_sectors=np.array([[-0.8412,  0.4396,  0.3149],
        [-0.8743,  0.457 ,  0.1635],
        [-0.8862,  0.4632, -0.    ],
        [-0.8743,  0.457 , -0.1635],
        [-0.8412,  0.4396, -0.315 ],
        [-0.7775,  0.5444,  0.3149],
        [-0.8082,  0.5658,  0.1635],
        [-0.8191,  0.5736,  0.    ],
        [-0.8082,  0.5659, -0.1634],
        [-0.7775,  0.5444, -0.3149],
        [-0.7008,  0.6401,  0.3149],
        [-0.7284,  0.6653,  0.1634],
        [-0.7384,  0.6744, -0.    ],
        [-0.7285,  0.6653, -0.1635],
        [-0.7008,  0.6401, -0.315 ]])
        #breakpoint()
        time_series_time,time_series_energies,time_series_data,time_series_uncert,energy_lims_eas=inspex.EAS_data_load(date_for_spec,start_time, end_time,epd_xyz_sectors,low_e_cutoff)
        if resample_dur!=None:
            time_series_time,time_series_data,time_series_uncert=inspex.resample_func(time_series_time,time_series_data,time_series_uncert,resample_dur)
    
    if inst=="SolO-EAS+STEP":
        #load eas
        low_e_cutoff=0.5
        date_for_spec=dt.datetime.strftime(dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S").date(),'%Y/%m/%d')
        epd_xyz_sectors=np.array([[-0.8412,  0.4396,  0.3149],
        [-0.8743,  0.457 ,  0.1635],
        [-0.8862,  0.4632, -0.    ],
        [-0.8743,  0.457 , -0.1635],
        [-0.8412,  0.4396, -0.315 ],
        [-0.7775,  0.5444,  0.3149],
        [-0.8082,  0.5658,  0.1635],
        [-0.8191,  0.5736,  0.    ],
        [-0.8082,  0.5659, -0.1634],
        [-0.7775,  0.5444, -0.3149],
        [-0.7008,  0.6401,  0.3149],
        [-0.7284,  0.6653,  0.1634],
        [-0.7384,  0.6744, -0.    ],
        [-0.7285,  0.6653, -0.1635],
        [-0.7008,  0.6401, -0.315 ]])
        
        eas_time_series_time,eas_time_series_energies,eas_time_series_data,eas_time_series_uncert,energy_lims_eas=inspex.EAS_data_load(date_for_spec,start_time, end_time,epd_xyz_sectors,low_e_cutoff)
        if resample_dur!=None:#resample eas
            eas_time_series_time,eas_time_series_data,eas_time_series_uncert=inspex.resample_func(eas_time_series_time,eas_time_series_data,eas_time_series_uncert,resample_dur)
        
        #load step
        if dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S")<dt.datetime.strptime('2021/10/22',"%Y/%m/%d"):#time before recalibration:
            step_time_series_time,step_time_series_energies,step_time_series_data,step_time_series_uncert,epd_xyz_sectors,energy_lims=inspex.load_early_step_data(start_time, end_time)
        else:#post-recalibration, later data recalibrated and changed-must have different routines to interpret
            step_time_series_time,step_time_series_energies,step_time_series_data,step_time_series_uncert,epd_xyz_sectors,energy_lims=inspex.load_late_step_data(start_time, end_time)
        
        if resample_dur!=None:#resample step
            step_time_series_time,step_time_series_data,step_time_series_uncert=inspex.resample_func(step_time_series_time,step_time_series_data,step_time_series_uncert,resample_dur)
        
        #after resampling, time series should line up. we take eas
        time_series_time=eas_time_series_time
        #breakpoint()
        #combine the data into one array
        time_series_data=np.concatenate((eas_time_series_data,step_time_series_data), axis=1)
        time_series_uncert=np.concatenate((eas_time_series_uncert,step_time_series_uncert), axis=1)
        time_series_energies=np.concatenate((eas_time_series_energies,step_time_series_energies))
    
    if inst=="SolO-EAS+STEP+FAF":
        #load eas
        low_e_cutoff=0.5
        date_for_spec=dt.datetime.strftime(dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S").date(),'%Y/%m/%d')
        epd_xyz_sectors=np.array([[-0.8412,  0.4396,  0.3149],
        [-0.8743,  0.457 ,  0.1635],
        [-0.8862,  0.4632, -0.    ],
        [-0.8743,  0.457 , -0.1635],
        [-0.8412,  0.4396, -0.315 ],
        [-0.7775,  0.5444,  0.3149],
        [-0.8082,  0.5658,  0.1635],
        [-0.8191,  0.5736,  0.    ],
        [-0.8082,  0.5659, -0.1634],
        [-0.7775,  0.5444, -0.3149],
        [-0.7008,  0.6401,  0.3149],
        [-0.7284,  0.6653,  0.1634],
        [-0.7384,  0.6744, -0.    ],
        [-0.7285,  0.6653, -0.1635],
        [-0.7008,  0.6401, -0.315 ]])
        
        eas_time_series_time,eas_time_series_energies,eas_time_series_data,eas_time_series_uncert,energy_lims_eas=inspex.EAS_data_load(date_for_spec,start_time, end_time,epd_xyz_sectors,low_e_cutoff)
        if resample_dur!=None:#resample eas
            eas_time_series_time,eas_time_series_data,eas_time_series_uncert=inspex.resample_func(eas_time_series_time,eas_time_series_data,eas_time_series_uncert,resample_dur)
        
        #load step
        if dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S")<dt.datetime.strptime('2021/10/22',"%Y/%m/%d"):#time before recalibration:
            step_time_series_time,step_time_series_energies,step_time_series_data,step_time_series_uncert,epd_xyz_sectors,energy_lims=inspex.load_early_step_data(start_time, end_time)
        else:#post-recalibration, later data recalibrated and changed-must have different routines to interpret
            step_time_series_time,step_time_series_energies,step_time_series_data,step_time_series_uncert,epd_xyz_sectors,energy_lims=inspex.load_late_step_data(start_time, end_time)
        
        if resample_dur!=None:#resample step
            step_time_series_time,step_time_series_data,step_time_series_uncert=inspex.resample_func(step_time_series_time,step_time_series_data,step_time_series_uncert,resample_dur)
        
        #after resampling, time series should line up. we take eas
        time_series_time=eas_time_series_time
        #breakpoint()
        #combine the data into one array
        time_series_data=np.concatenate((eas_time_series_data,step_time_series_data), axis=1)
        time_series_uncert=np.concatenate((eas_time_series_uncert,step_time_series_uncert), axis=1)
        time_series_energies=np.concatenate((eas_time_series_energies,step_time_series_energies))
        #breakpoint()
        
    #slice loaded data to range selected by user, as generally loads in full days    
    #set range to user defined fitting limits
    x_data_sliced=list()
    y_data_sliced=list()
    uncert_sliced=list()
    for pos,time in enumerate(time_series_time):
        #breakpoint()
        #print(time)
        #account for potential differences in time format
        if 'T' in str(time)[:-3]:
            time=dt.datetime.strptime(str(time)[:-3],"%Y-%m-%dT%H:%M:%S.%f")
        else:time=dt.datetime.strptime(str(time)[:-3],"%Y-%m-%d %H:%M:%S.%f")
        
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
        
        if inst=="SolO-STEP":plotchans=[0, 4, 8, 12, 16, 20, 24, 28,30]
        else:plotchans=np.linspace(0, len(time_series_energies) - 1, 9, dtype=int)
        #breakpoint()
        for channel in plotchans:        
            label=f'{round(time_series_energies[channel],2)} keV'
            
            pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=label,linewidth=0.75,rot=30,fontsize=5)#
        




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



        fig_TS.canvas.draw()
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
        
        #to allow spectrum alignment for eas and solo, need to check for instrument
        #and then generate the two halves and align into one

        
        if inst=="SolO-EAS+STEP+FAF":
            if selected_func=='fluence':
                #generate for EAS
                spec_eas,spec_uncert_eas=inspex.fluence_spec_gen(time_series_time,eas_time_series_energies,eas_time_series_data,eas_time_series_uncert,bg_mintime,bg_maxtime,spec_mintime,spec_maxtime)
                #generate for step
                spec_step,spec_uncert_step=inspex.fluence_spec_gen(time_series_time,step_time_series_energies,step_time_series_data,step_time_series_uncert,bg_mintime,bg_maxtime,spec_mintime,spec_maxtime)
                #generate alignment factor
                fact1=spec_step[4]/spec_eas[-1]
                fact2=spec_step[0]/spec_eas[-2]

                avgfact=np.mean([fact1,fact2])
                
                #combine spectra
                #change format of eas to array to allow operation over full spectrum before changing back
                spec=list(np.array(spec_eas)*avgfact)
                spec.extend(spec_step)
                spec_uncert=list(np.array(spec_uncert_eas)*avgfact)
                spec_uncert.extend(spec_uncert_step)        
                
                spec_type='fluence'
                
                
            if selected_func=='peak flux':
                #generate for EAS
                #breakpoint()
                spec_eas,spec_uncert_eas=inspex.peak_flux_spec_gen(time_series_time,eas_time_series_energies,eas_time_series_data,eas_time_series_uncert,bg_mintime,bg_maxtime,spec_mintime,spec_maxtime)
                #generate for step
                spec_step,spec_uncert_step=inspex.peak_flux_spec_gen(time_series_time,step_time_series_energies,step_time_series_data,step_time_series_uncert,bg_mintime,bg_maxtime,spec_mintime,spec_maxtime)
                #generate alignment factor
                fact1=spec_step[4]/spec_eas[-1]
                fact2=spec_step[0]/spec_eas[-2]

                avgfact=np.mean([fact1,fact2])
                
                #combine spectra
                #change format of eas to array to allow operation over full spectrum before changing back
                spec=list(np.array(spec_eas)*avgfact)
                spec.extend(spec_step)
                spec_uncert=list(np.array(spec_uncert_eas)*avgfact)
                spec_uncert.extend(spec_uncert_step)
                
                spec_type='peak_flux'
        
        else:
            if selected_func=='fluence':
                spec,spec_uncert=inspex.fluence_spec_gen(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime,spec_mintime,spec_maxtime)
                spec_type='fluence'
                
                
            if selected_func=='peak flux':
                #breakpoint()
                spec,spec_uncert=inspex.peak_flux_spec_gen(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime,spec_mintime,spec_maxtime)
                spec_type='peak_flux'
        
        
        
        date= spec_mintime.strftime("%d-%m-%Y")        
        TS_window.destroy()
        plt.close(fig_TS)
        inspex_fn(time_series_energies, spec, spec_uncert, date, inst, spec_type)
    

    
    button = tk.Button(master=TS_window, text="Select this background and spectrum range", command=TS_Select_btn_hndl)
    button.pack(side=tk.TOP)
    
    
    #for a selection of energy channels, convert from array to pd.series, resample for clarity then plot with appropriate label
    if inst=="SolO-STEP":plotchans=[0, 4, 8, 12, 16, 20, 24, 28,30]
    else:plotchans=np.linspace(0, len(time_series_energies) - 1, 9, dtype=int)
    for channel in plotchans:
        pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=f'{round(time_series_energies[channel],2)} keV')#
    ax_TS.set_ylim(bottom=1)
    #fig_TS.autofmt_xdate()


    fig_TS.canvas.draw()
    canvas_TS = FigureCanvasTkAgg(fig_TS, master=TS_window) 
    fig_TS.tight_layout()
    canvas_TS.draw()  

    canvas_TS.get_tk_widget().pack()


    TS_window.mainloop()
