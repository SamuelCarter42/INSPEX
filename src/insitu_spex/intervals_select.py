from .resamplers import resample_func
from .step_data_load import load_early_step_data,load_late_step_data
from .stereo_data_load import stereo_data_load
from .eas_data_load import EAS_data_load
from .interval_spectrum_gen  import interval_spec_gen 
from .fitting_and_resids import fitting
from .fitting_gui import fitting_gui
from . import state  #shared cross-module state


import numpy as np
import datetime as dt#handles general datetime operations
import pandas as pd #module for dataframe and time series handling
import matplotlib.pyplot as plt
import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)#allows plotting to a tkinter window

#%%window to select time intervals

def intervals_select(inst,start_time,end_time,spec_type_sel,resample_dur):
    
    #    window_inst.destroy()#closes instrument window window
    
    #load in data for selected probe. list of times, list of energies in keV, array of data in (times by energies), array of uncerts in (times by energies)
    if inst=="SolO-STEP":
        if dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S")<dt.datetime.strptime('2021/10/22',"%Y/%m/%d"):#time before recalibration:
            time_series_time,time_series_energies,time_series_data,time_series_uncert,epd_xyz_sectors,energy_lims=load_early_step_data(start_time, end_time)
        else:#post-recalibration, later data recalibrated and changed-must have different routines to interpret
            time_series_time,time_series_energies,time_series_data,time_series_uncert,epd_xyz_sectors,energy_lims=load_late_step_data(start_time, end_time)
                            
        if resample_dur!=None:
            time_series_time,time_series_data,time_series_uncert=resample_func(time_series_time,time_series_data,time_series_uncert,resample_dur)
    
    if inst=="STEREO STE":
        time_series_time,time_series_energies,time_series_data,time_series_uncert=stereo_data_load(start_time, end_time)
        if resample_dur!=None:
            time_series_time,time_series_data,time_series_uncert=resample_func(time_series_time,time_series_data,time_series_uncert,resample_dur)
    
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
        time_series_time,time_series_energies,time_series_data,time_series_uncert,energy_lims_eas=EAS_data_load(date_for_spec,start_time, end_time,epd_xyz_sectors,low_e_cutoff)
        if resample_dur!=None:
            time_series_time,time_series_data,time_series_uncert=resample_func(time_series_time,time_series_data,time_series_uncert,resample_dur)
    
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
        
        eas_time_series_time,eas_time_series_energies,eas_time_series_data,eas_time_series_uncert,energy_lims_eas=EAS_data_load(date_for_spec,start_time, end_time,epd_xyz_sectors,low_e_cutoff)
        if resample_dur!=None:#resample eas
            eas_time_series_time,eas_time_series_data,eas_time_series_uncert=resample_func(eas_time_series_time,eas_time_series_data,eas_time_series_uncert,resample_dur)
        
        #load step
        if dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S")<dt.datetime.strptime('2021/10/22',"%Y/%m/%d"):#time before recalibration:
            step_time_series_time,step_time_series_energies,step_time_series_data,step_time_series_uncert,epd_xyz_sectors,energy_lims=load_early_step_data(start_time, end_time)
        else:#post-recalibration, later data recalibrated and changed-must have different routines to interpret
            step_time_series_time,step_time_series_energies,step_time_series_data,step_time_series_uncert,epd_xyz_sectors,energy_lims=load_late_step_data(start_time, end_time)
        
        if resample_dur!=None:#resample step
            step_time_series_time,step_time_series_data,step_time_series_uncert=resample_func(step_time_series_time,step_time_series_data,step_time_series_uncert,resample_dur)
        
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
        
        eas_time_series_time,eas_time_series_energies,eas_time_series_data,eas_time_series_uncert,energy_lims_eas=EAS_data_load(date_for_spec,start_time, end_time,epd_xyz_sectors,low_e_cutoff)
        if resample_dur!=None:#resample eas
            eas_time_series_time,eas_time_series_data,eas_time_series_uncert=resample_func(eas_time_series_time,eas_time_series_data,eas_time_series_uncert,resample_dur)
        
        #load step
        if dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S")<dt.datetime.strptime('2021/10/22',"%Y/%m/%d"):#time before recalibration:
            step_time_series_time,step_time_series_energies,step_time_series_data,step_time_series_uncert,epd_xyz_sectors,energy_lims=load_early_step_data(start_time, end_time)
        else:#post-recalibration, later data recalibrated and changed-must have different routines to interpret
            step_time_series_time,step_time_series_energies,step_time_series_data,step_time_series_uncert,epd_xyz_sectors,energy_lims=load_late_step_data(start_time, end_time)
        
        if resample_dur!=None:#resample step
            step_time_series_time,step_time_series_data,step_time_series_uncert=resample_func(step_time_series_time,step_time_series_data,step_time_series_uncert,resample_dur)
        
        #after resampling, time series should line up. we take eas
        time_series_time=eas_time_series_time
        #breakpoint()
        #combine the data into one array
        time_series_data=np.concatenate((eas_time_series_data,step_time_series_data), axis=1)
        time_series_uncert=np.concatenate((eas_time_series_uncert,step_time_series_uncert), axis=1)
        time_series_energies=np.concatenate((eas_time_series_energies,step_time_series_energies))
        #breakpoint()
    
    spec_type="intervals" #the type of the spectra this generates
    
    #slice loaded data to range selected by user, as generally loads in full days    
    #set range to user defined fitting limits
    x_data_sliced=list()
    y_data_sliced=list()
    uncert_sliced=list()
    #breakpoint()
    for pos,time in enumerate(time_series_time):
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
            fig_TS.autofmt_xdate()

            ax_TS.tick_params(axis='x', rotation=45, labelright=False)

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
            
            pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=label,linewidth=0.75,rot=45)#

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
            
        for tick in ax_TS.get_xticklabels():
            tick.set_rotation(45)
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
            
            pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=label,linewidth=0.75,rot=45)#
        fig_TS.autofmt_xdate()

        ax_TS.tick_params(axis='x', rotation=45, which='major', labelright=False)

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
            
            pd.Series(time_series_data_raw[:,channel],time_series_time_raw).resample(tsres).mean().plot(ax = ax_TS, logy=True, label=label,linewidth=0.75,rot=45)#
        fig_TS.autofmt_xdate()

        ax_TS.tick_params(axis='x', rotation=45, which='major', labelright=False)  # simple & reliable

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
        for tick in ax_TS.get_xticklabels():
            tick.set_rotation(45)
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
                
        
        #to allow spectrum alignment for eas and solo, need to check for instrument
        #and then generate the two halves and align into one
        if inst=="SolO-EAS+STEP+FAF":
            #generate the list of spectra for each instrument
            spectra_eas=interval_spec_gen(time_series_time, eas_time_series_energies, eas_time_series_data, eas_time_series_uncert, bg_mintime, bg_maxtime, intervals)
            spectra_step=interval_spec_gen(time_series_time, step_time_series_energies, step_time_series_data, step_time_series_uncert, bg_mintime, bg_maxtime, intervals)
   
            #cross calibrate and combine each spectrum 
            spectra=list()
            for ind,spec_eas in enumerate(spectra_eas):
                spec_step=spectra_step[ind]
                spec_uncert_eas=spec_eas[1]
                spec_uncert_step=spec_step[1]
                spec_eas=spec_eas[0]
                spec_step=spec_step[0]
                
                
                fact1=spec_step[4]/spec_eas[-1]
                fact2=spec_step[0]/spec_eas[-2]

                avgfact=np.mean([fact1,fact2])
                
                #combine spectra
                #change format of eas to array to allow operation over full spectrum before changing back
                spec=list(np.array(spec_eas)*avgfact)
                spec.extend(spec_step)
                spec_uncert=list(np.array(spec_uncert_eas)*avgfact)
                spec_uncert.extend(spec_uncert_step)
                spectra.append((spec,spec_uncert))
        else:
            spectra=interval_spec_gen(time_series_time, time_series_energies, time_series_data, time_series_uncert, bg_mintime, bg_maxtime, intervals)
        
        
        
        
        
        
        
        date= min(time_series_time)        
        inters_window.destroy()
        plt.close(fig_TS)
        
        fitted_params=np.empty([len(spectra),2],dtype=dict)
        if manl_loop: #if user wishes to do all fits mannually
            for count,i in enumerate(spectra):#fit all the spectra
                spec=i[0]
                spec_uncert=i[1]
                fitting_gui(time_series_energies, spec, spec_uncert, intervals[count], inst, spec_type)
                fitted_params[count,0]=state.parvals
                fitted_params[count,1]=state.param_uncert_calced
                

        else: #auto loop intervals after first one
            fitting_gui(time_series_energies, spectra[0][0], spectra[0][1], intervals[0], inst, spec_type)
            fitted_params[0,0]=state.parvals
            fitted_params[0,1]=state.param_uncert_calced
            output=state.parvals
            for count,i in enumerate(spectra[1:]):#fit all the spectra
                spec=i[0]
                spec_uncert=i[1]
                state.parvals,state.param_uncert_calced,x_data_E_sliced=fitting(state.header,output,state.vary,state.minval,state.maxval,time_series_energies,spec,spec_uncert,state.fitmin,state.fitmax)
                #state.fit_window.quit()
                #state.preview_window.quit()
                fitted_params[count+1,0]=state.parvals
                fitted_params[count+1,1]=state.param_uncert_calced
                output=state.parvals

        if state.fit_window is not None:
            #close any open figues
            #state.fit_window.destroy()
            state.fit_window=None
            
        if state.preview_window is not None:
            #close any open figues
            #state.preview_window.destroy()
            state.preview_window=None
        
        
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
            if state.header[index] == '1':  # Check if the corresponding function is present
                for button_text in buttons:
                    create_button(button_text, button_text, invl_res_display_opts)
        
        invl_res_display_opts.pack()
        invl_res_display_wind.mainloop()
        
        
    
        
    
    button = tk.Button(master=inters_window, text="Select this background and spectrum intervals", command=TS_Select_btn_hndl)
    button.pack(side=tk.BOTTOM)
    
    
    
    
    
    
    inters_window.mainloop()