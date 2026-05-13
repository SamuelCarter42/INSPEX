# -*- coding: utf-8 -*-

#%%Initial set up
import sys#for file path handling
import os#has general functions for file manipulation

import inspex
#breakpoint()
sys.path.append(f"{os.getcwd()}/Dependencies")#ensures that dependencies folder is available at point that modules are loaded

import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk
import pandas as pd #module for dataframe and time series handling
import re#for handling regexs to validate inputs





def instrument_choice():#function for the instrument choice window
    global window_inst
    window_inst = tk.Tk()#define window
    window_inst.title('Select instrument and time range for spectral calculations')
    window_inst.minsize(400, 400)
    
    frame_instopts=tk.Frame(master=window_inst)
    
    
    greeting = tk.Label(master=frame_instopts,text="Inspex fitting GUI")#window name. MUST only have one tk.Tk(), all else must be .toplevel else crashes. This is .TK as is first to open 
    greeting.pack()
    OPTIONS = [
        "SolO EPD STEP",
        "[IDL REQUIRED] STEREO STE",
        "SolO SWA EAS, FOV aligned to EPD STEP",
        "Combined SolO SWA EAS and EPD STEP",
        "Cross calibrated SolO SWA EAS and EPD STEP"
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
    TYP_OPTIONS = [
        "fluence","peak flux","flux at set time(s)"
        
        ]     
    var_spec_type = tk.StringVar()
    var_spec_type.set(TYP_OPTIONS[1]) # default value    
    spec_opts = tk.OptionMenu(frame_specopts, var_spec_type, *TYP_OPTIONS)
    spec_opts.pack()
    frame_specopts.pack(side=tk.LEFT)
    
    #handling resampling selection
    RES_OPTIONS = [
        "No Resampling",
        "1 minute resampling",
        "2 minute resampling",
        "5 minute resampling",
        "10 minute resampling",
        "30 minute resampling"
        ]     
    res_var = tk.StringVar()
    res_var.set(RES_OPTIONS[0]) # default value
    
    res_opts = tk.OptionMenu(frame_instopts, res_var, *RES_OPTIONS)
    res_opts.pack()
    
    
    
    #define spectrum loading, with file validation
    def spec_file_validate(file_obj):
        path,exten=os.path.splitext(file_obj.name)
        if exten=='.txt' or exten=='.csv':
            return True
        else:return False
        
        
        
    def load_spec_hndl():
        file_obj=tk.filedialog.askopenfile()
        if not spec_file_validate(file_obj):#if correctly formatted 
            tk.messagebox.showerror("Invalid Input",'Loaded spectrum must be .txt in correct format, saved via the fitting GUI')
        else:        
            spec_df=pd.read_csv(file_obj)
            file_obj.close()
            load_energies=spec_df['energies'].values
            load_fluxes=spec_df['fluxes'].values
            load_uncerts=spec_df['errors'].values
            date=spec_df['date'].values[0]
            inst=spec_df['inst'].values[0]
            spec_type=spec_df['spec_type'].values[0]
            window_inst.destroy()#closes current window
        
            inspex_fn(load_energies, load_fluxes, load_uncerts, date, inst, spec_type)

    

        
    #create spectrum load button
    load_spec_button=tk.Button(
    master=window_inst,
    text="Load Previously created INSPEX (.txt) spectrum",
    width=35,
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
        if not (validate_date(start_time) and validate_date(end_time)):#if correctly formatted 
            tk.messagebox.showerror("Invalid Input",'Dates must have format: YYYY/mm/dd HH:MM:SS')
        else:
            selected_func=variable.get()
            if selected_func=='SolO EPD STEP':
                inst="SolO-STEP"
            if selected_func=='[IDL REQUIRED] STEREO STE':
                inst="STEREO STE"
            if selected_func=='SolO SWA EAS, alligned to EPD STEP':
                inst="SolO-EAS"
            if selected_func== "Combined SolO SWA EAS and EPD STEP":
                inst="SolO-EAS+STEP"
            if selected_func== "Cross calibrated SolO SWA EAS and EPD STEP":
                inst="SolO-EAS+STEP+FAF"
                    
            #handling the resampling selection
            res_sel=res_var.get()
            if res_sel=="No Resampling":
                if inst=="SolO-EAS+STEP" or inst=="SolO-EAS+STEP+FAF":
                    tk.messagebox.showerror("Invalid Input",'Combined instruments must have some resampling to align timings')
                else:resample_dur=None#no resampling
            else:
                resample_dur=res_sel.split()[0]+"min"
                    
            
                    
            window_inst.destroy()#closes current window
            window_inst.update()
        
            spec_type_sel=var_spec_type.get()
            
            if spec_type_sel=="fluence" or spec_type_sel=="peak flux":
                
                inspex.time_rng_select(inst,start_time,end_time,spec_type_sel,resample_dur)#runs time range selection function
            
            elif spec_type_sel=="flux at set time(s)":
                intervals_select(inst,start_time,end_time,spec_type_sel,resample_dur)
        
        



    button = tk.Button(master=frame_instopts, text="Choose Instrument, data dates, and spectrum type", command=inst_opts_select)
    button.pack(side=tk.BOTTOM)

    frame_instopts.pack()   


    window_inst.mainloop()