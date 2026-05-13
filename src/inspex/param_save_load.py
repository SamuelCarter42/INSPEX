#%%Initial set up
import sys#for file path handling
import os#has general functions for file manipulation

import inspex
#breakpoint()
sys.path.append(f"{os.getcwd()}/Dependencies")#ensures that dependencies folder is available at point that modules are loaded

import tkinter as tk #this module contains most of the functions to run the gui
from tkinter import ttk
import lmfit #this module contains the functions for the curve fitting
import numpy as np #general mathematical operations
from scipy.special import erf #imports an erf function for use in some of the fitting operations
from matplotlib import pyplot as plt #general plotting operations
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)#allows plotting to a tkinter window

from tqdm import tqdm #for tracking progress of long iterables

import pickle #saves vars
import threading  #required to allow gui updates during code

#%%save load params from previous fits

def param_save(date,inst,spec_type, bpl_pres, therm_func_pres, gauss_pres,power_pres,kappa_pres,bpl_and_therm_pres, double_therm_func_pres, tpl_pres, qpl_pres, quint_pl_pres,redchi): #function that saves the parameters for later retrieval
    
    #open the file select dialogue to choose a save location
    files = [('Text Document','*.txt')]
    file_obj=tk.filedialog.asksaveasfile(filetypes = files, defaultextension=".txt")
    #file_obj=open(fileloc,'w')#define name of save file, opens it in write mode
    content=list()#set up empty list of contents
    global header #use/define header for all functions
    header=f"bpl_pres={bpl_pres}; therm_func_pres={therm_func_pres}; gauss_pres={gauss_pres}; power_pres={power_pres}; kappa_pres={kappa_pres}; bpl_and_therm_pres={bpl_and_therm_pres}; double_therm_func_pres={double_therm_func_pres}; tpl_pres={tpl_pres}; qpl_pres={qpl_pres}; quint_pl_pres={quint_pl_pres};\n"#define header. header defines which functions have been used to fit the data. ends with newline character
    content.append(header)#add header to top of file
    
    save_pars=dict()
    try:#try excpet statement is to validate inputs as integers
        if therm_func_pres==1:#if thermal function present, save parameter options from the gui for that function
            
            global frame_therm
            
            save_pars['T']=None if init_T_entry.get()=='None' else float(init_T_entry.get())            
            save_pars['amp']=None if init_amp_entry.get()=='None' else float(init_amp_entry.get())            
            save_pars['alpha']=None if init_alpha_entry.get()=='None' else float(init_alpha_entry.get())

            
        if bpl_pres==1:#if bpl function present, save parameter options from the gui for that function            
            
            save_pars['x1']=None if init_x1_entry.get()=='None' else float(init_x1_entry.get())
            save_pars['A2']=None if init_A2_entry.get()=='None' else float(init_A2_entry.get())
            save_pars['B2']=None if init_B2_entry.get()=='None' else float(init_B2_entry.get())
            save_pars['B']=None if init_B_entry.get()=='None' else float(init_B_entry.get())
            save_pars['A']=None if init_A_entry.get()=='None' else float(init_A_entry.get())
            save_pars['x0_bpl']=None if init_x0_bpl_entry.get()=='None' else float(init_x0_bpl_entry.get())
            save_pars['dx_bpl']=None if init_dx_bpl_entry.get()=='None' else float(init_dx_bpl_entry.get())

        if gauss_pres==1:#if gaussian function present, save parameter options from the gui for that function
            save_pars['gauss_centre']=None if init_gauss_centre_entry.get()=='None' else float(init_gauss_centre_entry.get())
            save_pars['gauss_amp']=None if init_gauss_amp_entry.get()=='None' else float(init_gauss_amp_entry.get())
            save_pars['sigma']=None if init_sigma_entry.get()=='None' else float(init_sigma_entry.get())
           
        if power_pres==1:#if single power law function present, save parameter options from the gui for that function
           save_pars['B_sing']=None if init_B_sing_entry.get()=='None' else float(init_B_sing_entry.get())
           save_pars['A_sing']=None if init_A_sing_entry.get()=='None' else float(init_A_sing_entry.get())
           save_pars['x0_sing']=None if init_x0_sing_entry.get()=='None' else float(init_x0_sing_entry.get())
           save_pars['dx_sing']=None if init_dx_sing_entry.get()=='None' else float(init_dx_sing_entry.get())
           

           
        if kappa_pres==1:#if kappa function present, save parameter options from the gui for that function

           save_pars['A_k']=None if init_A_k_entry.get()=='None' else float(init_A_k_entry.get())
           
           save_pars['T_k']=None if init_T_k_entry.get()=='None' else float(init_T_k_entry.get())
           
           save_pars['m_i']=None if init_m_i_entry.get()=='None' else float(init_m_i_entry.get())
           
           save_pars['n_i']=None if init_n_i_entry.get()=='None' else float(init_n_i_entry.get())

           save_pars['kappa']=None if init_kappa_entry.get()=='None' else float(init_kappa_entry.get())
           
           
        if bpl_and_therm_pres==1:
            save_pars['T_c']=None if init_T_c_entry.get()=='None' else float(init_T_c_entry.get())
            
            save_pars['amp_c']=None if init_amp_c_entry.get()=='None' else float(init_amp_c_entry.get())
            save_pars['alpha_c']=None if init_alpha_c_entry.get()=='None' else float(init_alpha_c_entry.get())
            
            save_pars['x1_c']=None if init_x1_c_entry.get()=='None' else float(init_x1_c_entry.get())
            save_pars['x0_c']=None if init_x0_c_entry.get()=='None' else float(init_x0_c_entry.get())
            save_pars['B2_c']=None if init_B2_c_entry.get()=='None' else float(init_B2_c_entry.get())
            
            save_pars['B_c']=None if init_B_c_entry.get()=='None' else float(init_B_c_entry.get())
            
        if double_therm_func_pres==1:#if double thermal function present, save parameter options from the gui for that function
            
            save_pars['T_d_1']=None if init_T_d_1_entry.get()=='None' else float(init_T_d_1_entry.get())
            save_pars['amp_d_1']=None if init_amp_d_1_entry.get()=='None' else float(init_amp_d_1_entry.get())
            save_pars['alpha_d_1']=None if init_alpha_d_1_entry.get()=='None' else float(init_alpha_d_1_entry.get())
            save_pars['T_d_2']=None if init_T_d_2_entry.get()=='None' else float(init_T_d_2_entry.get())
            
            save_pars['amp_d_2']=None if init_amp_d_2_entry.get()=='None' else float(init_amp_d_2_entry.get())
            save_pars['alpha_d_2']=None if init_alpha_d_2_entry.get()=='None' else float(init_alpha_d_2_entry.get())
        
        
        if tpl_pres==1:#if tpl function present, save parameter options from the gui for that function
            save_pars['x1_tpl']=None if init_x1_tpl_entry.get()=='None' else float(init_x1_tpl_entry.get())
            save_pars['x2_tpl']=None if init_x2_tpl_entry.get()=='None' else float(init_x2_tpl_entry.get())
            save_pars['A2_tpl']=None if init_A2_tpl_entry.get()=='None' else float(init_A2_tpl_entry.get())
            save_pars['B2_tpl']=None if init_B2_tpl_entry.get()=='None' else float(init_B2_tpl_entry.get())
            save_pars['B_tpl']=None if init_B_tpl_entry.get()=='None' else float(init_B_tpl_entry.get())
            save_pars['A_tpl']=None if init_A_tpl_entry.get()=='None' else float(init_A_tpl_entry.get())
            save_pars['A3_tpl']=None if init_A3_tpl_entry.get()=='None' else float(init_A3_tpl_entry.get())            
            save_pars['B3_tpl']=None if init_B3_tpl_entry.get()=='None' else float(init_B3_tpl_entry.get())
            save_pars['x0_tpl']=None if init_x0_tpl_entry.get()=='None' else float(init_x0_tpl_entry.get())
            save_pars['dx_tpl']=None if init_dx_tpl_entry.get()=='None' else float(init_dx_tpl_entry.get())
            
        if qpl_pres==1:#if qpl function present, save parameter options from the gui for that function
            save_pars['x1_qpl']=None if init_x1_qpl_entry.get()=='None' else float(init_x1_qpl_entry.get())
            save_pars['x2_qpl']=None if init_x2_qpl_entry.get()=='None' else float(init_x2_qpl_entry.get())
            save_pars['x3_qpl']=None if init_x3_qpl_entry.get()=='None' else float(init_x3_qpl_entry.get())
            save_pars['A2_qpl']=None if init_A2_qpl_entry.get()=='None' else float(init_A2_qpl_entry.get())
            save_pars['B2_qpl']=None if init_B2_qpl_entry.get()=='None' else float(init_B2_qpl_entry.get())
            save_pars['B_qpl']=None if init_B_qpl_entry.get()=='None' else float(init_B_qpl_entry.get())
            save_pars['A_qpl']=None if init_A_qpl_entry.get()=='None' else float(init_A_qpl_entry.get())
            save_pars['A3_qpl']=None if init_A3_qpl_entry.get()=='None' else float(init_A3_qpl_entry.get())
            save_pars['B3_qpl']=None if init_B3_qpl_entry.get()=='None' else float(init_B3_qpl_entry.get())
            save_pars['A4_qpl']=None if init_A4_qpl_entry.get()=='None' else float(init_A4_qpl_entry.get())
            save_pars['B4_qpl']=None if init_B4_qpl_entry.get()=='None' else float(init_B4_qpl_entry.get())
            save_pars['x0_qpl']=None if init_x0_qpl_entry.get()=='None' else float(init_x0_qpl_entry.get())
            save_pars['dx_qpl']=None if init_dx_qpl_entry.get()=='None' else float(init_dx_qpl_entry.get())
            
        if quint_pl_pres==1:#if 5pl function present, save parameter options from the gui for that function
            save_pars['x1_5pl']=None if init_x1_5pl_entry.get()=='None' else float(init_x1_5pl_entry.get())
            save_pars['x2_5pl']=None if init_x2_5pl_entry.get()=='None' else float(init_x2_5pl_entry.get())
            save_pars['x3_5pl']=None if init_x3_5pl_entry.get()=='None' else float(init_x3_5pl_entry.get())            
            save_pars['x4_5pl']=None if init_x4_5pl_entry.get()=='None' else float(init_x4_5pl_entry.get())
            save_pars['A2_5pl']=None if init_A2_5pl_entry.get()=='None' else float(init_A2_5pl_entry.get())
            save_pars['B2_5pl']=None if init_B2_5pl_entry.get()=='None' else float(init_B2_5pl_entry.get())
            save_pars['B_5pl']=None if init_B_5pl_entry.get()=='None' else float(init_B_5pl_entry.get())
            save_pars['A_5pl']=None if init_A_5pl_entry.get()=='None' else float(init_A_5pl_entry.get())
            save_pars['A3_5pl']=None if init_A3_5pl_entry.get()=='None' else float(init_A3_5pl_entry.get())
            save_pars['B3_5pl']=None if init_B3_5pl_entry.get()=='None' else float(init_B3_5pl_entry.get())
            save_pars['A4_5pl']=None if init_A4_5pl_entry.get()=='None' else float(init_A4_5pl_entry.get())
            save_pars['B4_5pl']=None if init_B4_5pl_entry.get()=='None' else float(init_B4_5pl_entry.get())            
            save_pars['A5_5pl']=None if init_A5_5pl_entry.get()=='None' else float(init_A5_5pl_entry.get())
            save_pars['B5_5pl']=None if init_B5_5pl_entry.get()=='None' else float(init_B5_5pl_entry.get())
            save_pars['x0_5pl']=None if init_x0_5pl_entry.get()=='None' else float(init_x0_5pl_entry.get())
            save_pars['dx_5pl']=None if init_dx_5pl_entry.get()=='None' else float(init_dx_5pl_entry.get())
            
    except ValueError as e:
           tk.messagebox.showerror("Invalid Input","Inputs should be floating point intergers")
           print(e)
           
           
    for i in range(len(save_pars)):    #for each parameter
        par_name=np.array(list(save_pars.keys()))[i]
        par_value=str(save_pars[par_name])
        
        content.append(f'{par_name}:{par_value}\n')#add parameter name and value to the content, with newline character at the end
        
    content.append(f'redchi:{redchi}\n')
    content.append(f'bic:{bic}\n')
    content=np.array(content)#make content into array
    file_obj.writelines(content)#write content to file
    
    file_obj.close()#close file
    
    
    
    
    
    
    
def param_load(date,inst,spec_type): #function to load data from the file
    file_obj=tk.filedialog.askopenfile()
    content=[x for x in file_obj]#reads file content into a list line by line
       
    file_obj.close()#close file
    
    parvals=dict()#set up dictionary for loaded parameters
    global header#define header for all functions
    header= content[0]#header is first line of file
    for i in content[1:]:#for remaining lines, get parameter name and value and save to pre-defined dict
        parvals[i.split(':')[0]]=float(i.split(':')[1][:-1])
        
    return header, parvals #output header and parameter dict

