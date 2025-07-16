import os
import subprocess
import wexpect
import sys
import datetime as dt#handles general datetime operations
import idl_locator#for finding the user's IDL install
from idl_command_execute import run_idl_command# my script to run IDL commands through a python call


def wind_data_load_calib(start_time,end_time,date,this_folder):
    
    #download all the relevant data files required for the IDL script
    #%%set up file architecture required for the code
    
    #%% load swe data
    
    #%%rename swe data
    
    #%% load mfi data
    
    #%%load waves data
    
    #%%load orbit asciis
    
    #%%load 3dp data
    
    #%%run the correction routine for the data
    
    #%%save the data
    
    #%%run the correction routine for the uncerts
    
    #%%save the uncerts    
    
