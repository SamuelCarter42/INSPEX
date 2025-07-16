import os
import subprocess
import wexpect
import sys
import datetime as dt#handles general datetime operations
import idl_locator#for finding the user's IDL install
from idl_command_execute import run_idl_command

def stereo_data_load_calib(start_time,end_time,date,this_folder):
    
       
    #get the folder where this file is stored
    dependencies_folder=os.path.dirname(os.path.realpath(__file__))
    
    dependencies_folder=str(dependencies_folder).replace('\\', '/')
    
    #obtain the data for the specified dates
    data_dir=fr'{this_folder}/data/Stereo'
    
    idl_command=list()
    
    #make sure IDL session is clean
    idl_command.append(fr".reset_session")

    
    #compile the data load command
    idl_command.append(fr".r '{dependencies_folder}/SPEDAS/spedas_6_0/general/missions/stereo/st_ste_load.pro'")
    
    
    
    
    #ensure that the data directory is added to the environment
    idl_command.append(fr"setenv,'ROOT_DATA_DIR={data_dir}'")
    
    
    
    
    
    start_date_input=dt.datetime.strptime(start_time,"%Y/%m/%d %H:%M:%S").strftime('%y-%m-%d/%H:%M:%S')
    end_date_input=dt.datetime.strptime(end_time,"%Y/%m/%d %H:%M:%S").strftime('%y-%m-%d/%H:%M:%S')
    

    #run the data load command
    idl_command.append(fr"st_ste_load,probes='a',TRANGE=['{start_date_input}','{end_date_input}']")
    
    
    
    #%% the conversion to flux and saving the structure
    #compile the data conversion command
    idl_command.append(fr".r '{dependencies_folder}/ste_cnt_to_flux.pro'")
    
    
    #run the data conversion command for sensor D0
    idl_command.append(fr"STE_CNT_TO_FLUX,'sta_ste_D0',sum_bin=INDGEN(32),err=1")
    
    
    #place D0 data into structure
    idl_command.append(fr"get_data,'sta_ste_D0_f',data=structure0")
    
    

    
    #save the D0 data
    idl_command.append(fr"save,structure0,filename='{data_dir}/Processed_data/STA_STE_D0_{date}_range.sav'")
    
    
    
    
    
    #run the data conversion command for sensor D1
    idl_command.append(fr"STE_CNT_TO_FLUX,'sta_ste_D1',sum_bin=INDGEN(32),err=1")
    
    
    #place D1 data into structure
    idl_command.append(fr"get_data,'sta_ste_D1_f',data=structure1")
    
    
    #save the D1 data
    idl_command.append(fr"save,structure1,filename='{data_dir}/Processed_data/STA_STE_D1_{date}_range.sav'")
    
    
    #run the data conversion command for sensor D2
    idl_command.append(fr"STE_CNT_TO_FLUX,'sta_ste_D2',sum_bin=INDGEN(32),err=1")
    
    
    #place D2 data into structure
    idl_command.append(fr"get_data,'sta_ste_D2_f',data=structure2")
    
    
    #save the D2 data
    idl_command.append(fr"save,structure2,filename='{data_dir}/Processed_data/STA_STE_D2_{date}_range.sav'")
    
    
    #run the data conversion command for sensor D3
    idl_command.append(fr"STE_CNT_TO_FLUX,'sta_ste_D3',sum_bin=INDGEN(32),err=1")
    
    
    #place D3 data into structure
    idl_command.append(fr"get_data,'sta_ste_D3_f',data=structure3")
    

    #save the D3 data
    idl_command.append(fr"save,structure3,filename='{data_dir}/Processed_data/STA_STE_D3_{date}_range.sav'")
    
    idl_command.append('print,"hello"')
                       
    
                       
    
    #run the command list we generated above
    idl_command_joined = '\n'.join(idl_command)

    run_idl_command(idl_command_joined)
    
