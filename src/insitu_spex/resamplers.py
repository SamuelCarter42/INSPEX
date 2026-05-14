
import numpy as np
import pandas as pd

def custom_resampler(arraylike):
        
    arraylike2=np.nanmean(arraylike)
    if np.isnan(arraylike2).any():
        arraylike2=np.nan#this should prevent the data from being plotted while maintaining consitent length

    return (arraylike2)

def custom_resampler_uncert(arraylike):
    try:#if the uncert resampling gets broken, this should avoid the issue
        arraylike2=np.sqrt(sum(arraylike**2))/(len(arraylike))
    except:arraylike2=0
        
    if np.isnan(arraylike2).any():
        arraylike2=0
    return (arraylike2)
def resample_func(time,data,uncert,resample_dur):
    
    series_list=list()
    uncert_series_list=list()
    #must transpose to make E by t for resampling
    for e_row in data.transpose():    #resample each energy row of data array
        data_series=pd.Series(e_row,time).resample(resample_dur).apply(custom_resampler)
        series_list.append(data_series)
    
    for e_row in uncert.transpose():#resample each energy row of uncert array
        uncert_series=pd.Series(e_row,time).resample(resample_dur).apply(custom_resampler_uncert)
        uncert_series_list.append(uncert_series)
    
    res_data=list()
    res_uncert=list()
    
    for e_row in series_list:    #unpack each row of the data series array
        this_chan=e_row.values
        res_data.append(this_chan)
    
    for e_row in uncert_series_list:    #unpack each row of the uncert series array
        this_chan=e_row.values
        res_uncert.append(this_chan)
    
    res_times=np.array(series_list[0].index)#times will be same for every row, just take first
    #must transpose to make t by E for rest of process
    return res_times,np.array(res_data).transpose(),np.array(res_uncert).transpose()