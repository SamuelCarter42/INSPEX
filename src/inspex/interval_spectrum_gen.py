import numpy as np
import inspex


#%%spectrum generation for a single time
def interval_spec_gen(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime,intervals):
    bg_spec, bg_spec_uncert=inspex.avg_bg_calc(time_series_time,time_series_energies,time_series_data,time_series_uncert,bg_mintime,bg_maxtime)#background generating function
    #print(time_series_time)
    spectra=list()
    for interval in intervals:
        
        #find time in data closest to selected time
        closest=min(time_series_time, key=lambda x: abs(x - interval))
        
        pos=list(time_series_time).index(closest)
        this_spec_raw=time_series_data[pos,:]
        this_spec_uncert_raw=time_series_uncert[pos,:]

        this_spec=this_spec_raw-list(bg_spec.values())
        this_spec_uncert=np.sqrt(np.array(this_spec_uncert_raw)**2+np.array(list(bg_spec_uncert.values()))**2)
        spectra.append((this_spec,this_spec_uncert))
    
    
    return spectra