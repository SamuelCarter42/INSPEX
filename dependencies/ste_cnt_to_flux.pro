;+
;PROCEDURE:	ste_cnt_to_flux
;PURPOSE:	
;  Convert count to flux, for STE data, not for public use
;
;INPUTS:	
;  name: tplot variable from load_sta_l1_ste, etc.
;KEYWORDS:
;   sum_bin:  what bins to sum over for energy bands, ~intarr[32] 
;             e.g. [-1,0,0,0,1,1,1,1,.....] , then ignore bin#0,  bin# 1,2,3 for band #0;
;            bin# 4,5,6 for band #1. 
; gm_factor: geometric factors
;       df : if set, get distribution function
;  delta_t : if set, average the data due to delta_t ( in seconds)
;        p : if set, the flux and df is converted for protons 
;        U : if set, convert the flux for STE-U detectors
;      err : if set, calculate the standard error of mean and standard deviation for flux
;
;OUTPUTS:
;   ee : central energy of energy band
; e0,e1: lower and upper bound of energy band
;
;CREATED BY:	Linghua Wang
;LAST MODIFIED:	03/2/09
;   Feb 27, 2009  -- fix the bug at the presence of data gaps in time
;   
;   Mar 2, 2009  -- add the uncertainty estimate for time-average
;-


pro ste_cnt_to_flux,name,sum_bin = sum_bin,gm_factor = gm_factor,ee=ee,e0=e0,e1=e1, df = df, delta_t = delta_t, p = p, U= U, err =err

 nn = n_elements(name)
 
 if not keyword_set (p) then mass =9.1094d-31 else mass = 1.67262d-27  ; in kilogram for electrons or protons

 if not keyword_set(sum_bin) then sum_bin = indgen(32)

 if not keyword_set(gm_factor) then begin
   ;gm_factor = replicate(0.43*0.1,nn)  ;  old rough gf in cm^2 * str for STE one pixel 
   if not keyword_set(U) then begin
      gm_factor = [0.029,0.029,0.021,0.021]  ;  for STE-D
      print,'STE-D'
   endif else  begin
      gm_factor = [0.022,0.022,0.0173,0.0173]  ;  for STE-U
      print,'STE-U'
   endelse
 endif else begin
   if n_elements(gm_factor) eq 1 then gm_factor = replicate(gm_factor,nn)
 endelse  
 
 emin = 1.5d3  ; in eV, the lowest energy for bin #0, could be incorrect
 
 n_e = max(sum_bin)
 
 ee = fltarr(n_e+1,nn)
 e0 = fltarr(n_e+1,nn)
 e1 = fltarr(n_e+1,nn)
 
 for i =0, nn-1 do begin
   get_data, name(i),data=pp,dl=dl
   yy = pp.y
   vv = pp.v* 1.e3 ; in eV
   xx = pp.x
   nt2 = n_elements(xx)
   
   if keyword_set(delta_t) then begin
      ni = -1
      nt = 0
      xx2 = dblarr(nt2)
      yy2 = fltarr(nt2,32)
      vv2 = fltarr(nt2,32)
      dtt0 = dblarr(nt2)
      err0 = fltarr(nt2,32)
      err2 = fltarr(nt2,32)
      
      while(ni lt nt2 - 1) do begin
        tt0 = xx(ni+1)
        tlist = where((xx lt tt0 + delta_t - 0.5) and (xx  ge tt0 - 1.e-6), tind)
        if tind gt 0L then begin
           ;print, nt
           xx2(nt) = average(xx(tlist),/nan)
           yy2(nt,0:31) = total(yy(tlist,*),1,/nan)
           vv2(nt,0:31) = average(vv(tlist,*),1,/nan)
           dtt0(nt) = tind*10.d0   ;; original time resolution is 10 s
           nt =  nt+1L
           ni = tlist(tind-1) 
           
           ;stop
           ;print,nt,ni,tind
           ;print,tlist
           ;print,time_string(xx(tlist))
           ;if ni gt 880 then stop           
        endif
      endwhile
      
      xx = xx2(0:nt-1)
      yy = yy2(0:nt-1,*)
      vv = vv2(0:nt-1,*)
      tt = dtt0(0:nt-1)
   endif else  begin
      nt = nt2
      tt = 10.0
   endelse

   dt = fltarr(nt,n_e+1)
   denergy = fltarr(nt,n_e+1)
   cnt = fltarr(nt,n_e+1)   
   e_min = fltarr(nt,32)
   e_max = fltarr(nt,32)
   old_e = fltarr(nt,n_e+1)
   cent_e = fltarr(nt,n_e +1)
   low_e = fltarr(nt,n_e+1)
   up_e = fltarr(nt,n_e+1)

   gf = replicate(gm_factor(i),nt,n_e+1)

  ;--- old stuff------
  ;    tt0 = [2.*xx(0)-xx(1), xx(0:nt-2)]
  ;    tt = xx -tt0
  ;-------------------
  
 
     
   e_max = vv
   e_min(*,1:31) = vv(*,0:30)
   e_min(*,0) = replicate(emin,nt)
   
      
   for j =0, n_e do begin
     dt(*,j) = tt
     elist = where(sum_bin eq j, eind)
     if eind eq 1 then begin
        cnt(*,j) = yy(*,elist) 
        denergy(*,j) = e_max(*,elist) - e_min(*,elist)
        old_e(*,j) = vv(*,elist)
        cent_e(*,j) = (e_max(*,elist) + e_min(*,elist))/2.0
        low_e(*,j) = e_min(*,elist)
        up_e(*,j) = e_max(*,elist)
     endif else begin
        cnt(*,j) = total(yy(*,elist),2,/nan)
        denergy(*,j) = e_max(*,elist(eind-1)) - e_min(*,elist(0))
        old_e(*,j) = average(vv(*,elist), 2, /nan)
        cent_e(*,j) = (e_max(*,elist(eind-1)) + e_min(*,elist(0)))/2.0
        low_e(*,j) = e_min(*,elist(0))
        up_e(*,j) = e_max(*,elist(eind-1))
     endelse   
     
   endfor

   ; correct for the energy loss

   if keyword_set(p) then begin
      denergy = denergy *1.06538 
      cent_e = cent_e*1.06538 + 2.3e3
      old_e = old_e*1.06538 + 2.3e3
      low_e = low_e*1.06538 + 2.3e3
      up_e = up_e*1.06538 + 2.3e3
   endif else begin
      cent_e = cent_e + 350. ;;; for electrons
      old_e = old_e + 350.
      low_e = low_e + 350.
      up_e = up_e + 350.
   endelse
   
   flux = cnt/(dt * gf * denergy)
   
   store_data, name(i)+'_f', data = {x:xx, y:flux, v: cent_e,v2:old_e}, dl=dl
   options,name(i)+'_f','spec',0
   options,name(i)+'_f','labels',strmid(strtrim(average(cent_e,1,/nan)/1.e3,2),0,4) + 'keV'
   options,name(i)+'_f','colors',[1,2,3,4,5,6]
   
   ee(*,i) = average(cent_e,1,/nan)
   e0(*,i) = average(low_e,1,/nan)
   e1(*,i) = average(up_e,1,/nan)

   if keyword_set(df) then begin
      if keyword_set(p) then begin
         
         vel = velocity_new(cent_e,/p) * 1.d0 ; m/s, velocity for proton
         gamma = cent_e/938.27d6 + 1.
         d_f = cnt/(dt * gf * 1.d-4 * denergy *1.602d-19 * vel * vel / (mass * gamma * gamma * gamma))
         store_data, name(i)+'_df', data = {x:xx, y:d_f, v: cent_e,v2:old_e}, dl=dl
         options,name(i)+'_df','spec',0
         options,name(i)+'_df','labels',strmid(strtrim(average(cent_e,1,/nan)/1.e3,2),0,4) + 'keV'
         options,name(i)+'_df','colors',[1,2,3,4,5,6]
      endif else begin

         vel = velocity_new(cent_e,/e) * 1.d0 ; m/s
         gamma = cent_e/0.511d6 + 1.
         d_f = cnt/(dt * gf * 1.d-4 * denergy *1.602d-19 * vel * vel / (mass * gamma * gamma * gamma))
         store_data, name(i)+'_df', data = {x:xx, y:d_f, v: cent_e,v2:old_e}, dl=dl
         options,name(i)+'_df','spec',0
         options,name(i)+'_df','labels',strmid(strtrim(average(cent_e,1,/nan)/1.e3,2),0,4) + 'keV'
         options,name(i)+'_df','colors',[1,2,3,4,5,6]
      endelse
      
   endif

   if keyword_set(err) then begin
      get_data, name(i),data=pp,dl=dl
      yy = pp.y
      xx2 =pp.x
      nt2 = dimen1(yy)
      cnt2 = fltarr(nt2,n_e+1)     ; counts for summed-over energy bins

      for j =0, n_e do begin
        elist = where(sum_bin eq j, eind)
        if eind eq 1 then cnt2(*,j) = yy(*,elist) else cnt2(*,j) = total(yy(*,elist),2,/nan)
      endfor
   
      if keyword_set(delta_t) then begin
         ni = -1
         nt = 0
         avg = fltarr(nt2,n_e+1)
         nnt0 = fltarr(nt2,n_e+1)       ; the number of time intervals for time-aveage
         dev22 = fltarr(nt2,n_e+1)     ; the square of deviation counts from the average
      
         while(ni lt nt2 - 1) do begin
           tt0 = xx2(ni+1)
           tlist = where((xx2 lt tt0 + delta_t - 0.5) and (xx2  ge tt0 - 1.e-6), tind)
           if tind gt 0L then begin
              avg = average(cnt2(tlist,*),1,/nan)
              nnt0(nt,*) = tind

              for m =0, n_e do dev22(nt,m) = total((cnt2(tlist,m)-avg(m))^2,/nan)
              
              nt =  nt+1
              ni = tlist(tind-1) 
           endif
         endwhile
      
         nnt = nnt0(0:nt-1,*)
         dev2 = dev22(0:nt-1,*)
      endif else  begin
         nnt = 1.0
         dev2 = replicate(0.0,nt2,n_e+1)
      
      endelse

      dcnt1 = sqrt(dev2)/(nnt*1.0)    ;  standard error of mean  ; /sqrt(count) for standard deviation
      dcnt2 = sqrt(dev2)/sqrt(nnt*1.0) ; standard deviation
      dcnt3 = sqrt(cnt) ; square root of total count during each time interval
      err1 = dcnt1/(dt * gf * denergy)
      err2 = dcnt2/(dt * gf * denergy)
      err3 = dcnt3/(dt * gf * denergy)

      store_data, name(i)+'_f', data = {x:xx, y:flux, v: cent_e,v2:old_e, err1: err1,err2:err2,err3:err3}, dl=dl
      options,name(i)+'_f','spec',0
      options,name(i)+'_f','labels',strmid(strtrim(average(cent_e,1,/nan)/1.e3,2),0,4) + 'keV'
      options,name(i)+'_f','colors',[1,2,3,4,5,6]
   endif  

    
 endfor
    
 return
         
end
