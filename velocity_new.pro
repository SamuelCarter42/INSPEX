function  velocity_new,energy,e=e,p=p,i=i

  ; To caluculate the velocity for electrons, protons and ions
  ;
  ;CREATED BY:  Linghua Wang
  ;LAST MODIFIED: 05/15/06

  ; energy in unit of eV

  if keyword_set(e) then v=(sqrt(1.-(0.511d6/(0.511d6+energy))^2)*299792458.) ; for electron

  if keyword_set(p) then v=(sqrt(1.-(931.49d6*1.007276/(931.49d6*1.007276+energy))^2)*299792458.) ; for proton

  if keyword_set(i) then v=(sqrt(1.-(931.49d6/(931.49d6+energy))^2)*299792458.) ; for ion per amu

  return,v ; m/s

end