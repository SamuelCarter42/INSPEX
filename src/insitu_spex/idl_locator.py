

import os
import subprocess
import sys

import tkinter as tk


import tkinter.filedialog


#this function searches for and outputs the end user's idl install
def idl_locator():
    #check for previous selection of install location
    #if none, set as default windows location
    this_dir=os.listdir(os.getcwd())
    if "idl_loc.txt" not in this_dir:        
        idl_install_dir=fr"C:\Program Files\NV5\IDL90"    
        idl_install_exe=os.path.join(idl_install_dir, 'bin', 'bin.x86_64', 'idl.exe')
        with open("idl_loc.txt", "w") as f:
            f.write(idl_install_exe)
    
    
    #open stored IDL install location
    with open("idl_loc.txt", "r") as f:
        idl_install_exe=f.read()
    
    #create window
    finder = tk.Tk()
    finder.title("Select IDL install")
    
    
    #display current IDL filepath
    path_var = tk.StringVar(value=idl_install_exe)
    text=tk.Label(finder, text="Currently selected IDL install:", wraplength=400, anchor="w", justify="left")
    text.pack()
    displayloc=tk.Label(finder, textvariable=path_var, wraplength=400, anchor="w", justify="left")
    displayloc.pack()
    
    #create continue button
    def continue_hndl():
        finder.destroy()
    
    cnt_button = tk.Button(master=finder, text="Continue", command=continue_hndl)
    cnt_button.pack(side=tk.BOTTOM)
    
    #create choose button
    def choose_hndl():
        nonlocal idl_install_exe
        file_obj=tk.filedialog.askopenfile()
        if file_obj is not None:
            idl_install_exe = file_obj.name
            with open("idl_loc.txt", "w") as f:
                f.write(idl_install_exe)
            path_var.set(idl_install_exe)

    chs_button = tk.Button(master=finder, text="Select IDL install", command=choose_hndl)
    chs_button.pack(side=tk.BOTTOM)


            
    

    
    
    finder.mainloop()
    return idl_install_exe 
