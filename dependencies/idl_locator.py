

import os
import subprocess
import sys


#this function searches for and outputs the end user's idl install
def idl_locator():
    idl_install_dir=fr"C:\Program Files\NV5\IDL90"
    
    idl_install_exe=os.path.join(idl_install_dir, 'bin', 'bin.x86_64', 'idl.exe')
    
    
    
    
    return idl_install_exe
