
    #need to include method to locate idl install directory in other system
import idl_locator#for finding the user's IDL install 
import subprocess

def run_idl_command(idl_command):

    idl_executable =idl_locator.idl_locator() 
    # Form the command to execute IDL
    with subprocess.Popen(idl_executable, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        stdout, stderr = process.communicate(idl_command)
    print(stdout)
    if stderr:
        print("Errors:", stderr)


        

        
        

def run_idl_script(idl_commands):
    idl_script = "\n".join(idl_commands)
    with subprocess.Popen(['idl'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        stdout, stderr = process.communicate(idl_script)
        print(stdout)
        if stderr:
            print("Errors:", stderr)
