import subprocess, os
from subprocess import CREATE_NEW_CONSOLE
linux = 1
if linux:
    subprocess.call(os.path.realpath('../../debug_quickstart.sh'),\
     creationflags = CREATE_NEW_CONSOLE )
else:
    subprocess.call(os.path.realpath('../../../debug_quickstart.bat'),\
creationflags=subprocess.CREATE_NEW_CONSOLE)


# subprocess.call(os.path.realpath('../../../debug_quickstart.bat'))
# pause(.1)
