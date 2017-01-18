import multiprocessing
import subprocess
from experiment import experiment as ex
import os
def worker(file, linux = 1):
    # subprocess.call(["python", file])

    if linux:
        print(file[-2:])
        if file[-2:] == 'py':
            tmp = 'python '
        else : tmp = ''

        # assert 0
        subprocess.call(['gnome-terminal', '-x', tmp + file ])

if __name__ =='__main__':
    # fil   es = ['eeg_connect','buffer_connect','experiment' ]
    files = ['SignalProcessing.py', os.path.realpath('../../debug_quickstart.sh')]
    print(files);
    # assert 0

    # for i in files:
    #     print(i)
    #     p = multiprocessing.Process(target = worker(i + '.py'))
    #     p.start()
    workers = []
    for i in files:
        tmp = multiprocessing.Process(target = worker, args = (i,))
        # tmp.start()
        workers.append(tmp)
    for i in workers:
        i.start()

    ex('calibration')
    ex('test')

    #     i.join()
    # for i in workers:
    #     i.join()

    # p = multiprocessing.Process(target = worker, args = (files[0] +'.py',))
