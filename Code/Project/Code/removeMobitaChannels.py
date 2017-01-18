from h5py   import File
import os
from shutil import move
'''
This is a temporary script to remove the number of channels from the
data we already have.
We had 37 output by the mobita, but we only need the first channels as dictated
by the cap file. This is fixed within the signal processing script
'''
def removeChannels(fileDir):
    # remove the lab
    newStorage  = fileDir
    a, b        = newStorage.split('_LAB')
    newStorage  = a + b
    with File(fileDir, 'r') as f:
        for i in f: print(i)
        rawData     = f['rawData'].value
        procData    = f['processedData'].value
        caps        = f['cap'].value

    # only keep the first nChans
    nChans      = caps.shape[0]
    newData     = [tmp[:, :, :nChans] for tmp in [rawData, procData]]
    file        = fileDir.split('/')[-1]
    # move file to backup
    move(fileDir, os.path.realpath('../Data/Backup/' + file))
    with File(newStorage,'w') as f:
        labels = ['rawData', 'processedData','cap']
        data   = [newData[0], newData[1], caps]
        for label, datai in zip(labels, data):
            f.create_dataset(label, data = datai)





removeChannels('../Data/calibration_subject_4_LAB.hdf5')

# sanity check
# with File('../Data/calibration_subject_5.hdf5') as f :
#     for i in f: print(i)
#     data = f['processedData'].value
#     print(data.shape)
