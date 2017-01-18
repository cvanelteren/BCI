from shutil import move
from h5py   import File

'''
This is a temporary script to remove the number of channels from the
data we already have.
We had 37 output by the mobita, but we only need the first channels as dictated
by the cap file
'''
def removeChannels(fileDir):
    # remove the lab
    newStorage  = fileDir
    a, b        = newStorage.split('_LAB')
    newStorage  = a + b
    with File(fileDir) as f:
        for i in f: print(i)
        rawData     = f['rawData'].value
        procData    = f['processedData'].value
        caps        = f['cap'].value

    # only keep the first nChans
    nChans      = caps.shape[0]
    newData     = [tmp[:, :, :nChans] for tmp in [rawData, procData]]
    file        = fileDir.split('/')[-1]
    # print(file)
    # shutil.move(fileDir, 'Backups/' + file)
    with File(newStorage) as f:
        labels = ['rawData', 'processedData','cap']
        data   = [newData[0], newData[1], caps]
        for label, datai in zip(data, labels):
            print(label, datai)




removeChannels('../Data/calibration_subject_5_LAB.hdf5')
