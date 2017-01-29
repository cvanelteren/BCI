from __future__ import print_function
import os

def checkOverwrite(dir, conditionType, subjectNumber,  fileType = '.hdf5'):
    '''
    Checks whether a file is in the directory
    If it is in the directory it will ask for changing the subject number
    return the storage filepath
    '''
    # keep checking until file is not in the folder
    i = 0
    while True:
        if os.path.isfile(dir + conditionType + str(subjectNumber) + fileType):
            subjectNumber = i
            i += 1
        # expecting not more than 20 subjects; ask for input
            # if subjectNumber > 20:
            #     subjectNumber = raw_input('Please enter a new subject number ')
        else:
            file = dir  + conditionType + str(subjectNumber) + fileType
            break
    print('Storing data in: \n\t ', file)
    return file


def enterSubjectNumber(subjectNumber):
    '''
    This function is used for loading data sets
    It will check whether the requiested calibration file is present in the folder
    if not it will ask for another input
    '''

    import os
    while True:
        subjectNumber = raw_input('Please enter a subject number')
        file = '../Data/calibration_subject_{0}.hdf'.format(subjectNumber)
        yes = os.path.isfile(file)
        print(yes)
        if yes:
            break
        else:
            print('Error! Subject number does not exist')
    return file
