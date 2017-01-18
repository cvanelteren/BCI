import numpy as np

def genPrediction(data, model, events):
    # average over time
    mdata = np.mean(data, 1)
    # compute the probability of each event with respect to
    # the binary classes
    pred = model.predict_proba(mdata)
    # obtain the row, i.e. event, of the max probability
    # with respect to the target class
    # target class has index 0
    idx = np.argmax(pred, 0)[0] # returns a tuple (oddly enough); first index is row which is what we want
    pred = events[idx].value
    return pred
