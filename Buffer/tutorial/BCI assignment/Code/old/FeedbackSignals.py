
filterBand = [0, 60]
tic = time.time()
# linear detrend, filter, average across epochs
data = preproc.stdPreproc(data, filterBand, hdr)
# average over time
mdata = np.mean(data, 1)
# compute the probability of each event with respect to
# the binary classes
pred = model.predict_proba(mdata)
# obtain the row, i.e. event, of the max probability
# with respect to the target class
idx = np.argmax(pred, int(labels[0]))[0]
pred = events[idx].value
print('prediction', pred)

