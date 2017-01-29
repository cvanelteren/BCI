from pylab import *
# from removeThis2 import press
import removeThis2
print(removeThis2.press)
removeThis2.init()
removeThis2.test.append('hello')
start = removeThis2.start
print(start)
#
#
fig, ax = subplots()
fig.canvas.mpl_connect('key_press_event', removeThis2.press)
fig.canvas.draw()
print('here')

while start:
    start = removeThis2.getStart()
    print(start)
    if start == 0:
        break
    # print(start)
    pause(.1)
show(0)
