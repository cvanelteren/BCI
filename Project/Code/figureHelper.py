from pylab import pause
def init():
    '''
    initialize global variable space
    '''

    global test, start
    test = []
    start = 1

def press(event):
  global start
  print(event.key)
  if event.key == ' ':
      start = 0
      setStart(start)

  if event.key == 'a':
      start = 0
  print('>', start)
  # return start

def setStart(switch):
    global start
    start = switch

def getStart():
    global start
    return start

def waitForSpacePress():
    '''
    Waits until space bar is pressed
    '''
    global start
    start = 0
    alpha = True
    while alpha:
        if start == 1:
            alpha = False

        pause(.1)
