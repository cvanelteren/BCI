from pylab import *
from numpy import *
from sklearn.linear_model import LinearRegression as LR


x = random.randn(100,2)
t = sign(x.dot(array([1,0], ndmin = 2).T))
t = array([['a']] * x.shape[0])
print(t.shape)
print(np.unique(t))

c = LR()
c.fit(x, t)
print(c.score(x,t))
n = array([[3,0.9]])
d = c.predict(n)
print(d)
fig, ax = subplots()
ax.scatter(*x.T, c = t, cmap = 'seismic')
ax.scatter(*n, c = d)
show()
