from numpy import *
from numpy.linalg import norm

def f(x, y):
    return x+y-100.0*((x**2+y**2-1)**2)
    
def dfdx(x, y):
    return 1-400.0*(x)*(x**2+y**2-1)

def dfdy(x, y):
    return 1-400.0*(y)*(x**2+y**2-1)

def gradf(x, y):
    return array([dfdx(x, y), dfdy(x, y)])  

def grad_descent2(f, gradf, init_t, alpha):
    EPS = 1e-5
    prev_t = init_t-10*EPS
    t = init_t.copy()
    
    max_iter = 1000
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*gradf(t[0], t[1])
        print( t, f(t[0], t[1]), gradf(t[0], t[1]))
        iter += 1
    return t
    
grad_descent2(f, gradf, array([1.0, 1.0]), 0.005)
