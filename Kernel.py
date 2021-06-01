import numpy as np

def Linear_Kernel(x, y):
    """
    """
    return np.dot(x,y.T)

def Gaussian_Kernel(x,y,sigma=2**0):
    """
    """
    numxPoints = np.shape(x)[0];
    numyPoints = np.shape(y)[0];
    XminusYSquared =np.sum(x*x,axis=1).reshape([numxPoints,1])-2*np.dot(x,y.T)+np.sum(y*y,axis=1).reshape([1,numyPoints]); 
    return np.exp(-XminusYSquared/(2*sigma**2))

def Hyperbolic_Tangent_Kernel(x, y, kappa=1, theta=1):
    """
    """
    return np.tanh(kappa*np.dot(x,y.T)+theta);

def Polynomial_Kernel(x, y, power=1, constant=1):
    """
    """
    return np.power(np.dot(x,y.T)+constant, power);