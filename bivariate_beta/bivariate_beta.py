import numpy as np
import scipy as sp
from scipy.special import loggamma
from scipy.special import digamma
from scipy.integrate import quad

# This library contains methods for estimation of sampling from a 5-parameter bivariate beta distribution, as described in Arnold & Ng, 2011
# - Flexible bivariate beta distributions, https://doi.org/10.1016/j.jmva.2011.04.001

def fVWU3U4U5(v, w, u3, u4, u5, alpha1, alpha2, alpha3, alpha4, alpha5):
    # This function implements the density just before eq. (2)
    
    if (u4/w - u5 < u3 < (u4 + u5)*v and u4 > 0 and u5 > 0 and v > 0 and w > 0):
        
        f1 = (u3 + u5)*(u4 + u5)/np.exp(loggamma(alpha1) + loggamma(alpha2) + loggamma(alpha3) + loggamma(alpha4) + loggamma(alpha5))
        f2 = (v*(u4 + u5) - u3)**(alpha1 - 1)
        f3 = (w*(u3 + u5) - u4)**(alpha2 - 1)
        f4 = u3**(alpha3 - 1)*u4**(alpha4 - 1)*u5**(alpha5 - 1)
        f5 = np.exp(-(u3*w + u4*v + u5*(v + w + 1)))
        
        return f1*f2*f3*f4*f5
       
    else:
        
        return 0

def integrand3(u3, u4, u5, v, w, alpha1, alpha2, alpha3, alpha4, alpha5):
    # This function implements the innermost integrand (in variable u3) in eq. (2)
        
    n = len(np.array([u3]))
    
    if (n == 1):
        
        return fVWU3U4U5(v, w, u3, u4, u5, alpha1, alpha2, alpha3, alpha4, alpha5)
    
    else:
        fout = np.zeros(n)
            
        for i in range(n):
            fout[i] = fVWU3U4U5(v, w, u3[i], u4, u5, alpha1, alpha2, alpha3, alpha4, alpha5)
                
        return fout

def integrand4(u4, u5, v, w, alpha1, alpha2, alpha3, alpha4, alpha5):
    # This function implements the middle integrand (in variable u4) in eq. (2)
    
    n = len(np.array([u4]))
    
    if (n == 1):
        return quad(integrand3, u4/w-u5, (u4+u5)*w, args=(u4, u5, v, w, alpha1, alpha2, alpha3, alpha4, alpha5))[0]
    else:
        fout = np.zeros(n)     
            
        for i in range(n):    
            fout[i] = quad(integrand3, u4[i]/w-u5, (u4[i]+u5)*w, args=(u4[i], u5, v, w, alpha1, alpha2, alpha3, alpha4, alpha5))[0]
                
        return fout

def integrand5(u5, v, w, alpha1, alpha2, alpha3, alpha4, alpha5):
    # This function implements the outermost integrand (in variable u5) in eq. (2)
    
    n = len(np.array([u5]))
    
    if (n == 1):
        return quad(integrand4, 0, np.inf, args=(u5, v, w, alpha1, alpha2, alpha3, alpha4, alpha5))[0]
    else:
        fout = np.zeros(n)    
            
        for i in range(n):
            fout[i] = quad(integrand4, 0, np.inf, args=(u5[i], v, w, alpha1, alpha2, alpha3, alpha4, alpha5))[0]
                
        return fout

def fVW(v, w, alpha1, alpha2, alpha3, alpha4, alpha5):
    # This function implements the density in eq. (2)
    
    if (v > 0 and w > 0):
    
        return quad(integrand5, 0, np.inf, args=(v, w, alpha1, alpha2, alpha3, alpha4, alpha5))[0]
    
    else:
        return 0
    
def fXY(x, y, alpha1, alpha2, alpha3, alpha4, alpha5):
    # This function implements the bivariate beta density in eq. (3)
    
    if (0 < x < 1 and 0 < y < 1):
        
        f1 = fVW(x/(1 - x), y/(1 - y), alpha1, alpha2, alpha3, alpha4, alpha5)
        f2 = 1/(1 - x)**2/(1 - y)**2
        
        return f1*f2
    
    else:
        
        return 0
    
def moment_estimator(xdata, ydata):
    # This function implements the method of moments estimator based on sample means and variances, as described in Sec. 3.2
    
    n = len(xdata)
    
    mx = np.mean(xdata)
    my = np.mean(ydata)
    
    varx = np.var(xdata)
    vary = np.var(ydata)
    
    a = mx*(mx*(1 - mx)/varx - 1)
    b = (1 - mx)*(mx*(1 - mx)/varx - 1)
    
    c = my*(my*(1 - my)/vary - 1)
    d = (1 - my)*(my*(1 - my)/vary - 1)
    
    B = b*c + a*c + a*d - b - d
    C = (a - 1)*(c - 1)*b*d - np.sum(np.exp(np.log(1 - xdata) + np.log(1 - ydata) - np.log(xdata) - np.log(ydata)))*a*c*(a - 1)*(c - 1)/n
    
    alpha5 = np.maximum(0, (-B + np.sqrt(B**2 - 4*C))/2)
    alpha4 = np.maximum(0, b - alpha5)
    alpha3 = np.maximum(0, d - alpha5)
    alpha2 = np.maximum(0, c - alpha4)
    alpha1 = np.maximum(0, a - alpha3)
    
    return alpha1, alpha2, alpha3, alpha4, alpha5

def modified_ml_estimator(xdata, ydata):
    # This function implements the modified maximum likelihood estimator, as described in Sec. 3.1
    # The method of moments estimator based on sample means and variances is used as an initial guess
    
    n = len(xdata)
    
    mx = np.mean(xdata)
    my = np.mean(ydata)
    
    varx = np.var(xdata)
    vary = np.var(ydata)
    
    a0 = mx*(mx*(1 - mx)/varx - 1)
    b0 = (1 - mx)*(mx*(1 - mx)/varx - 1)
    
    c0 = my*(my*(1 - my)/vary - 1)
    d0 = (1 - my)*(my*(1 - my)/vary - 1)
    
    def fcnx(params):
        
        a = params[0]
        b = params[1]
        
        f1 = digamma(a) - digamma(a + b) - np.sum(np.log(xdata))/n
        f2 = digamma(b) - digamma(a + b) - np.sum(np.log(1 - xdata))/n
        
        return np.array([f1, f2])
    
    sol1 = sp.optimize.root(fcnx, np.array([a0, b0]))
    a = sol1.x[0]
    b = sol1.x[1]
    
    def fcny(params):
        
        c = params[0]
        d = params[1]
        
        f1 = digamma(c) - digamma(c + d) - np.sum(np.log(ydata))/n
        f2 = digamma(d) - digamma(c + d) - np.sum(np.log(1 - ydata))/n
        
        return np.array([f1, f2])
    
    sol2 = sp.optimize.root(fcny, np.array([c0, d0]))
    c = sol2.x[0]
    d = sol2.x[1]
    
    B = b*c + a*c + a*d - b - d
    C = (a - 1)*(c - 1)*b*d - np.sum(np.exp(np.log(1 - xdata) + np.log(1 - ydata) - np.log(xdata) - np.log(ydata)))*a*c*(a - 1)*(c - 1)/n
    
    alpha5 = np.maximum(0, (-B + np.sqrt(B**2 - 4*C))/2)
    alpha4 = np.maximum(0, b - alpha5)
    alpha3 = np.maximum(0, d - alpha5)
    alpha2 = np.maximum(0, c - alpha4)
    alpha1 = np.maximum(0, a - alpha3)
    
    return alpha1, alpha2, alpha3, alpha4, alpha5

def sample(n, alpha1, alpha2, alpha3, alpha4, alpha5):
    # This function draws a sample of size from the bivariate beta distribution, using the characterisation in eq. (1)
    
    
    xsample = np.zeros(n)
    ysample = np.zeros(n)
    
    for i in range(n):
        
        U1 = np.random.gamma(alpha1)
        U2 = np.random.gamma(alpha2)
        U3 = np.random.gamma(alpha3)
        U4 = np.random.gamma(alpha4)
        U5 = np.random.gamma(alpha5)
    
        xsample[i] = (U1 + U3)/(U1 + U3 + U4 + U5)
        ysample[i] = (U2 + U4)/(U2 + U3 + U4 + U5)
        
    return xsample, ysample