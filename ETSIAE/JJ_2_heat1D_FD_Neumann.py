"""
Heat equation with Neumann B.C.

u_t - [alpha(x) * u_x]_x = 0 in [0,1]
u(0,x) = f(x)
u_x(t,0) = DL; u_x(t,1) = DR 

author: jj.sanchez@upm.es
Date: 16th march 2023
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time

# - - - - - - - - - - - - - - - - - - - - - - - Numerical parameters
dt = 1e-3  # Time step
t = 0.0  # Initial time
tf = 0.5  # Final time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Mesh 1D
Nptos = 100  # Number of points (inluding boundaries)
x = np.linspace(0, 1, Nptos)  # Full interval [0,1]
h = 1.0/(Nptos-1)  # space step

# - - - - - - - - - - - - - - - - - - - - - - - Boundary Conditions
DL = 0  # Left B.C.
DR = 1  # Rigth B.C.

# - - - - - - - - - - - - - - - - - - - - - - - - Initial condition
def f_IC(x):
    return np.cos(2*np.pi*x)

# - - - - - - - - - - - - - - - - - - - - - - - Thermal diffusivity 
sigma = 0.025  #  standard deviation
mu = 0.5  # mean 
s0 = 0.2  # Base value
Amp = 0.0  # Amplitude 

def f_alpha(x, sigma=0.3, mu=0.5, s0=0.4, Amp=1.0):
    """
    Thermal diffusivity 
    Return alpha and its derivative
    """
    s_ = sigma**2
    
    a = s0+Amp*np.exp(-((x-mu)**2)/s_)
    ap = -Amp*(x-mu)*np.exp(-((x-mu)**2)/s_)/s_
    
    return a, ap


# ===========================================
#             Algorimth start
# ===========================================

alpha, alpha_p = f_alpha(x, sigma, mu, s0, Amp) # alpha, alpha'  

u = f_IC(x)  # Initial condition

a = 0.5*alpha*dt/h**2  
b = 0.25*alpha_p*dt/h


# - - - - - - - - - - - - - - - - - - - - - - - Coefficient matrix
A =  np.diag(b[:-1]-a[:-1], -1) + np.diag(1+2*a) - np.diag(a[1:]+b[1:], 1)
A[ 0, 1] += -a[0]  # First row
A[-1,-2] += -a[-1] # Last row


# - - - - - - - - - - - - - - - - - - - - - - - - - Temporal bucle
start = time.time()
sol = u.copy()
tt = [t]
while t < tf:
    t += dt    
    u_old = u.copy()
    
    # Rigth hand side
    B = (1-2*a)*u_old
    B[1:-1] += (a[1:-1]-b[1:-1])*u_old[:-2] + (a[1:-1]+b[1:-1])*u_old[2:]
    B[ 0] += -4*h*DL*(a[0]-b[0]) + 2*a[0]*u_old[1]
    B[-1] +=  4*h*DR*(a[-1]+b[-1]) + 2*a[-1]*u_old[-2]

    u = la.solve(A, B)
    
    sol = np.c_[sol, u]
    tt = np.c_[tt, t]
# -- end Temporal bucle --

end = time.time()
#print(end - start)

# - - - - - - - - - - - - - - - - - - - - - - - Save results

FICHERO = f'heat1D_Neumann_alpha_mu{mu}_sigma{sigma}_s{s0}_A{Amp}.npz'
np.savez(FICHERO, t=tt, x=x, u=sol, 
                  mu=mu, sigma=sigma, s0=s0, Amp=Amp, alpha=alpha)


# - - - - - - - - - - - - - - - - - - - - - - -  Plotting 
# plt.figure()
# plt.plot(x, alpha)
# plt.xlabel("$x$")
# plt.ylabel("$alpha(x)$")
# plt.grid()
# plt.title('1D Heat equation - Thermal Diffusitity')
# plt.show()


# m = int(np.size(tt)/2)
# plt.figure()
# plt.plot(x, sol[:, 0])
# #plt.plot(s, sol[:, m])
# plt.plot(x, sol[:, -1])
# plt.xlabel("$x$")
# plt.ylabel("$u(t,x)$")
# plt.grid()
# plt.title('1D Heat equation - B.C. Neumann')
# plt.show()

