"""
Heat equation with Dirichlet B.C.

u_t - [alpha(x) * u_x]_x = 0 in [0,1]
u(0,x) = f(x)
u(t,0) = uL; u(t,1) = uR 

author: jj.sanchez@upm.es
Date: 16th march 2023
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import time

# - - - - - - - - - - - - - - - - - - - - - - - Numerical parameters
dt = 1e-4  # Time step
t = 0.0  # Initial time
tf = 1.0  # Final time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Mesh 1D
Nptos = 1000  # Number of points (inluding boundaries)
s = np.linspace(0, 1, Nptos)  # Full interval [0,L]
x = s[1:Nptos-1].copy()  # Inner points
h = 1.0/(Nptos-1)  # Should be equal to s[1] = s[1]-s[0]
n = Nptos-2  # Number of interior points. Size of the problem.


# - - - - - - - - - - - - - - - - - - - - - - - Boundary Conditions
uL = 0.0  # Left B.C.
uR = 1.0  # Rigth B.C.

# - - - - - - - - - - - - - - - - - - - - - - - - Initial condition
def f_IC(x):
    return np.sin(0.5*np.pi*x)

# - - - - - - - - - - - - - - - - - - - - - - - Thermal diffusivity 
sigma = 0.025  #  standard deviation
mu = 0.7  # mean 
s0 = 0.75  # Base value
Amp = 1.0  # Amplitude 

def f_alpha(x, sigma, mu, s0, Amp):
    """
    Thermal diffusivity 
    Return alpha and its derivative
    """
    a = s0+Amp*np.exp(-0.5*((x-mu)/sigma)**2)
    ap = -Amp*((x-mu)/sigma**2)*np.exp(-0.5*((x-mu)/sigma)**2)
    
    return a, ap


# ===========================================
#             Algorimth start
# ===========================================

alpha, alpha_p = f_alpha(x, sigma, mu, s0, Amp) # alpha, alpha'  


u = f_IC(x)  # Initial condition
sol = u.copy()

a = 0.5*alpha*dt/h**2  
b = 0.25*alpha_p*dt/h

# - - - - - - - - - - - - - - - - - - - - - - - Coefficient matrix
A =  np.diag(b[:-1]-a[:-1], -1) + np.diag(1+2*a) - np.diag(a[1:]+b[1:], 1)

# - - - - - - - - - - - - - - - - - - - - - - - - - Temporal bucle
start = time.time()
tt = np.array([t])
nite = 0
while t < tf:
    t += dt    
    u_old = u.copy()
    # Rigth hand side
    B = (1-2*a)*u_old
    B[1:-1] += (a[1:-1]-b[1:-1])*u_old[:-2] + (a[1:-1]+b[1:-1])*u_old[2:]
    B[ 0] += 2*uL*(a[0]-b[0]) + (a[0]+b[0])*u_old[1]
    B[-1] += 2*uR*(a[-1]+b[-1]) + (a[-1]-b[-1])*u_old[-2]

    u = la.solve(A, B)
    
    sol = np.c_[sol, u]
    tt = np.append(tt, t)

# -- end Temporal bucle --
end = time.time()

# - - - - - - - - - - - - - - - - - - - - - - - Save results
# Adding Boundary points
# sol = [each column is the solution u(x) in a particular instant]
#       [The extrems of the interval are not includes (Dirichlet)]
sol = np.r_[[uL*np.ones_like(tt)], sol]  # Insert uL as first row
sol = np.r_[sol, [uR*np.ones_like(tt)]]  # Insert uR as last row

alpha, alpha_p = f_alpha(s, sigma, mu, s0, Amp) 


FICHERO = f'heat1D_Dirichlet_alpha_mu{mu}_sigma{sigma}_s{s0}_A{Amp}.npz'

np.savez(FICHERO, t=tt, x=s, u=sol, 
                  mu=mu, sigma=sigma, s0=s0, Amp=Amp, alpha=alpha)


print(f"Iterations: {np.size(tt)-1} | System size: {A.shape}")
print(f"Temporal bucle elapsed time: {(end - start):.3g} s")
print(f"Output file: {FICHERO}")

# - - - - - - - - - - - - - - - - - - - - - - -  Plotting 
# plt.figure()
# plt.plot(s, alpha)
# plt.plot(s, alpha_p)
# plt.xlabel("$x$")
# plt.ylabel("$alpha(x)$")
# plt.grid()
# plt.title('1D Heat equation - Thermal Diffusitity')
# plt.show()


# m = int(np.size(tt)/2)
# plt.figure()
# plt.plot(s, sol[:, 0])
# plt.plot(s, sol[:, m], linestyle='dashed')
# plt.plot(s, sol[:, -1])
# plt.xlabel("$x$")
# plt.ylabel("$u(t,x)$")
# plt.grid()
# plt.title('1D Heat equation - B.C. Dirichlet')
# plt.show()


