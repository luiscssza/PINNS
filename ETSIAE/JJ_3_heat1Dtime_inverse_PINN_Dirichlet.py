"""
Inverse Problem
Heat equation with Dirichlet B.C.

u_t - [alpha(x) * u_x]_x = 0 in [0,1]

I.C. : u(0,x) = sin(0.5*pi*x)
B.C. : u(t,0) = 0; u(t,1) = 1

alpha(x) = a0+Amp*exp(-((x-mu)**2)/sigma**2)

Author: jj.sanchez@upm.es
Date: 17th march 2023
"""
import deepxde as dde
import numpy as np

from deepxde.backend import tf


def load_training_data(N_obs=100):
                  
    dataFile = "./heat1D_Dirichlet_alpha_mu0.7_sigma0.025_s0.75_A1.0.npz"

    data = np.load(dataFile)

    t = data['t']
    x = data['x']
    u = data['u']

    tf = t[-2]  # Final time = 1.0

    mu = data['mu']
    sigma = data['sigma']
    alpha0 = data['s0']
    Amp = data['Amp']
    
    idx = np.random.choice(x.shape[0], N_obs, replace=False)

    observe_x = np.vstack((x[idx], np.full((N_obs), tf))).T  
    
    observe_u = dde.icbc.PointSetBC(observe_x, u[idx,-1], component=0)
    
    return [observe_x, observe_u, mu, sigma, alpha0, Amp]


# Initial Condition
def funcIC(x):
    return tf.sin(0.5*np.pi * x[:, 0:1]) 


def fun_bc(x):
    """
    u(x=0) = 0
    u(x=1) = 1
    """
    return x[:, 0:1]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - Get the training data
[observe_x, observe_u, mu, sigma, a0, Amp] = load_training_data(500)


# - - - - - - - - - - - - - - - - - - - - - - - - Define Spatio-temporal domain
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


Amp = dde.Variable(0.5)  # True value Amp = 1.0
mu = dde.Variable(0.5)  # True value mu = 0.7

bc = dde.icbc.DirichletBC(geomtime, fun_bc, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, funcIC, lambda _, on_initial: on_initial)


def Heat_Variable_Thermal_Diffusivity(x, u):
     
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
 
    return (
        du_t +
        ((Amp*(x[:, 0:1]-mu)/sigma**2) 
        * tf.exp(-0.5*((x[:, 0:1]-mu)/sigma)**2)) * du_x 
        - (a0 + Amp * tf.exp(-0.5*((x[:, 0:1]-mu)/sigma)**2)) * du_xx
    )
    

# - - - - - - - - - - - - - - - - - - - - - - - -  Neural Network setup
# Training datasets and Loss
data = dde.data.TimePDE(
    geomtime,
    Heat_Variable_Thermal_Diffusivity,
    [bc, ic, observe_u],
    num_domain=5000,
    num_boundary=100,
    num_initial=100,
    anchors=observe_x,
)

layer_size = [2] + [50] * 6 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

# callbacks for storing results
fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue([Amp, mu], 
                                       period=1000, 
                                       filename=fnamevar)

# Compile, train and save model
model.compile("adam", lr=1e-3, external_trainable_variables=[Amp, mu])
loss_history, train_state = model.train(
    iterations=10000, callbacks=[variable], 
    display_every=1000, 
    disregard_previous_best=True
)

# dde.saveplot(loss_history, train_state, issave=True, isplot=True)
# model.compile("adam", lr=1e-4, external_trainable_variables=[Amp, mu])
# loss_history, train_state = model.train(
#     epochs=10000, callbacks=[variable], 
#     display_every=1000, 
#     disregard_previous_best=True
# )

dde.saveplot(loss_history, train_state, issave=True, isplot=True)
model.save(save_path = "./Heat_inverse_model/model")
#f = model.predict(ob_xt, operator=Heat_Variable_Thermal_Diffusivity)
#print("Mean residual:", np.mean(np.absolute(f)))



