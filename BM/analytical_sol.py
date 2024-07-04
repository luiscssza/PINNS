###########################################################################################################################################
### Import all what you need:
import os

import torch

import torchinfo 
from torchsummary import summary # https://pypi.org/project/torch-summary/

import matplotlib.pyplot as plt
import numpy as np

import time

###########################################################################################################################################



# Damped Harmonic Oscillator:

def exact_solution(d, w0, t):
    "Defines the analytical solution to the under-damped harmonic oscillator problem above."
    assert d < w0             
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*t)
    exp = torch.exp(-d*t)
    u = exp*2*A*cos
    return u