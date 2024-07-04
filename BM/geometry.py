###########################################################################################################################################
### Import all what you need:
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchinfo 
from torchsummary import summary # https://pypi.org/project/torch-summary/

import matplotlib.pyplot as plt
import numpy as np

import random
import time

###########################################################################################################################################


def point_generation(init_cond_mu_interval=(1,10), physic_t_interval=(1e-2,1), point_resolution_range=(40,40)):
    """
    Function to generate the needed points for the initial conditions, training and test
        Args:
            - t_interval (tuple of floats):
            - mu_interval (tuple of floats):
            - point_resolution_range (tuple of floats): definition of an interval 
        Returns:
        
    """
    point_resolution = random.randint(point_resolution_range[0], point_resolution_range[1])

    #### Generation of t and mu initial points (Initial condition 1) 
    ic1_t_mu = torch.stack([torch.zeros(point_resolution).requires_grad_(True), torch.linspace(init_cond_mu_interval[0], init_cond_mu_interval[1], point_resolution).requires_grad_(True)],-1)
    ic1_scope = torch.ones_like(ic1_t_mu[:,0:1]).requires_grad_(True)

    #### Generation of t and mu initial points (Initial condition 2) 
    ic2_t_mu = torch.stack([torch.zeros(point_resolution).requires_grad_(True), torch.linspace(init_cond_mu_interval[0], init_cond_mu_interval[1], point_resolution).requires_grad_(True)], -1)
    ic2_scope = torch.zeros_like(ic2_t_mu[:,0:1]).requires_grad_(True)
   
    ### Generate domain physic loss sample points:
    physic_in_t_mu = [torch.linspace(physic_t_interval[0],physic_t_interval[1], point_resolution).requires_grad_(True), torch.linspace(init_cond_mu_interval[0], init_cond_mu_interval[1], point_resolution).requires_grad_(True)] # Mejora: se puede utilizar torch.rand
    physic_domain_t_mu = torch.stack(torch.meshgrid(*physic_in_t_mu, indexing='ij'), -1).view(-1, 2).requires_grad_(True)

    return point_resolution, ic1_t_mu, ic1_scope, ic2_t_mu, ic2_scope, physic_in_t_mu, physic_domain_t_mu



def adim_point_generation(physic_t_interval=(1e-2,1), point_resolution_range_t = 40):
    """
    Function to generate the needed points for the initial conditions, training and test
        Args:
            - t_interval (tuple of floats):
            - mu_adim_interval (tuple of floats):
            - point_resolution_range (tuple of floats): definition of an interval 
        Returns:
        
    """
    point_resolution = random.randint(point_resolution_range[0], point_resolution_range[1])

    #### Generation of t and mu_adim initial points (Initial condition 1) 
    ic_stacked = torch.stack(
        [
            torch.zeros(point_resolution).requires_grad_(True), 
            torch.linspace(init_cond_mu_interval[0], init_cond_mu_interval[1], point_resolution).requires_grad_(True)
        ],
        -1
    )
    ic1_t_mu = ic_stacked.clone()    
    ic1_scope = torch.ones_like(ic1_t_mu[:,0:1]).requires_grad_(True)

    #### Generation of t and mu initial points (Initial condition 2) 
    ic2_t_mu = ic_stacked.clone()
    #ic2_t_mu = torch.stack([torch.zeros(point_resolution).requires_grad_(True), torch.linspace(init_cond_mu_interval[0], init_cond_mu_interval[1], point_resolution).requires_grad_(True)], -1)
    ic2_scope = torch.zeros_like(ic2_t_mu[:,0:1]).requires_grad_(True)
   
    ### Generate domain physic loss sample points:
    physic_in_t_mu = [torch.linspace(physic_t_interval[0],physic_t_interval[1], point_resolution).requires_grad_(True), torch.linspace(init_cond_mu_interval[0], init_cond_mu_interval[1], point_resolution).requires_grad_(True)] # Mejora: se puede utilizar torch.rand
    physic_domain_t_mu = torch.stack(torch.meshgrid(*physic_in_t_mu, indexing='ij'), -1).view(-1, 2).requires_grad_(True)

    return point_resolution, ic1_t_mu, ic1_scope, ic2_t_mu, ic2_scope, physic_in_t_mu, physic_domain_t_mu