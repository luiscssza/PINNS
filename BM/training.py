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


class TrainModel:
    def __init__(
        self, 
        model,
        original_input_size,
        original_hidden_layers,
        original_output_size,
        t_interval,
        ic1_t_mu, 
        ic1_scope,
        ic2_t_mu,
        ic2_scope, 
        physic_in_t_mu, 
        physic_domain_t_mu,
        point_resolution,
        test_ic1_t_mu, 
        test_ic1_scope, 
        test_ic2_t_mu, 
        test_ic2_scope, 
        test_physic_in_t_mu, 
        test_physic_domain_t_mu,
        test_point_resolution,
        u_exact, 
        test_mu = 5, 
        mass = 1, 
        w0 = 20, 
        learning_rate = 0.01, 
        num_epochs= 20000, 
        lambda_ic_dudt= 1e-1, 
        lambda_diff_equat = 1e-4, 
        checkpoint_interval = 1000, 
        stagnation_amplitude= 0.0001, 
        stagnation_range = 200
    ):

        self.model = model
        self.original_input_size = original_input_size
        self.original_hidden_layers = original_hidden_layers
        self.original_output_size = original_output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_interval = checkpoint_interval
        self.stagnation_amplitude = stagnation_amplitude
        self.stagnation_range = stagnation_range

        self.t_interval = t_interval
        
        self.training_ic1_t_mu = ic1_t_mu
        self.training_ic1_scope = ic1_scope
        self.training_ic2_t_mu = ic2_t_mu
        self.training_ic2_scope = ic2_scope
        self.training_physic_in_t_mu = physic_in_t_mu
        self.training_physic_domain_t_mu = physic_domain_t_mu

        self.point_resolution = point_resolution

        self.test_ic1_t_mu = test_ic1_t_mu
        self.test_ic1_scope = test_ic1_scope 
        self.test_ic2_t_mu = test_ic2_t_mu 
        self.test_ic2_scope = test_ic2_scope 
        self.test_physic_in_t_mu = test_physic_in_t_mu 
        self.test_physic_domain_t_mu = test_physic_domain_t_mu
        self.test_point_resolution = test_point_resolution
        
        #self.test_in_t_constant_mu = test_in_t_constant_mu
        
        self.test_mu = test_mu
        self.lambda_ic_dudt = lambda_ic_dudt
        self.lambda_diff_equat = lambda_diff_equat

        self.training_loss_ic1_history = []
        self.training_loss_ic2_history = []
        self.training_loss_differential_equation_history = []
        self.training_loss_total_history = []
        self.training_loss_history = []

        self.test_loss_ic1_history = []
        self.test_loss_ic2_history = []
        self.test_loss_differential_equation_history = []
        self.test_loss_total_history = []
        self.test_loss_history = []

        self.max_stagnation_amplitude_history = []

        self.mass = mass 
        self.w0 = w0
        self.k = self.mass*self.w0**2

        self.u_exact = u_exact(self.test_mu/(2*self.mass), self.w0, self.test_physic_in_t_mu[0])
        
        #self.u_exact()
        
        self.train()
        # self.plot_weights(fig_size = (10,5), font_size = 8) ## plot_weights are defined in class FCN and not in class TrainModel

    def compute_loss(self, ic1_t_mu, ic1_scope, ic2_scope,  physic_domain_t_mu):
        # compute initial condition 1 loss:
        ic1_predicted= self.model(ic1_t_mu)        
        residuals_ic1 = (ic1_predicted - ic1_scope)**2        
        loss_ic1 = torch.mean(residuals_ic1)
    
        # compute initial condition 2 loss:
        du_dtdmu_initial = torch.autograd.grad(outputs = ic1_predicted, inputs = ic1_t_mu, grad_outputs= torch.ones_like(ic1_predicted), create_graph= True)[0]
        ic2_du_dt, ic2_du_dmu = du_dtdmu_initial[:, 0:1], du_dtdmu_initial[:,1:2]
        
        residuals_ic2 = (ic2_du_dt- ic2_scope)**2        
        loss_ic2 = torch.mean(residuals_ic2)
    
        # compute physic loss:
        physic_domain_predicted = self.model(physic_domain_t_mu)
        physic_domain_du_dtdmu = torch.autograd.grad(outputs = physic_domain_predicted, inputs = physic_domain_t_mu, grad_outputs= torch.ones_like(physic_domain_predicted), create_graph= True)[0]
        physic_domain_d2u_d2t_d2mu = torch.autograd.grad(outputs = physic_domain_du_dtdmu[:,0:1], inputs = physic_domain_t_mu, grad_outputs= torch.ones_like(physic_domain_du_dtdmu[:,0:1]), create_graph= True)[0]
        
        residuals_differential_equation = ((1/(self.mass*self.k))*(self.mass * physic_domain_d2u_d2t_d2mu[:,0:1] + physic_domain_t_mu[:,1:2] * physic_domain_du_dtdmu[:,0:1] + self.k * physic_domain_predicted))**2
        
        loss_differential_equation = torch.mean( residuals_differential_equation)
        
        # compute total loss:
        loss = loss_ic1 + self.lambda_ic_dudt * loss_ic2 + self.lambda_diff_equat * loss_differential_equation
        
        return loss, loss_differential_equation, loss_ic1, loss_ic2, residuals_differential_equation, residuals_ic1, residuals_ic2

    # def compute_loss(self, ic1_t_mu, ic1_scope, ic2_scope,  physic_domain_t_mu):
    #     # compute initial condition 1 loss:
    #     self.ic1_predicted= self.model(self.ic1_t_mu)        
    #     self.residuals_ic1 = self.ic1_predicted - self.ic1_scope        
    #     self.loss_ic1 = torch.mean((self.residuals_ic1)**2)
    
    #     # compute initial condition 2 loss:
    #     self.du_dtdmu_initial = torch.autograd.grad(outputs = self.ic1_predicted, inputs = self.ic1_t_mu, grad_outputs= torch.ones_like(self.ic1_predicted), create_graph= True)[0]
    #     self.ic2_du_dt, ic2_du_dmu = self.du_dtdmu_initial[:, 0:1], self.du_dtdmu_initial[:,1:2]
        
    #     self.residuals_ic2 = self.ic2_du_dt- self.ic2_scope        
    #     self.loss_ic2 = torch.mean((self.residuals_ic2)**2)
    
    #     # compute physic loss:
    #     self.physic_domain_predicted = self.model(self.physic_domain_t_mu)
    #     self.physic_domain_du_dtdmu = torch.autograd.grad(outputs = self.physic_domain_predicted, inputs = self.physic_domain_t_mu, grad_outputs= torch.ones_like(self.physic_domain_predicted), create_graph= True)[0]
    #     self.physic_domain_d2u_d2t_d2mu = torch.autograd.grad(outputs = self.physic_domain_du_dtdmu[:,0:1], inputs = self.physic_domain_t_mu, grad_outputs= torch.ones_like(self.physic_domain_du_dtdmu[:,0:1]), create_graph= True)[0]
        
    #     self.residuals_differential_equation = self.physic_domain_d2u_d2t_d2mu[:,0:1] + self.physic_domain_t_mu[:,1:2] * self.physic_domain_du_dtdmu[:,0:1] + self.k * self.physic_domain_predicted 
        
    #     self.loss_differential_equation = torch.mean( (self.residuals_differential_equation)**2)
        
    #     # compute total loss:
    #     self.loss = self.loss_ic1 + self.lambda_ic_dudt * self.loss_ic2 + self.lambda_diff_equat * self.loss_differential_equation
        
    #     return self.loss, self.loss_differential_equation, self.loss_ic1, self.loss_ic2, self.residuals_differential_equation, self.residuals_ic1, self.residuals_ic2
        

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr= self.learning_rate)

        start_time = time.time()
        
        # Predicted values using a constant mu:
        self.test_in_t_constant_mu = torch.stack([torch.linspace(self.t_interval[0], self.t_interval[1], self.test_point_resolution), self.test_mu*torch.ones(self.test_point_resolution)], -1).view(-1,2)
        
        #self.test_predicted = self.model(self.test_in_t_constant_mu)
        print("+"*200)
        print(f"Initial state using a constant mu({self.test_mu})")
        self.plot_constant_mu(fig_size = (10,2.5))
        print("+"*200)

        for self.i in range(1, self.num_epochs + 1):
            
            # TRAINING MODE for TRAINING LOSS
            
            self.model.train()  # Set model to training mode
            optimizer.zero_grad() # clears the gradients of all optimized parameters. Gradients accumulated from the previous iteration are reset to zero.
            
            # Computation of TRAINING LOSS:
            self.training_loss, self.training_loss_differential_equation, self.training_loss_ic1, self.training_loss_ic2, self.training_residuals_differential_equation, self.training_residuals_ic1, self.training_residuals_ic2 = self.compute_loss(self.training_ic1_t_mu, self.training_ic1_scope, self.training_ic2_scope, self.training_physic_domain_t_mu)

            # For plotting the history:
            self.training_loss_ic1_history.append(self.training_loss_ic1.item())
            self.training_loss_ic2_history.append(self.training_loss_ic2.item())
            self.training_loss_differential_equation_history.append(self.training_loss_differential_equation.item())
            self.training_loss_total_history.append(self.training_loss.item())
            self.training_loss_history.append(self.training_loss.item())
            
            if self.i ==1:                
                self.initial_loss_value = self.training_loss.item()        
            
            self.training_loss.backward() # computes the gradients of the loss with respect to all the learnable parameters of the model (backpropagation)
            optimizer.step()              # updates the parameters of the model based on the computed gradients and the optimization algorithm (e.g., Adam)

            # EVALUATION MODE for TEST LOSS
            #with torch.no_grad():        # disables gradient calculation globally within its context. 
            self.model.eval()  # Set model to evaluation mode
            
                #pde_loss_test = self.compute_pde_loss(x_test, t_test)
                #boundary_loss_test = self.compute_boundary_loss(x_test[0:2], x_test[2:4], t_test[0:2], t_test[2:4])
                #loss_test = pde_loss_test + boundary_loss_test
                # self.test_losses.append(loss_test.item())
                                                                                                                                                                                   #test_ic2_t_mu, test_physic_in_t_mu 
            # Computation of TEST LOSS
            self.test_loss, self.test_loss_differential_equation, self.test_loss_ic1, self.test_loss_ic2, self.test_residuals_differential_equation, self.test_residuals_ic1, self.test_residuals_ic2 = self.compute_loss(self.test_ic1_t_mu, self.test_ic1_scope, self.test_ic2_scope, self.test_physic_domain_t_mu)
            # For plotting the test loss history:
            self.test_loss_ic1_history.append(self.test_loss_ic1.item())
            self.test_loss_ic2_history.append(self.test_loss_ic2.item())
            self.test_loss_differential_equation_history.append(self.test_loss_differential_equation.item())
            self.test_loss_total_history.append(self.test_loss.item())
            self.test_loss_history.append(self.test_loss.item())

            
            # Update loss history and ensure it contains the losses of the last check_range epochs
            if len(self.training_loss_history) and len(self.test_loss_history) > self.stagnation_range:
                self.training_loss_history.pop(0)  # Remove the oldest training loss value
                self.test_loss_history.pop(0)  # Remove the oldest test loss value

            # Check if the difference between max and min loss in the last 100 epochs is within the threshold
            self.max_stagnation_amplitude = max(self.training_loss_history) - min(self.training_loss_history)
            self.max_stagnation_amplitude_history.append(self.max_stagnation_amplitude)
            self.absolut_loss_value = sum(self.training_loss_history) / len(self.training_loss_history)

            if (len(self.training_loss_history) == self.stagnation_range) and (self.max_stagnation_amplitude <= self.stagnation_amplitude)  and  (self.absolut_loss_value < self.initial_loss_value):
                print(f"Stopping training at epoch {self.i} as the loss stabilized within the threshold.")
                print(f"max_stagnation_amplitude = {self.max_stagnation_amplitude} \n absolute_loss_value: {self.absolut_loss_value} ")
                break
            
            if self.i%self.checkpoint_interval == 0:
                print("#"*200)
                print(f"PLOTTING THE RESULTS FOR EPOCH: {self.i}")
                #self.plot_results(self.training_residuals_ic1, self.training_residuals_ic2, self.training_residuals_differential_equation, self.training_loss_ic1_history, self.training_loss_ic2_history, self.training_loss_differential_equation_history, self.training_loss_total_history, self.training_loss_history)
                self.plot_results(fig_size_loss = (10,5),  fig_size_residuals = (10,5))

                torch.save({
                            "epoch": self.i,
                            "model_state_dict": self.model.state_dict(),
                            "optimiser_state_dict": optimizer.state_dict(),
                            "loss": self.training_loss,
                           },                    
                            f"lr{self.learning_rate}_epoch{self.i}.pth")
                print(f"Saved the checkpoint corresponding to epoch: {self.i}")
                print(f"RESULTS PLOTTED FOR EPOCH: {self.i}")
                print("#"*200)
        # Plot of final results:
         #self.plot_results(self.training_residuals_ic1, self.training_residuals_ic2, self.training_residuals_differential_equation, self.training_loss_ic1_history, self.training_loss_ic2_history, self.training_loss_differential_equation_history, self.training_loss_total_history, self.training_loss_history)
        if self.i % self.checkpoint_interval != 0:
            print("*"*200)
            print(f"PLOTTING THE RESULTS FOR LAST EPOCH: {self.i}")
            self.plot_results(fig_size_loss = (10,5),  fig_size_residuals = (10,5))

        # Elapsed time:
        end_time = time.time()
        execution_time = (end_time - start_time)
        print(f"Training elapsed time (s): {execution_time}")
        
        # Final checkpoint:
        torch.save({
                    "epoch": self.i,
                    "model_state_dict": self.model.state_dict(),
                    "optimiser_state_dict": optimizer.state_dict(),
                    "loss": self.training_loss,
                    },                    
                    f"original_model.pth")
        print(f"Saved the checkpoint corresponding to last epoch: {self.i}")

    def plot_constant_mu(self, fig_size = (10,2.5)):
        
        self.test_predicted = self.model(self.test_in_t_constant_mu)
        
        self.fig_size = fig_size
        plt.figure(figsize=self.fig_size)
        plt.plot(
                self.test_physic_in_t_mu[0].detach(), 
                self.u_exact.detach(), 
                label="Exact solution", 
                color="tab:grey", 
                alpha=0.6
        )
        plt.plot(
                self.test_physic_in_t_mu[0].detach(), 
                self.test_predicted[:,0].detach(), 
                label="PINN solution (initial)", 
                color="tab:green"
        )
        plt.scatter(
                self.training_physic_in_t_mu[0].detach(), 
                torch.zeros_like(self.training_physic_in_t_mu[0]).detach(), 
                s=20, 
                lw=0, 
                color="tab:red",
                alpha=0.6,
                label= f"Training points ({self.point_resolution})"
        )
        plt.scatter(
                self.test_physic_in_t_mu[0].detach(), 
                torch.zeros_like(self.test_physic_in_t_mu[0]).detach(), 
                s=20, 
                lw=0, 
                color="tab:blue",
                alpha=0.6,
                label=  f"Test points ({self.test_point_resolution})"
        )
        ### model and activation has to be manually adapted
        # original_input_size, self.original_hidden_layers, original_output_size
        #plt.title(f"Exact and predicted solution for a nn with following architecture: [{self.model.original_input_size}, {self.model.original_hidden_layers}, {self.model.original_output_size}] \n u(t=(0,1), $\mu$ = {self.test_mu}), model: {self.model.__class__.__name__}, activation function: {self.model.activation}, epoch = 1")
        #plt.title(f"Exact and predicted solution for a nn with following architecture: [X, Y, Z] \n u(t=(0,1), $\mu$ = {self.test_mu}), model: {self.model.__class__.__name__}, activation function: {self.model.activation}, epoch = 1")
        plt.title(f"Exact and predicted solution \n (NN Architecture: [{self.original_input_size}, {self.original_hidden_layers}, {self.original_output_size}]) \n u(t=(0,1), $\mu$ = {self.test_mu}), model: {self.model.__class__.__name__}, activation function: {self.model.activation}")
        plt.xlabel('t [s]')
        plt.ylabel('u [m]')
        plt.grid()
        plt.legend()
        plt.show()
        

    
#    def plot_results(self, residuals_ic1, residuals_ic2, residuals_differential_equation, loss_ic1_history, loss_ic2_history, loss_differential_equation_history, loss_total_history, loss_history, fig_size_loss = (10,5),  fig_size_residuals = (10,5)):
    def plot_results(self, fig_size_loss = (10,5),  fig_size_residuals = (10,5)):
        self.fig_size_loss = fig_size_loss
        self.fig_size_residuals = fig_size_residuals
        
        print(f"Maximum stagnation amplitude by the loss convergence: {self.max_stagnation_amplitude}")
        print(f'Decomposition of the TRAINING loss terms: \n loss({self.training_loss}) = loss1({self.training_loss_ic1}) + {self.lambda_ic_dudt} * loss2({self.training_loss_ic2}) + {self.lambda_diff_equat} * loss3({self.training_loss_differential_equation})')
        print(f'Decomposition of the TEST loss terms: \n loss({self.test_loss}) = loss1({self.test_loss_ic1}) + {self.lambda_ic_dudt} * loss2({self.test_loss_ic2}) + {self.lambda_diff_equat} * loss3({self.test_loss_differential_equation})')

        self.plot_constant_mu(fig_size = (10,2.5))
        
        # plt.figure(figsize = self.fig_size_loss)
        # plt.plot(self.test_physic_in_t_mu[0].detach().numpy(), 
        #          self.u_exact.detach().numpy(), 
        #          label="Exact solution", 
        #          color="tab:grey", 
        #          alpha=0.6
        # )
        # plt.plot(
        #         self.test_physic_in_t_mu[0].detach().numpy(), 
        #         self.test_predicted[:,0].detach().numpy(), 
        #         label="PINN solution", 
        #         color="tab:green"
        # )
        # plt.scatter(
        #             self.training_physic_in_t_mu[0].detach().numpy(), 
        #             torch.zeros_like(self.training_physic_in_t_mu[0]), 
        #             s=20, 
        #             lw=0, 
        #             color="tab:red",
        #             alpha=0.6,
        #             label= "Training points"
        # )
        # plt.scatter(
        #             self.test_physic_in_t_mu[0].detach().numpy(), 
        #             torch.zeros_like(self.test_physic_in_t_mu[0]), 
        #             s=20, 
        #             lw=0, 
        #             color="tab:green",
        #             alpha=0.6,
        #             label= "Testing points"
        # )
        # #plt.title(f"Exact and predicted solution \n u(t=(0,1), $\mu$ = {test_mu}), model: {self.model.__class__.__name__}, activation function: {self.model.activation()}, epoch = {self.i} \n (learning rate: {learning_rate}, lambda_ic_dudt: {lambda_ic_dudt}, lambda_diff_equat: {lambda_diff_equat})")
        # plt.title(f"Exact and predicted solution \n u(t=(0,1), $\mu$ = {self.test_mu}), model: {self.model.__class__.__name__}, activation function: Tanh, epoch = {self.i} \n (learning rate: {self.learning_rate}, lambda_ic_dudt: {self.lambda_ic_dudt}, lambda_diff_equat: {self.lambda_diff_equat})")
        # plt.grid()
        # plt.legend()
        # plt.show()
        
        ######################################
        # # Loss history:
        # # Plot the loss history as before
        # fig, axs = plt.subplots(nrwos = 1, ncols = 4, figsize=(5, 3))
        # #plt.figure(figsize=(5, 3))
        # plt.plot(self.training_loss_history, label= f'Training Loss' )
        # plt.plot(self.test_loss_history, label= f'Test Loss' )
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title(f'Training and Test Loss: last {self.stagnation_range} epochs')
        # plt.legend()
        # plt.show()
        # # 
        # plt.figure(figsize=(5, 3))
        # plt.plot(self.training_loss_total_history, label='Training Loss')
        # plt.plot(self.test_loss_total_history, label='Test Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Total Loss history: whole history')
        # plt.legend()
        # plt.show()
        # ######################################
        # # max_stagnation_amplitude
        # plt.figure(figsize=(5, 3))
        # plt.plot(self.max_stagnation_amplitude_history)
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss amplitude')
        # plt.title('Max. Loss amplitude: whole history')
        # plt.show()


        ######################################
        ######################################
        # Total Loss history:
        # Plot the loss history as before
        fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize=(20, 5))
        #plt.figure(figsize=(5, 3))
        fig.suptitle(f"Total Loss history (epoch: {self.i})", fontsize = 14)

        axs[0].plot(self.training_loss_total_history, label='Training Loss')
        axs[0].plot(self.test_loss_total_history, label='Test Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Total Loss')
        axs[0].set_title('Total Loss history: whole history')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(self.training_loss_total_history[-1000:], label='Training Loss')
        axs[1].plot(self.test_loss_total_history[-1000:], label='Test Loss')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Total Loss')
        axs[1].set_title(f'Total Training and Test Loss: last 1000 epochs')
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(self.training_loss_history, label= 'Training Loss' )
        axs[2].plot(self.test_loss_history, label= 'Test Loss' )
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Total Loss')
        axs[2].set_title(f'Total Training and Test Loss: last {self.stagnation_range} epochs')
        axs[2].legend()
        axs[2].grid(True)
        ######################################
        ######################################
        # Maximum stagnation amplitude by the convergence
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(20, 5))
        #plt.figure(figsize=(5, 3))
        fig.suptitle(f" Maximum stagnation amplitude by the convergence (epoch: {self.i})", fontsize = 14)
        # max_stagnation_amplitude
        axs[0].plot(self.max_stagnation_amplitude_history)
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss amplitude')
        axs[0].set_title('Maximum stagnation amplitude of the Total Loss: whole history')
        axs[0].grid(True)
        # max_stagnation_amplitude
        axs[1].plot(self.max_stagnation_amplitude_history[-100:])
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss amplitude')
        axs[1].set_title(f'Maximum stagnation amplitude of the Total Loss: last {self.stagnation_range} epochs')
        axs[1].grid(True)
        ######################################      
        ######################################
        # plot the contribution of every loss term (loss1, loss2 and loss3)
        fig, axs = plt.subplots(nrows= 2, ncols= 2, layout = 'constrained', sharex = False, figsize = (20,10))
        #fig.suptitle(f"Decomposition of the loss terms using {original_model.__class__.__name__} model and Tanh #{original_model.activation()}# activation function \n (learning_rate: {learning_rate}, lambda_ic_dudt: {lambda_ic_dudt}, lambda_diff_equat: {lambda_diff_equat})", fontsize = 14)
        fig.suptitle(f"Decomposition of the loss terms using {self.model.__class__.__name__} model and Tanh activation function \n (learning_rate: {self.learning_rate}, lambda_ic_dudt: {self.lambda_ic_dudt}, lambda_diff_equat: {self.lambda_diff_equat}, (epoch: {self.i})", fontsize = 14)
        
        axs[0,0].plot(self.training_loss_ic1_history, '-', label = "training loss1: residuals of u(t=0)=0", color = "tab:red")
        axs[0,0].plot(self.training_loss_ic2_history, '-', label = "training loss2: residuals of du/dt(t=0)=0", color = "tab:blue")
        axs[0,0].plot(self.test_loss_ic1_history, '--', label = "test loss1: residuals of u(t=0)=1", color = "tab:red")
        axs[0,0].plot(self.test_loss_ic2_history, '--', label = "test loss2: residuals of du/dt(t=0)=0", color = "tab:blue")
        axs[0,0].set_title("Training and test loss1 (u(t=0)=1) and loss2 (du/dt(t=0)=0)")
        axs[0,0].set_xlabel("Epochs")
        #axs[0,0].set_xlim(0,1000)
        axs[0,0].set_ylabel("Loss")
        axs[0,0].grid()
        axs[0,0].legend()

        axs[0,1].plot(self.training_loss_ic1_history[-100:], '-', label = "training loss1: residuals of u(t=0)=1", color = "tab:red")
        axs[0,1].plot(self.training_loss_ic2_history[-100:], '-', label = "training loss2: residuals of du/dt(t=0)=0", color = "tab:blue")
        axs[0,1].plot(self.test_loss_ic1_history[-100:], '--', label = "test loss1: residuals of u(t=0)=1", color = "tab:red")
        axs[0,1].plot(self.test_loss_ic2_history[-100:], '--', label = "test loss2: residuals of du/dt(t=0)=0", color = "tab:blue")
        axs[0,1].set_title("Training and test loss1: (u(t=0)=1) and training and test loss2: (du/dt(t=0)=0) (last 100 epochs)")
        axs[0,1].set_xlabel("Epochs")
        axs[0,1].set_ylabel("Loss")
        axs[0,1].set_xlim(0,100)
        axs[0,1].grid()
        axs[0,1].legend()
        
        axs[1,0].plot(self.training_loss_differential_equation_history, '-', label= "training loss3: residuals of the differential equation", color = "tab:grey")
        axs[1,0].plot(self.test_loss_differential_equation_history, '--', label= "test loss3: residuals of the differential equation", color = "tab:grey")
        axs[1,0].set_title("Training and test loss3: residuals of the differential equation (whole history)")
        axs[1,0].set_xlabel("Epochs")
        axs[1,0].set_ylabel("Loss")
        axs[1,0].legend()
        axs[1,0].grid()

        axs[1,1].plot(self.training_loss_differential_equation_history[-100:], '-', label= "training loss3: residuals of the differential equation", color = "tab:grey")
        axs[1,1].plot(self.test_loss_differential_equation_history[-100:], '--', label= "test loss3: residuals of the differential equation", color = "tab:grey")
        axs[1,1].set_title("Training and test loss3: residuals of the differential equation (last 100 epochs)")
        axs[1,1].set_xlabel("Epochs")
        axs[1,1].set_ylabel("Loss")
        axs[1,1].set_xlim(0,100)
        axs[1,1].legend()
        axs[1,1].grid()
        
        # # plot the contribution of every loss term (loss1, loss2 and loss3)
        # fig, (loss1_2, loss3) = plt.subplots(1,2, layout = 'constrained', sharex = True, figsize = (12,5))
        # #fig.suptitle(f"Decomposition of the loss terms using {original_model.__class__.__name__} model and Tanh #{original_model.activation()}# activation function \n (learning_rate: {learning_rate}, lambda_ic_dudt: {lambda_ic_dudt}, lambda_diff_equat: {lambda_diff_equat})", fontsize = 14)
        # fig.suptitle(f"Decomposition of the loss terms using {self.model.__class__.__name__} model and Tanh activation function \n (learning_rate: {self.learning_rate}, lambda_ic_dudt: {self.lambda_ic_dudt}, lambda_diff_equat: {self.lambda_diff_equat})", fontsize = 14)
        
        # loss1_2.plot(self.training_loss_ic1_history, '-', label = "training loss1: residuals of u(t=0)=1", color = "tab:red")
        # loss1_2.plot(self.training_loss_ic2_history, '-', label = "training loss2: residuals of du/dt(t=0)=0", color = "tab:blue")
        # loss1_2.plot(self.test_loss_ic1_history, '--', label = "test loss1: residuals of u(t=0)=1", color = "tab:red")
        # loss1_2.plot(self.test_loss_ic2_history, '--', label = "test loss2: residuals of du/dt(t=0)=0", color = "tab:blue")
        # loss1_2.set_title("training and test loss1: (u(t=0)=1) and training and test loss2: (du/dt(t=0)=0)")
        # loss1_2.set_xlabel("epochs")
        # loss1_2.set_ylabel("residuals")
        # loss1_2.grid()
        # loss1_2.legend()
        
        # loss3.plot(self.training_loss_differential_equation_history, '-', label= "training loss3: residuals of the differential equation", color = "tab:grey")
        # loss3.plot(self.test_loss_differential_equation_history, '--', label= "test loss3: residuals of the differential equation", color = "tab:grey")
        # loss3.set_title("training and test loss3: residuals of the differential equation")
        # loss3.set_xlabel("epochs")
        # loss3.set_ylabel("residuals")
        # loss3.legend()
        # loss3.grid()        
        
        ######################################      
        ######################################
    #     # RESIDUAL DISTRIBUTION in the T-MU SUBSPACE (CONTOURF):
    # #def plot_pinn_points (self, ic1_t_mu, ic2_t_mu, physic_domain_t_mu, self.point_resolution, test_ic1_t_mu, test_ic2_t_mu, test_physic_domain_t_mu, self.test_point_resolution, figsize = (15,7.5)):
    #     ## Plotting initial conditions training and test points:
    #     plt.figure(figsize = fig_size_residuals)
        
    #     # Contour plot
    #     contour = plt.contourf(physic_domain_t_mu[:,0].reshape(self.point_resolution,self.point_resolution).detach().numpy(), physic_domain_t_mu[:,1].reshape(self.point_resolution,self.point_resolution).detach().numpy(), self.training_residuals_differential_equation.reshape(self.point_resolution,self.point_resolution).detach().numpy(), cmap='hot', levels = 40)  # Using 'hot' colormap for heatmap
    #     #plt.colorbar(contour)  # Add color bar for reference
    #     # Create a colorbar associated with the contour plot
    #     cbar = plt.colorbar(contour)
    #     # Set the colorbar ticks to match the contour levels
    #     cbar.set_ticks(contour.levels)
    #     ##plt.xlabel('t')
    #     ##plt.ylabel('mu')
    #     ##plt.title(f'Location of initial conditions, training points ({self.point_resolution}) and test points ({self.test_point_resolution})')
        
    #     # plt.scatter(
    #     #             ic1_t_mu[:, 0].detach(), 
    #     #             ic1_t_mu[:, 1].detach(), 
    #     #             color="blue",
    #     #             label="Initial condition 1",
    #     #             #color="tab:grey",
    #     #             marker = 'x',
    #     #             alpha=0.6,
    #     #             s=20
    #     #             )
    #     # plt.scatter(
    #     #             ic2_t_mu[:, 0].detach(), 
    #     #             ic2_t_mu[:, 1].detach(), 
    #     #             color="red",
    #     #             label="Initial condition 2",
    #     #             #color="tab:grey",
    #     #             marker = '+',
    #     #             alpha=0.6,
    #     #             s=20
    #     #             )           
    #     # plt.scatter(
    #     #             physic_domain_t_mu[:, 0].detach(), 
    #     #             physic_domain_t_mu[:, 1].detach(), 
    #     #             color="black",
    #     #             label="Training points",
    #     #             #color="tab:grey",
    #     #             marker = "D",
    #     #             alpha=0.6,
    #     #             s=20
    #     #             ) 
    #     # #############################################################
    #     # plt.scatter(
    #     #             test_ic1_t_mu[:, 0].detach(), 
    #     #             test_ic1_t_mu[:, 1].detach(), 
    #     #             color="red",
    #     #             label="Test Initial condition 1",
    #     #             #color="tab:grey",
    #     #             marker = 's',
    #     #             alpha=0.6,
    #     #             s=20
    #     #             )
    #     # plt.scatter(
    #     #             test_ic2_t_mu[:, 0].detach(), 
    #     #             test_ic2_t_mu[:, 1].detach(), 
    #     #             color="red",
    #     #             label="Test Initial condition 2",
    #     #             #color="tab:grey",
    #     #             marker = 's',
    #     #             alpha=0.6,
    #     #             s=20
    #     #             )           
    #     # plt.scatter(
    #     #             test_physic_domain_t_mu[:, 0].detach(), 
    #     #             test_physic_domain_t_mu[:, 1].detach(), 
    #     #             color="green",
    #     #             label="Test points",
    #     #             #color="tab:grey",
    #     #             marker = "s",
    #     #             alpha=0.6,
    #     #             s=20
    #     #             ) 
        
    #     plt.xlabel('t')
    #     plt.ylabel(f'$\mu$')
    #     plt.title(f'Distribution of the Residuals**2 \n and location of initial conditions points, training points ({self.point_resolution}) and test points ({self.test_point_resolution}) \n (epoch: {self.i})')
    #     #plt.legend(loc = "upper right")
    #     plt.grid(True)
    #     plt.show()

    #     #----------------------
    #     plt.figure(figsize = fig_size_residuals)
        
    #     # Contour plot
    #     contour = plt.contourf(test_physic_domain_t_mu[:,0].reshape(self.test_point_resolution, self.test_point_resolution).detach().numpy(), test_physic_domain_t_mu[:,1].reshape(self.test_point_resolution, self.test_point_resolution).detach().numpy(), self.test_residuals_differential_equation.reshape(self.test_point_resolution, self.test_point_resolution).detach().numpy(), cmap='hot', levels = 40)  # Using 'hot' colormap for heatmap
    #     #plt.colorbar(contour)  # Add color bar for reference
    #     # Create a colorbar associated with the contour plot
    #     cbar = plt.colorbar(contour)
    #     # Set the colorbar ticks to match the contour levels
    #     cbar.set_ticks(contour.levels)
    #     plt.xlabel('t')
    #     plt.ylabel(f'$\mu$')
    #     plt.title(f'Distribution of the Residuals**2 \n and location of initial conditions points, training points ({self.point_resolution}) and test points ({self.test_point_resolution}) \n (epoch: {self.i})')
    #     #plt.legend(loc = "upper right")
    #     plt.grid(True)
    #     plt.show()

         # Plotting initial conditions training and test points:
        fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (25,20)) # fig_size_residuals

        fig.suptitle(f'Distribution of the Residuals**2 and location of initial conditions points, training points ({self.point_resolution}) and test points ({self.test_point_resolution}) \n (epoch: {self.i}) by TRAINING and TEST')
                     
        # Contour plot TRAINING residuals differential equation
        contour_plot_training = ax1.contourf(self.training_physic_domain_t_mu[:,0].reshape(self.point_resolution, self.point_resolution).detach().numpy(), self.training_physic_domain_t_mu[:,1].reshape(self.point_resolution, self.point_resolution).detach().numpy(), self.training_residuals_differential_equation.reshape(self.point_resolution, self.point_resolution).detach().numpy(), cmap='hot', levels = 40)  # Using 'hot' colormap for heatmap
        contour_bar_training = fig.colorbar(contour_plot_training, ax= ax1, orientation = "vertical") # Create a colorbar associated with the contour plot
        contour_bar_training.set_ticks(contour_plot_training.levels)                                  # Set the colorbar ticks to match the contour levels
        ax1.scatter(
                    self.training_ic1_t_mu[:, 0].detach(), 
                    self.training_ic1_t_mu[:, 1].detach(), 
                    color="blue",
                    label="Initial condition 1",
                    #color="tab:grey",
                    marker = 'x',
                    linewidths=2,
                    alpha=0.6,
                    s=20
                    )
        ax1.scatter(
                    self.training_ic2_t_mu[:, 0].detach(), 
                    self.training_ic2_t_mu[:, 1].detach(), 
                    color="red",
                    label="Initial condition 2",
                    #color="tab:grey",
                    marker = '+',
                    linewidths=2,
                    alpha=0.6,
                    s=20
                    )           
        ax1.scatter(
                    self.training_physic_domain_t_mu[:, 0].detach(), 
                    self.training_physic_domain_t_mu[:, 1].detach(), 
                    color="white",
                    label="Training points",
                    #color="tab:grey",
                    marker = "D",
                    alpha=0.6,
                    s=20
                    ) 
        ax1.set_xlabel('t')
        ax1.set_ylabel(f'$\mu$')
        ax1.set_title(f'Residuals of differential equation (training points ({self.point_resolution}))')
        ax1.grid(True)


        # Contour plot TEST residuals differential equation
        contour_plot_test = ax2.contourf(self.test_physic_domain_t_mu[:,0].reshape(self.test_point_resolution, self.test_point_resolution).detach().numpy(), self.test_physic_domain_t_mu[:,1].reshape(self.test_point_resolution, self.test_point_resolution).detach().numpy(), self.test_residuals_differential_equation.reshape(self.test_point_resolution, self.test_point_resolution).detach().numpy(), cmap='hot', levels = 40)  # Using 'hot' colormap for heatmap
        contour_bar_test = fig.colorbar(contour_plot_test, ax= ax2, orientation = "vertical") # Create a colorbar associated with the contour plot
        contour_bar_test.set_ticks(contour_plot_test.levels)                                  # Set the colorbar ticks to match the contour levels
        ax2.scatter(
                    self.test_ic1_t_mu[:, 0].detach(), 
                    self.test_ic1_t_mu[:, 1].detach(), 
                    color="blue",
                    label="Initial condition 1",
                    #color="tab:grey",
                    marker = 'x',
                    linewidths=2,
                    alpha=0.6,
                    s=20
                    )
        ax2.scatter(
                    self.test_ic2_t_mu[:, 0].detach(), 
                    self.test_ic2_t_mu[:, 1].detach(), 
                    color="red",
                    label="Initial condition 2",
                    #color="tab:grey",
                    marker = '+',
                    linewidths=2,
                    alpha=0.6,
                    s=20
                    )           
        ax2.scatter(
                    self.test_physic_domain_t_mu[:, 0].detach(), 
                    self.test_physic_domain_t_mu[:, 1].detach(), 
                    color="white",
                    label="Training points",
                    #color="tab:grey",
                    marker = "D",
                    alpha=0.6,
                    s=20
                    ) 
        ax2.set_xlabel('t')
        ax2.set_ylabel(f'$\mu$')
        ax2.set_title(f'Residuals of differential equation (test points ({self.test_point_resolution}))')
        ax2.grid(True)

        plt.show()


        # # Contour plot
        # ax2.contourf(test_physic_domain_t_mu[:,0].reshape(self.test_point_resolution, self.test_point_resolution).detach().numpy(), test_physic_domain_t_mu[:,1].reshape(self.test_point_resolution, self.test_point_resolution).detach().numpy(), self.test_residuals_differential_equation.reshape(self.test_point_resolution, self.test_point_resolution).detach().numpy(), cmap='hot', levels = 40)  # Using 'hot' colormap for heatmap
        # #plt.colorbar(contour)  # Add color bar for reference
        # # Create a colorbar associated with the contour plot
        # cbar = plt.colorbar(contour)
        # # Set the colorbar ticks to match the contour levels
        # cbar.set_ticks(contour.levels)
        # plt.xlabel('t')
        # plt.ylabel(f'$\mu$')
        # plt.title(f'Distribution of the Residuals**2 \n and location of initial conditions points, training points ({self.point_resolution}) and test points ({self.test_point_resolution}) \n (epoch: {self.i})')
        # #plt.legend(loc = "upper right")
        # plt.grid(True)
        # plt.show()


