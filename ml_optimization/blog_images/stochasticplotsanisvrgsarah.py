# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 11:47:20 2024

@author: Alex Salce
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate synthetic data with noise
np.random.seed(0)
n_samples = 100
n_inner = n_samples*10
X = np.random.rand(n_samples, 2)
true_coefficients = np.array([3.5, -2.0])  # True line coefficients

# Add noise to the data
noise_level = 3
y = X @ true_coefficients + noise_level * np.random.randn(n_samples)

# Implementing the SARAH algorithm with solution path and loss tracking
def sarah_with_path_and_loss(X, y, lr=0.01, n_epochs=50, batch_size=1):
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    
    # Lists to track the solution path and loss function
    solution_path = [coefficients.copy()]
    loss_values = [0.5 * np.mean((X @ coefficients - y) ** 2)]
    
    for epoch in range(n_epochs):
        # Compute the full gradient
        full_gradient = np.mean([X[i] * (X[i] @ coefficients - y[i]) for i in range(n_samples)], axis=0)
        coefficients -= lr * full_gradient
        last_gradient = full_gradient
        
        for i in range(0, n_inner, 10):
            batch_indices = np.random.choice(n_samples, batch_size, replace=False)
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Compute batch gradient
            batch_gradient = np.mean([batch_X[j] * (batch_X[j] @ coefficients - batch_y[j]) for j in range(batch_size)], axis=0)
            
            # Update coefficients using SARAH update rule
            last_gradient = (batch_gradient - np.mean([batch_X[j] * (batch_X[j] @ solution_path[-1] - batch_y[j]) for j in range(batch_size)], axis=0) + last_gradient)
            coefficients -= lr * last_gradient
            
            # Track the solution path and loss
            solution_path.append(coefficients.copy())
            loss_values.append(0.5 * np.mean((X @ coefficients - y) ** 2))
    
    return np.array(solution_path), loss_values

# Implementing the SVRG algorithm with solution path and loss tracking
def svrg_with_path_and_loss(X, y, lr=0.01, n_epochs=50, batch_size=1):
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    full_gradient = np.mean([X[i] * (X[i] @ coefficients - y[i]) for i in range(n_samples)], axis=0)
    
    # Lists to track the solution path and loss function
    solution_path = [coefficients.copy()]
    loss_values = [0.5 * np.mean((X @ coefficients - y) ** 2)]
    
    for epoch in range(n_epochs):
        coefficients_snapshot = coefficients.copy()
        for i in range(0, n_inner, 10):
            batch_indices = np.random.choice(n_samples, batch_size, replace=False)
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Compute batch gradient
            batch_gradient = np.mean([batch_X[j] * (batch_X[j] @ coefficients - batch_y[j]) for j in range(batch_size)], axis=0)
            
            # Update coefficients using SVRG update rule
            coefficients -= lr * (batch_gradient - np.mean([batch_X[j] * (batch_X[j] @ coefficients_snapshot - batch_y[j]) for j in range(batch_size)], axis=0) + full_gradient)
            
            # Track the solution path and loss
            solution_path.append(coefficients.copy())
            loss_values.append(0.5 * np.mean((X @ coefficients - y) ** 2))
        
        # Update the full gradient
        full_gradient = np.mean([X[i] * (X[i] @ coefficients - y[i]) for i in range(n_samples)], axis=0)
    
    return np.array(solution_path), loss_values

# SARAH fitting and tracking solution path and loss
sarah_solution_path, sarah_loss_values = sarah_with_path_and_loss(X, y)

# SVRG fitting and tracking solution path and loss
svrg_solution_path, svrg_loss_values = svrg_with_path_and_loss(X, y)


# Animation for solution path
fig_path, ax_path = plt.subplots(figsize=(8, 6))

line_sarah, = ax_path.plot([], [], 'g-', label='SARAH Path')
line_svrg, = ax_path.plot([], [], 'b-', label='SVRG Path')
true_point, = ax_path.plot([], [], 'rx', label='True Coefficients')

# ax_path.set_xlim(0, 3.75)
ax_path.set_xlim(0, 1)
# ax_path.set_ylim(-2.25, 0.25)
ax_path.set_ylim(-0.25, 0.25)
ax_path.set_xlabel(r'$\theta_1$')
ax_path.set_ylabel(r'$\theta_2$')
ax_path.legend(loc='upper right')
ax_path.grid(True)
ax_path.set_title(r'Solution Path in $x_1$ vs $x_2$ Space')

def init_path():
    line_sarah.set_data([], [])
    line_svrg.set_data([], [])
    true_point.set_data([true_coefficients[0]], [true_coefficients[1]])
    return line_sarah, line_svrg, true_point

def update_path(frame):
    sarah_data = sarah_solution_path[:frame+1]
    svrg_data = svrg_solution_path[:frame+1]
    
    line_sarah.set_data(sarah_data[:, 0], sarah_data[:, 1])
    line_svrg.set_data(svrg_data[:, 0], svrg_data[:, 1])
    
    return line_sarah, line_svrg, true_point

ani_path = animation.FuncAnimation(fig_path, update_path, frames=len(sarah_solution_path), init_func=init_path, blit=True, interval=100)
ani_path.save('stochasticgds.gif', writer='ffmpeg', fps=50)
# ani_path.save('stochasticgdssarahsvrg.gif', writer='MovieWriter', fps=100)

# Animation for loss plot
# fig_loss, ax_loss = plt.subplots(figsize=(8, 6))

# line_sarah_loss, = ax_loss.plot([], [], 'g-', label='SARAH Loss')
# line_svrg_loss, = ax_loss.plot([], [], 'b-', label='SVRG Loss')
# line_sgd_loss, = ax_loss.plot([], [], 'y-', label='SGD Loss')
# line_adam_loss, = ax_loss.plot([], [], 'm-.', label='ADAM Loss')

# ax_loss.set_xlim(0, len(sarah_loss_values))
# ax_loss.set_ylim(0, max(max(sarah_loss_values), max(svrg_loss_values), max(sgd_loss_values), max(adam_loss_values)) + 1)
# ax_loss.set_xlabel('Iteration')
# ax_loss.set_ylabel('Loss Value')
# ax_loss.legend(loc='upper right')
# ax_loss.grid(True)
# ax_loss.set_title('Loss Function over Iterations')

# def init_loss():
#     return line_sarah_loss, line_svrg_loss, line_sgd_loss, line_adam_loss

# def update_loss(frame):
#     ax_loss.clear()
#     ax_loss.plot(sarah_loss_values[:frame+1], label='SARAH Loss', color='green')
#     ax_loss.plot(svrg_loss_values[:frame+1], label='SVRG Loss', color='blue')
#     ax_loss.plot(sgd_loss_values[:frame+1], label='SGD Loss', color='orange', linestyle='--')
#     ax_loss.plot(adam_loss_values[:frame+1], label='ADAM Loss', color='purple', linestyle='-.')
#     ax_loss.set_xlabel('Iteration')
#     ax_loss.set_ylabel('Loss Value')
#     ax_loss.legend(loc='upper right')
#     ax_loss.grid(True)
#     ax_loss.set_title('Loss Function over Iterations')
#     return ax_loss

# ani_loss = animation.FuncAnimation(fig_loss, update_loss, frames=len(sarah_loss_values), init_func=init_loss, blit=True, interval=500)

# ani_loss.save('stochasticgdsloss.gif', writer='ffmpeg')

plt.show()

