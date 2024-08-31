# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 09:41:51 2024

@author: Alex Salce
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data with noise
np.random.seed(0)
n_samples = 100
X = np.random.rand(n_samples, 2)
true_coefficients = np.array([3.5, -2.0])  # True line coefficients

# Add noise to the data
noise_level = 2
y = X @ true_coefficients + noise_level * np.random.randn(n_samples)

# Implementing the SARAH algorithm with solution path and loss tracking
def sarah_with_path_and_loss(X, y, lr=0.01, n_epochs=100, batch_size=1):
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    
    # Lists to track the solution path and loss function
    solution_path = [coefficients.copy()]
    loss_values = [0.5 * np.mean((X @ coefficients - y) ** 2)]
    
    for epoch in range(n_epochs):
        # Compute the full gradient
        full_gradient = np.mean([X[i] * (X[i] @ coefficients - y[i]) for i in range(n_samples)], axis=0)
        coefficients -= lr * full_gradient
        
        for i in range(0, n_samples, 10):
            batch_indices = np.random.choice(n_samples, batch_size, replace=False)
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Compute batch gradient
            batch_gradient = np.mean([batch_X[j] * (batch_X[j] @ coefficients - batch_y[j]) for j in range(batch_size)], axis=0)
            
            # Update coefficients using SARAH update rule
            coefficients -= lr * (batch_gradient - np.mean([batch_X[j] * (batch_X[j] @ solution_path[-1] - batch_y[j]) for j in range(batch_size)], axis=0) + full_gradient)
            
            # Track the solution path and loss
            solution_path.append(coefficients.copy())
            loss_values.append(0.5 * np.mean((X @ coefficients - y) ** 2))
    
    return np.array(solution_path), loss_values

# Implementing the SVRG algorithm with solution path and loss tracking
def svrg_with_path_and_loss(X, y, lr=0.01, n_epochs=100, batch_size=1):
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    full_gradient = np.mean([X[i] * (X[i] @ coefficients - y[i]) for i in range(n_samples)], axis=0)
    
    # Lists to track the solution path and loss function
    solution_path = [coefficients.copy()]
    loss_values = [0.5 * np.mean((X @ coefficients - y) ** 2)]
    
    for epoch in range(n_epochs):
        coefficients_snapshot = coefficients.copy()
        for i in range(0, n_samples, 10):
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

# Implementing the SGD algorithm with solution path and loss tracking
def sgd_with_path_and_loss(X, y, lr=0.01, n_epochs=100, batch_size=1):
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    
    # Lists to track the solution path and loss function
    solution_path = [coefficients.copy()]
    loss_values = [0.5 * np.mean((X @ coefficients - y) ** 2)]
    
    for epoch in range(n_epochs):
        for i in range(0, n_samples, 10):
            batch_indices = np.random.choice(n_samples, batch_size, replace=False)
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Compute batch gradient
            batch_gradient = np.mean([batch_X[j] * (batch_X[j] @ coefficients - batch_y[j]) for j in range(batch_size)], axis=0)
            
            # Update coefficients using SGD update rule
            coefficients -= lr * batch_gradient
            
            # Track the solution path and loss
            solution_path.append(coefficients.copy())
            loss_values.append(0.5 * np.mean((X @ coefficients - y) ** 2))
    
    return np.array(solution_path), loss_values

# Implementing the ADAM algorithm with solution path and loss tracking
def adam_with_path_and_loss(X, y, lr=0.01, n_epochs=100, batch_size=10, beta1=0.9, beta2=0.999, epsilon=1e-8):
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    m = np.zeros(n_features)
    v = np.zeros(n_features)
    t = 0
    
    # Lists to track the solution path and loss function
    solution_path = [coefficients.copy()]
    loss_values = [0.5 * np.mean((X @ coefficients - y) ** 2)]
    
    for epoch in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            t += 1
            batch_indices = np.random.choice(n_samples, batch_size, replace=False)
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Compute batch gradient
            batch_gradient = np.mean([batch_X[j] * (batch_X[j] @ coefficients - batch_y[j]) for j in range(batch_size)], axis=0)
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * batch_gradient
            
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (batch_gradient ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1 ** t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2 ** t)
            
            # Update coefficients using ADAM update rule
            coefficients -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Track the solution path and loss
            solution_path.append(coefficients.copy())
            loss_values.append(0.5 * np.mean((X @ coefficients - y) ** 2))
    
    return np.array(solution_path), loss_values

# SARAH fitting and tracking solution path and loss
sarah_solution_path, sarah_loss_values = sarah_with_path_and_loss(X, y)

# SVRG fitting and tracking solution path and loss
svrg_solution_path, svrg_loss_values = svrg_with_path_and_loss(X, y)

# SGD fitting and tracking solution path and loss
sgd_solution_path, sgd_loss_values = sgd_with_path_and_loss(X, y)

# ADAM fitting and tracking solution path and loss
adam_solution_path, adam_loss_values = adam_with_path_and_loss(X, y)

# Creating subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plotting the solution path for x1 vs x2 for SARAH, SVRG, SGD, and ADAM in the first subplot
ax1.plot(sarah_solution_path[:, 0], sarah_solution_path[:, 1], label='SARAH Path', color='green', linestyle='-')
ax1.plot(svrg_solution_path[:, 0], svrg_solution_path[:, 1], label='SVRG Path', color='blue')
ax1.plot(sgd_solution_path[:, 0], sgd_solution_path[:, 1], label='SGD Path', color='orange', linestyle='--')
ax1.plot(adam_solution_path[:, 0], adam_solution_path[:, 1], label='ADAM Path', color='purple', linestyle='-.')

ax1.scatter([true_coefficients[0]], [true_coefficients[1]], color='red', marker='x', label='True Coefficients')
ax1.set_xlabel(r'$\theta_1$')
ax1.set_ylabel(r'$\theta_2$')
ax1.legend(loc='upper right')
ax1.grid(True)
ax1.set_title(r'Solution Path in $x_1$ vs $x_2$ Space')

# Plotting the loss function for SARAH, SVRG, SGD, and ADAM in the second subplot
ax2.plot(sarah_loss_values, label='SARAH Loss', color='green')
ax2.plot(svrg_loss_values, label='SVRG Loss', color='blue')
ax2.plot(sgd_loss_values, label='SGD Loss', color='orange', linestyle='--')
ax2.plot(adam_loss_values, label='ADAM Loss', color='purple', linestyle='-.')

ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss Value')
ax2.legend(loc='upper right')
ax2.grid(True)
ax2.set_title('Loss Function using SARAH, SVRG, SGD, and ADAM')

plt.tight_layout()
plt.show()



