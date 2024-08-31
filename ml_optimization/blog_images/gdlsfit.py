# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:33:41 2024

@author: Alex Salce
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate some noisy data
np.random.seed(20)
X = 2 * np.random.rand(100, 1)
y = 4 + 2 * X + np.random.randn(100, 1)

# Function to compute the predictions
def predict(X, theta):
    return X.dot(theta)

# Function to compute the cost (mean squared error)
def compute_cost(X, y, theta):
    m = len(y)
    cost = (1/(2*m)) * np.sum(np.square(predict(X, theta) - y))
    return cost

# Function to perform gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))

    for i in range(iterations):
        gradients = 1/m * X.T.dot(predict(X, theta) - y)
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)
        theta_history[i, :] = theta.T
    return theta, cost_history, theta_history

# Prepare data
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_initial = np.random.randn(2, 1)  # random initialization
learning_rate = 0.1
iterations = 100

# Perform gradient descent
theta, cost_history, theta_history = gradient_descent(X_b, y, theta_initial, learning_rate, iterations)

# Find the optimal theta
optimal_theta = np.array([4,2])

# Set up the figure and axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 15))

# Plot settings for the regression line plot
ax1.set_xlim(0, 2)
ax1.set_ylim(0, 15)
ax1.set_xlabel('X')
ax1.set_ylabel('y')
line, = ax1.plot([], [], 'r-', lw=2)
data_scatter = ax1.scatter(X, y)

# Stationary fitted line
stationary_slope = 2
stationary_intercept = 4
x_vals = np.array(ax1.get_xlim())
y_vals = stationary_intercept + stationary_slope * x_vals
ax1.plot(x_vals, y_vals, 'm--', lw=2, label='Optimal Solution')
ax1.legend()

# Plot settings for the error plot
ax2.set_xlim(0, iterations)
ax2.set_ylim(0, max(cost_history) * 1.1)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
error_line, = ax2.plot([], [], 'b-')

# Plot settings for the theta values
ax3.set_xlim(min(theta_history[:, 0]) - 1, max(theta_history[:, 0]) + 1)
ax3.set_ylim(min(theta_history[:, 1]) - 1, max(theta_history[:, 1]) + 1)
ax3.set_xlabel('θ_intercept')
ax3.set_ylabel('θ_slope')
theta_line, = ax3.plot([], [], 'go-')
optimal_point, = ax3.plot(optimal_theta[0], optimal_theta[1], 'm*', markersize=10, label='Optimal Solution')

# Add a dotted circle of radius 0.5 centered at (4, 2)
circle = plt.Circle((4, 2), 1, color='red', fill=False, linestyle='dotted', label='Example Tolerance')
ax3.add_artist(circle)
ax3.legend()

# Animation update function
def update(frame):
    theta = theta_history[frame]
    y_pred = predict(X_b, theta)
    line.set_data(X, y_pred)
    
    error_line.set_data(range(frame+1), cost_history[:frame+1])
    theta_line.set_data(theta_history[:frame+1, 0], theta_history[:frame+1, 1])
    
    return line, error_line, theta_line, optimal_point

# Create the animation
ani = FuncAnimation(fig, update, frames=range(iterations), blit=True)

plt.show()

ani.save('gdlsfit.gif', writer='ffmpeg')


