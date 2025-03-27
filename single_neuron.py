"""
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.
Input:
features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1
Output:
([0.4626, 0.4134, 0.6682], 0.3349)
"""

import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):

	# output = sigmoid(XW + b)
	# mse = mean(y_pred - y)^2

	output = np.dot(np.array(features), np.array(weights)) + np.array(bias)
	probabilities = 1/(1+ np.exp(-output))
	mse = np.mean((probabilities - labels)**2)

	return np.round(probabilities, 4), np.round(mse,4)
