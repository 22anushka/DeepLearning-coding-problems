import numpy as np

def sigmoid(x: float):
	if x > 0:
		return 1/ (1+ np.exp(-x))
	return np.exp(x) / (1+np.exp(x))

def swish(x: float) -> float:
	"""
	Implements the Swish activation function.

	Args:
		x: Input value

	Returns:
		The Swish activation value
	"""
	# x * sigmoid(x)
	return x * sigmoid(x)
	
