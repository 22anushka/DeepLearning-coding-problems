import numpy as np
def selu(x: float) -> float:
	"""
	Implements the SELU (Scaled Exponential Linear Unit) activation function.

	Args:
		x: Input value

	Returns:
		SELU activation value
	"""
	alpha = 1.6732632423543772
	scale = 1.0507009873554804
	# given that SELU(x) = lambda* alpha* (e^x -1) for if x <= 0, else lambda*x
	return (scale*alpha)*(np.exp(x)-1) if x <= 0 else scale*x
