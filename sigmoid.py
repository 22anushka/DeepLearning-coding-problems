import math

def sigmoid(z: float) -> float:
	# original 1/(1+e^-x)
	# mathematically stable 
	if z > 0:
		return round(1/(1+ math.exp(-z)), 4)

	return round(math.exp(z) / (1+math.exp(z)), 4)
