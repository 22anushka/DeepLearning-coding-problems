import numpy as np
def softmax(scores: list[float]) -> list[float]:
	# exp(x)/sum(exp(scores)) where x in scores
	probabilities = np.exp(scores)/np.sum(np.exp(scores), axis=0)
	return probabilities
